import argparse
import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
from loguru import logger

from config import LLM_ENDPOINT, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT
from prompt import chunk_to_questions
from utils import load_corpus, save_trainset


async def _call_llm_async(
    session: aiohttp.ClientSession,
    prompt: str
) -> str:
    """
    Call LLM API asynchronously.

    Args:
        session: aiohttp client session for making HTTP requests.
        prompt: The prompt text to send to the LLM.

    Returns:
        The generated response content, or empty string on failure.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS
    }

    try:
        async with session.post(LLM_ENDPOINT, json=payload, timeout=aiohttp.ClientTimeout(total=LLM_TIMEOUT)) as resp:
            if resp.status != 200:
                return ""
            result = await resp.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception:
        return ""


async def synthesis_question(
    session: aiohttp.ClientSession,
    chunk_repr: str,
    questions_per_chunk: int,
    lang: str
) -> List[Dict[str, Any]]:
    """
    Generate synthetic questions for a table chunk using LLM.

    Args:
        session: aiohttp client session for making HTTP requests.
        chunk_repr: Text representation of the table chunk.
        questions_per_chunk: Number of questions to generate per chunk.
        lang: Language for generated questions ('zh' or 'en').

    Returns:
        List of training pairs, each containing 'query' and 'pos' keys.
        Returns empty list if generation fails after 5 attempts.
    """
    prompt = chunk_to_questions.format(
        table_chunk=chunk_repr,
        questions_per_chunk=questions_per_chunk,
        lang=lang
    )

    for attempt in range(5):
        response = await _call_llm_async(session, prompt)

        if not response:
            continue

        try:
            # Parse JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            json_str = json_match.group(1) if json_match else response
            data = json.loads(json_str)
            questions = data.get('questions', [])

            if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
                continue

            # Build training pairs
            training_pairs = []
            for q in questions:
                q = q.strip()
                if q:
                    training_pairs.append({
                        'query': q,
                        'pos': [chunk_repr]
                    })

            return training_pairs

        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Parse error (attempt {attempt+1}/5): {e}")
            continue

    logger.warning(f"Failed to generate questions after 5 attempts")
    return []


async def main():
    parser = argparse.ArgumentParser(
        description="Generate Synthetic Questions for Table Chunks"
    )
    parser.add_argument("--corpus", type=Path, required=True, help="Input corpus file (chunk.jsonl)")
    parser.add_argument("--output", type=Path, required=True, help="Output file path")
    parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"], help="Language (default: zh)")
    parser.add_argument("--questions-per-chunk", type=int, default=5, help="Questions per chunk (default: 5)")

    args = parser.parse_args()

    logger.info("Loading corpus...")
    chunks = load_corpus(args.corpus)

    logger.info(f"Generating training data for {len(chunks)} chunks...")

    async with aiohttp.ClientSession() as session:
        tasks = [
            synthesis_question(session, chunk['representation'], args.questions_per_chunk, args.lang)
            for chunk in chunks
        ]

        results = []
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            result = await task
            results.append(result)
            if i % 50 == 0:
                logger.info(f"Completed {i}/{len(tasks)}")

    # Flatten results
    training_data = []
    for pairs in results:
        training_data.extend(pairs)

    logger.info(f"Generated {len(training_data)} training samples")

    # Shuffle and save
    random.shuffle(training_data)
    save_trainset(training_data, args.output)

    # Statistics
    logger.info("Statistics:")
    logger.info(f"  Chunks: {len(chunks)}")
    logger.info(f"  Training samples: {len(training_data)}")


if __name__ == "__main__":
    asyncio.run(main())
