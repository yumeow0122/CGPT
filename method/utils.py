"""
Utility functions for file I/O and text processing.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from loguru import logger


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of dictionaries loaded from the file.

    Raises:
        FileNotFoundError: If file_path does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = [json.loads(line) for line in open(file_path, encoding='utf-8')]
    logger.info(f"Loaded {len(data)} items from {file_path.name}")
    return data


def save_jsonl(data: List[Dict[str, Any]], output_path: Path, description: str = "data") -> None:
    """
    Save data to JSONL file.

    Args:
        data: List of dictionaries to save.
        output_path: Output file path. Parent directories will be created if needed.
        description: Description for log message (e.g., "corpus", "trainset").
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.success(f"Saved {len(data)} {description} to {output_path}")


# Aliases for backward compatibility
load_corpus = load_jsonl
save_corpus = lambda chunks, path: save_jsonl(chunks, path, "chunks")
save_trainset = lambda trainset, path: save_jsonl(trainset, path, "samples")


def load_tables(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load tables from input directory.

    Args:
        input_path: Directory path containing table.jsonl file.

    Returns:
        List of table dictionaries loaded from the JSONL file.

    Raises:
        FileNotFoundError: If table.jsonl does not exist in input_path.
    """
    table_file = input_path / "table.jsonl"
    if not table_file.exists():
        raise FileNotFoundError(f"Table file not found: {table_file}")

    tables = [json.loads(line) for line in open(table_file, encoding='utf-8')]
    logger.info(f"Loaded {len(tables)} tables")
    return tables


def clean_text(text: str) -> str:
    """
    Clean text by splitting on commas, removing empty cells, and joining with spaces.

    Args:
        text: Raw text string, typically a comma-separated row.

    Returns:
        Cleaned text with cells joined by single spaces.
    """
    cells = [c.strip() for c in text.split(',') if c.strip()]
    return ' '.join(cells)


def serialize_table(
    file_name: str,
    sheet_name: str,
    header: str,
    instances: List[str]
) -> str:
    """
    Serialize table metadata and instances to text representation.

    Args:
        file_name: Name of the source file.
        sheet_name: Name of the sheet within the file.
        header: Table header as a string.
        instances: List of table row instances.

    Returns:
        Formatted text representation of the table chunk.
    """
    instances_text = '\n'.join(instances)
    return f"""File Name:
{file_name}
Sheet Name
{sheet_name}
Header:
{header}
Instance:
{instances_text}
"""


def call_llm_api(
    endpoint: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int = 120,
    max_retries: int = 5
) -> str:
    """
    Call LLM API with retry mechanism.

    Args:
        endpoint: API endpoint URL.
        model: Model name/identifier.
        prompt: The prompt text to send.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.

    Returns:
        Generated response content, or empty string on failure.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(endpoint, json=payload, timeout=timeout)

            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")

            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            if not content:
                raise Exception("Empty response")

            return content

        except Exception as e:
            logger.warning(f"LLM error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    return ""
