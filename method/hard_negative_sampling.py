import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from loguru import logger

from config import DEFAULT_MODEL_NAME, DEFAULT_BATCH_SIZE
from embedding_model import init_embedding_model
from utils import load_corpus, save_trainset


def sample_hard_negatives(
    pos_chunk_id: int,
    corpus: List[Dict[str, Any]],
    query_emb: np.ndarray,
    corpus_embs: np.ndarray,
    num_negatives: int
) -> List[str]:
    """
    Sample hard negatives using dense retrieval only

    Args:
        pos_chunk_id: Positive chunk ID (to exclude from negatives)
        corpus: List of corpus chunks
        query_emb: Query embedding
        corpus_embs: Corpus embeddings matrix
        num_negatives: Number of negatives to sample

    Returns:
        List of negative chunk representations
    """
    negatives = []
    pos_idx = None

    # Find positive chunk index
    for idx, chunk in enumerate(corpus):
        if chunk['id'] == pos_chunk_id:
            pos_idx = idx
            break

    if pos_idx is None:
        logger.warning(f"Positive chunk {pos_chunk_id} not found in corpus")
        return []

    # Dense similarity (cosine)
    dense_scores = np.dot(corpus_embs, query_emb) / (
        np.linalg.norm(corpus_embs, axis=1) * np.linalg.norm(query_emb)
    )

    # Top-k (exclude positive)
    scored = [(i, dense_scores[i]) for i in range(len(corpus)) if i != pos_idx]
    scored.sort(key=lambda x: x[1], reverse=True)

    for idx, _ in scored[:num_negatives]:
        negatives.append(corpus[idx]['representation'])

    return negatives


def add_hard_negatives(
    trainset: List[Dict[str, Any]],
    corpus: List[Dict[str, Any]],
    model,
    num_negatives: int
) -> List[Dict[str, Any]]:
    """
    Add hard negatives to training data using dense retrieval only

    Args:
        trainset: List of training samples (query + pos)
        corpus: List of corpus chunks
        model: Embedding model
        num_negatives: Number of negatives per sample

    Returns:
        List of training samples with hard negatives
    """
    # Encode corpus
    corpus_texts = [c['representation'] for c in corpus]
    logger.info("Encoding corpus...")
    corpus_embs = model.encode(corpus_texts).dense_vecs
    logger.success(f"Corpus embeddings: {corpus_embs.shape}")

    # Encode queries
    logger.info("Encoding queries...")
    queries = [item['query'] for item in trainset]
    query_embs = model.encode(queries).dense_vecs
    logger.success(f"Query embeddings: {query_embs.shape}")

    # Add hard negatives
    logger.info("Sampling hard negatives...")
    trainset_with_negatives = []

    for idx, item in enumerate(trainset):
        query = item['query']
        pos_repr = item['pos'][0]  # Assume single positive

        # Find pos chunk_id
        pos_chunk_id = None
        for chunk in corpus:
            if chunk['representation'] == pos_repr:
                pos_chunk_id = chunk['id']
                break

        if pos_chunk_id is None:
            logger.warning(f"Positive chunk not found for query: {query[:50]}...")
            continue

        # Sample hard negatives
        neg_reprs = sample_hard_negatives(
            pos_chunk_id, corpus,
            query_embs[idx], corpus_embs,
            num_negatives
        )

        trainset_with_negatives.append({
            'query': query,
            'pos': [pos_repr],
            'neg': neg_reprs
        })

    logger.success(f"Generated {len(trainset_with_negatives)} samples with hard negatives")
    return trainset_with_negatives


def main():
    parser = argparse.ArgumentParser(
        description="Add hard negatives to training data using dense retrieval"
    )
    parser.add_argument("--trainset-input", type=Path, required=True, help="Input trainset (query + pos)")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus file path")
    parser.add_argument("--output", type=Path, required=True, help="Output file path")
    parser.add_argument("--num-negatives", type=int, default=8, help="Number of negatives per sample (default: 8)")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use (default: 0)")

    args = parser.parse_args()

    # Load trainset
    logger.info("Loading trainset...")
    trainset = load_corpus(args.trainset_input)  # Reuse load_corpus for jsonl loading

    # Load corpus
    corpus = load_corpus(args.corpus)

    # Initialize embedding model
    logger.info(f"Initializing embedding model on GPU {args.gpu_id}...")
    model = init_embedding_model(DEFAULT_MODEL_NAME, gpu_id=args.gpu_id, batch_size=DEFAULT_BATCH_SIZE)

    # Add hard negatives
    trainset_with_negatives = add_hard_negatives(
        trainset, corpus, model, args.num_negatives
    )

    # Shuffle
    random.shuffle(trainset_with_negatives)

    # Save
    save_trainset(trainset_with_negatives, args.output)

    # Statistics
    logger.info("Completed!")
    logger.info(f"Statistics:")
    logger.info(f"  Corpus chunks: {len(corpus)}")
    logger.info(f"  Training samples: {len(trainset_with_negatives)}")
    logger.info(f"  Negatives per sample: {args.num_negatives}")


if __name__ == "__main__":
    main()