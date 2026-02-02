import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans

from config import DEFAULT_MODEL_NAME, DEFAULT_BATCH_SIZE
from embedding_model import init_embedding_model
from utils import clean_text, load_corpus, serialize_table, save_corpus


def init_model_on_gpu(gpu_id: int):
    """Initialize embedding model on specified GPU (called in subprocess)."""
    return init_embedding_model(DEFAULT_MODEL_NAME, gpu_id=gpu_id, batch_size=DEFAULT_BATCH_SIZE)


def embed_tables_on_gpu(
    gpu_id: int,
    tables_with_ids: List[Tuple[int, Dict[str, Any]]]
) -> Dict[int, np.ndarray]:
    """
    Embed all instances for assigned tables on a specific GPU.
    Batches all instances together for efficient GPU utilization.

    Args:
        gpu_id: GPU device ID
        tables_with_ids: List of (table_id, table_dict) tuples

    Returns:
        Dict mapping table_id -> embeddings array (n_instances, dim)
    """
    model = init_model_on_gpu(gpu_id)

    # Collect all instances with table mapping
    all_instances: List[str] = []
    table_offsets: List[Tuple[int, int, int]] = []  # (table_id, start, end)

    for table_id, table in tables_with_ids:
        instances = table.get('instances', [])
        if not instances:
            continue

        instances_clean = [clean_text(str(inst)) for inst in instances]
        start = len(all_instances)
        all_instances.extend(instances_clean)
        end = len(all_instances)
        table_offsets.append((table_id, start, end))

    logger.info(f"GPU {gpu_id}: Encoding {len(all_instances)} instances from {len(table_offsets)} tables")

    # Single batch encode
    if not all_instances:
        return {}

    all_embeddings = model.encode(all_instances).dense_vecs

    # Split back to tables
    results = {}
    for table_id, start, end in table_offsets:
        results[table_id] = all_embeddings[start:end]

    logger.info(f"GPU {gpu_id}: Completed")
    return results


def cluster_instances_from_embeddings(
    embeddings: np.ndarray,
    min_instances_per_chunk: int,
    max_chunks_per_table: int | None
) -> List[List[int]]:
    """
    K-means clustering using pre-computed embeddings.

    Args:
        embeddings: Pre-computed embedding vectors of shape (n_instances, dim).
        min_instances_per_chunk: Minimum number of instances per cluster.
        max_chunks_per_table: Maximum number of clusters allowed. None for unlimited.

    Returns:
        List of instance index groups, where each group is a list of indices
        belonging to the same cluster.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n <= min_instances_per_chunk:
        return [list(range(n))]

    n_clusters = max(1, n // min_instances_per_chunk)
    if max_chunks_per_table is not None:
        n_clusters = min(max_chunks_per_table, n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    chunks = defaultdict(list)
    for idx, label in enumerate(labels):
        chunks[label].append(idx)

    return list(chunks.values())


def build_chunks_from_table(
    table: Dict[str, Any],
    table_id: int,
    embeddings: np.ndarray,
    min_instances_per_chunk: int,
    max_chunks_per_table: int | None,
    max_instances_per_representation: int,
    start_chunk_id: int
) -> List[Dict[str, Any]]:
    """
    Build chunks for a single table using pre-computed embeddings.

    Args:
        table: Table dictionary containing file_name, sheet_name, header, instances.
        table_id: Unique identifier for the table.
        embeddings: Pre-computed embeddings for table instances.
        min_instances_per_chunk: Minimum instances per chunk for clustering.
        max_chunks_per_table: Maximum chunks allowed per table. None for unlimited.
        max_instances_per_representation: Max instances to include in chunk representation.
        start_chunk_id: Starting ID for chunk numbering.

    Returns:
        List of chunk dictionaries with id, representation, and metadata.
    """
    file_name = table.get('file_name', '')
    sheet_name = table.get('sheet_name', '')
    header = table.get('header', [])
    instances = table.get('instances', [])

    # Clean
    if isinstance(header, list):
        header_clean = ' '.join(clean_text(str(h)) for h in header)
    else:
        header_clean = clean_text(str(header))

    instances_clean = [clean_text(str(inst)) for inst in instances]
    if not instances_clean:
        return []

    # Cluster using pre-computed embeddings
    chunk_groups = cluster_instances_from_embeddings(
        embeddings,
        min_instances_per_chunk,
        max_chunks_per_table
    )

    # Build chunks
    chunks = []
    for local_chunk_id, instance_indices in enumerate(chunk_groups):
        chunk_instances = [instances_clean[i] for i in instance_indices]

        chunk_item = {
            'id': start_chunk_id + local_chunk_id,
            'representation': serialize_table(
                file_name, sheet_name, header_clean,
                chunk_instances[:max_instances_per_representation]
            ),
            'metadata': {
                'file_name': file_name,
                'sheet_name': sheet_name,
                'header': header_clean,
                'instances': chunk_instances,
                'table_id': table_id,
                'chunk_id': local_chunk_id,
                'n_chunks_in_table': len(chunk_groups)
            }
        }
        chunks.append(chunk_item)

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Format tables into chunks using K-means clustering (multi-GPU support)"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input file path (table.jsonl)")
    parser.add_argument("--output", type=Path, required=True, help="Output file path")
    parser.add_argument("--gpu-ids", type=str, default="0", help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    parser.add_argument("--min-instances-per-chunk", type=int, default=10, help="Minimum instances per chunk (default: 10)")
    parser.add_argument("--max-chunks-per-table", type=int, default=5, help="Maximum chunks per table (default: 5)")
    parser.add_argument("--max-instances-per-representation", type=int, default=5, help="Maximum instances in chunk representation (default: 5)")

    args = parser.parse_args()
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Load tables
    logger.info("Loading tables...")
    tables = load_corpus(args.input)  # Reuse load_corpus for jsonl loading

    # Distribute tables to GPUs
    logger.info(f"Distributing tables to {num_gpus} GPUs: {gpu_ids}")
    gpu_assignments: List[List[Tuple[int, Dict]]] = [[] for _ in range(num_gpus)]
    for table_id, table in enumerate(tables):
        gpu_idx = table_id % num_gpus
        gpu_assignments[gpu_idx].append((table_id, table))

    for i, gpu_id in enumerate(gpu_ids):
        logger.info(f"  GPU {gpu_id}: {len(gpu_assignments[i])} tables")

    # Parallel embedding on multiple GPUs
    logger.info("Starting parallel embedding...")
    all_embeddings: Dict[int, np.ndarray] = {}

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {
            executor.submit(embed_tables_on_gpu, gpu_ids[i], gpu_assignments[i]): i
            for i in range(num_gpus)
        }

        for future in as_completed(futures):
            gpu_idx = futures[future]
            try:
                result = future.result()
                all_embeddings.update(result)
                logger.info(f"Collected embeddings from GPU {gpu_ids[gpu_idx]}")
            except Exception as e:
                logger.error(f"GPU {gpu_ids[gpu_idx]} failed: {e}")
                raise

    logger.info(f"All embeddings collected: {len(all_embeddings)} tables")

    # Build chunks using collected embeddings
    logger.info("Building chunks with K-means clustering...")
    all_chunks = []
    global_chunk_id = 0

    for table_id, table in enumerate(tables):
        embeddings = all_embeddings.get(table_id)
        if embeddings is None:
            continue

        chunks = build_chunks_from_table(
            table, table_id, embeddings,
            args.min_instances_per_chunk,
            args.max_chunks_per_table,
            args.max_instances_per_representation,
            global_chunk_id
        )
        all_chunks.extend(chunks)
        global_chunk_id += len(chunks)

    logger.success(f"Generated {len(all_chunks)} chunks from {len(tables)} tables")

    save_corpus(all_chunks, args.output)

    logger.info("Completed!")
    logger.info(f"Statistics:")
    logger.info(f"  Tables: {len(tables)}")
    logger.info(f"  Chunks: {len(all_chunks)} ({len(all_chunks)/len(tables):.1f} avg per table)")
    logger.info(f"  GPUs used: {num_gpus}")


if __name__ == "__main__":
    main()
