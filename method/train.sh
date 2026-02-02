#!/bin/bash
# Train mimo_en on all variants

datasets=("mimo_en" "mimo_ch" "e2ewtq" "feta" "ottqa")

GPUID=0
MASTER_PORT=29500

for d in "${datasets[@]}"; do
    for v in "${variants[@]}"; do
        echo "=========================================="
        echo "[Training: ${d}${v}]"
        echo "=========================================="

        CUDA_VISIBLE_DEVICES=${GPUID} uv run torchrun --nproc_per_node=1 \
            --master_port=${MASTER_PORT} \
            -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
            --model_name_or_path BAAI/bge-m3 \
            --train_data dataset/${d}.jsonl \
            --output_dir model/${d} \
            --learning_rate 1e-5 \
            --num_train_epochs 2 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 32 \
            --bf16 \
            --temperature 0.01 \
            --query_max_len 1024 \
            --passage_max_len 4096 \
            --normalize_embeddings \
            --save_steps 200 \
            --logging_steps 50 \
            --overwrite_output_dir

        echo "Completed: ${d}${v}"
        echo ""
    done
done

echo "All mimo_en variants trained!"