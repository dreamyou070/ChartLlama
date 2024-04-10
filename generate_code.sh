#!/bin/bash

# pretrain_mm_mlp_adapter="/mnt/private_yucheng/huggingface_hub/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5/mm_projector.bin"
CHUNKS=1
IDX=0

question_file="data/vistext/vistext_qa.json"
image_folder="data/vistext"
answers_file="data/vistext/sy_answer.json"
vision_tower="openai/clip-vit-large-patch14-336"
# head ...  ?
mm_projector_type="mlp2x_gelu"

CUDA_VISIBLE_DEVICES=1 \
  python generate_code.py \
  --model_base "lmsys/vicuna-13b-v1.5" \
  --model_path "listen2you002/ChartLlama-13b" \
  --vision_tower "${vision_tower}" \
  --question_file "${question_file}" \
  --image_folder "${image_folder}" \
  --answers_file "${answers_fil}" \
  --num-chunks $CHUNKS \
  --chunk-idx $IDX \
  --temperature 0 \
  --conv-mode vicuna_v1