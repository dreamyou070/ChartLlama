#!/bin/bash

# pretrain_mm_mlp_adapter="/mnt/private_yucheng/huggingface_hub/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5/mm_projector.bin"
CHUNKS=1
IDX=0

question_file=r"./data/vistext/vistext_qa.json"
image_folder=r"./data/vistext/sy"
answers_file=r"./data/vistext/sy_answer.json"
vision_tower="openai/clip-vit-large-patch14-336"
CUDA_VISIBLE_DEVICES=1 \
  python generate_code.py \
  --model_base "lmsys/vicuna-13b-v1.5" \
  --model_path "listen2you002/ChartLlama-13b" \
  --vision_tower "${vision_tower}" \
  --question_file "./data/ChartLlama-Dataset/ours/box_chart_100examples_simplified_qa.json" \
  --image_folder "./data/ChartLlama-Dataset/ours" \
  --answers_file "./data/ChartLlama-Dataset/answer/box_chart_answer.json" \
  --num-chunks $CHUNKS \
  --chunk-idx $IDX \
  --temperature 0 \
  --conv-mode vicuna_v1