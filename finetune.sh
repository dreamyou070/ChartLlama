#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:
# --deepspeed ./scripts/zero2.json \
################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################
# --bf16 True \
# --tf32 True \
# scripts/zero2.json
# model_name_or_path = /your_path_to/LLaVA/checkpoints/${output_name}
################## LLaMA-2 ##################
PROMPT_VERSION="llava_llama_2"
MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

python ChartLlama-code/llava/train/train_mem.py \
    --model_name_or_path "/share0/dreamyou070/dreamyou070/pretrained_model/${MODEL_VERSION}" \
    --version $PROMPT_VERSION \
    --data_path ./ChartLlama-code/playground/data/llava_instruct_80k.json \
    --image_folder ./data/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter "./ChartLlama-code/checkpoints/llava-${MODEL_VERSION}/mm_projector.bin" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb