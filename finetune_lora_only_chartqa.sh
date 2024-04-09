#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

################## vicuna-v1.5 ##################
# MODEL_VERSION="llama-2-7b-chat"
################## vicuna-v1.5 ##################


# the mm_mlp_adapter is ignored
#--pretrain_mm_mlp_adapter /mnt/private_yucheng/huggingface_hub/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5/mm_projector.bin \
# --group_by_modality_length True \

# just llave finetuning

# [1] model argument (what is mm ?)
#model_name_or_path="/mnt/gyfs/yuchenghan/llama2_models/vicuna-13b-v1.5"

# I think i need to change name
model_name_or_path="lmsys/vicuna-13b-v1.5"

version="v1"
vision_tower="openai/clip-vit-large-patch14-336"
#pretrain_mm_mlp_adapter="/mnt/private_yucheng/huggingface_hub/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5/mm_projector.bin"
pretrain_mm_mlp_adapter="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5/mm_projector.bin"
mm_projector_type="mlp2x_gelu"

# [2] data argument
#data_path="/mnt/private_yucheng/chartgpt/LLaVA/playground/llava_mix_plus_chartqa.json"
data_path='data/ChartLlama-Dataset/ours/box_chart_100examples_simplified_qa.json'
# image_folder /mnt/private_yucheng/chartgpt/LLaVA/playground/data \
image_folder='data/ChartLlama-Dataset/ours/ours/box_chart/png'

# [3] Training Argument



deepspeed train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${model_name_or_path} \
    --version ${version} \
    --vision_tower ${vision_tower} \
    --pretrain_mm_mlp_adapter ${pretrain_mm_mlp_adapter} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --lora_enable True \
    --data_path /mnt/private_yucheng/chartgpt/LLaVA/playground/only_chartqa.json \
    --image_folder ${image_folder} \
    --image_aspect_ratio pad \
    --fp16 True \
    --lora_r 64 \
    --output_dir ./checkpoints/finetune_lora_only_chartqa \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4