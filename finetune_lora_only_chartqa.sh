#!/bin/bash
# [1] model argument (what is mm ?)
model_name_or_path="lmsys/vicuna-13b-v1.5"
VERSION="v1"
vision_tower="openai/clip-vit-large-patch14-336"
pretrain_mm_mlp_adapter="/share0/dreamyou070/dreamyou070/CharLlama/ChartLlama/training_result_sy/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"
mm_projector_type="mlp2x_gelu"
# [2] data argument
data_path='data/ChartLlama-Dataset/box_chart_100examples_simplified_qa.json'
image_folder='data/ChartLlama-Dataset/ours'
# [3] Training Argument

deepspeed train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${model_name_or_path} \
    --version ${VERSION} \
    --vision_tower ${vision_tower} \
    --cache_dir training_result_sy \
    --freeze_backbone True \
    --lora_enable True \
    --data_path ${data_path} \
    --image_folder ${image_folder} \
    --image_aspect_ratio pad \
    --fp16 True \
    --lora_r 64 \
    --output_dir ./training_result_sy/finetune_llm_lora_only_chartqa \
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
    --dataloader_num_workers 4 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False