#!/bin/bash

result_name="result"
output_name="result"

python llava.eval.model_vqa_lora.py --model-path /${result_name}/LLaVA/checkpoints/${output_name} \
    --question-file /your_path_to/question.json \
    --image-folder ./playground/data/ \
    --answers-file ./playground/data/ans.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &