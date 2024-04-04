#!/bin/bash

result_name="result"
output_name="result"
model-path="chartllama" #Lora Checkpoint

python ./llava.eval.model_vqa_lora.py \
  --model-path ${model-path} \
  --question-file llava/${result_name}/question.json \
  --image-folder ./playground/data/ \
  --answers-file ./playground/data/ans.jsonl \
  --num-chunks $CHUNKS \
  --chunk-idx $IDX \
  --temperature 0 \
  --conv-mode vicuna_v1 &