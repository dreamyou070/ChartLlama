CHUNKS=1
IDX=0
# listen2you002/ChartLlama-13b
CUDA_VISIBLE_DEVICES=1 \
  python model_vqa_lora.py \
 --model-path "listen2you002/ChartLlama-13b" \
 --model-base "lmsys/vicuna-13b-v1.5" \
 --question-file "./data/ChartLlama-Dataset/ours/box_chart_100examples_simplified_qa.json" \
 --image-folder "./data/ChartLlama-Dataset/ours" \
 --answers-file "./data/ChartLlama-Dataset/answer/box_chart_answer.json" \
 --num-chunks $CHUNKS \
 --chunk-idx $IDX \
 --temperature 0 \
 --conv-mode vicuna_v1