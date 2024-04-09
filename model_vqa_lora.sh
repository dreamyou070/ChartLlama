model_name_or_path="listen2you002/ChartLlama-13b"
question-file=r"./data/ChartLlama-Dataset/ours/box_chart_100examples_simplified_qa.json"
image-folder=r"./data/ChartLlama-Dataset/ours"
answers-file=r"./data/ChartLlama-Dataset/answer/box_chart_answer.json"
CHUNKS=1
IDX=0

CUDA_VISIBLE_DEVICES=1 python -m llava/eval/model_vqa_lora.py \
 --model-path ${model_name_or_path} \
 --question-file /your_path_to/question.json \
 --image-folder ./playground/data/ \
 --answers-file ./playground/data/ans.jsonl \
 --num-chunks $CHUNKS \
 --chunk-idx $IDX \
 --temperature 0 \
 --conv-mode vicuna_v1