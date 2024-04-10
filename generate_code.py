import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import warnings
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
from llava.model import *
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math
from typing import Optional, List, Dict
from dataclasses import dataclass, field
import transformers
from peft import LoraConfig, get_peft_model
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    lora_further_tune_finetuned: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear') # mlp2x_gelu
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)


def load_pretrained_model(model_path, model_base,
                          model_name, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cuda"):

    kwargs = {"device_map": device_map}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    else:
        kwargs['torch_dtype'] = torch.float16

    # Load LLaVA model
    if model_base is None:
        raise ValueError('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
    if model_base is not None:
        print(f' [1] tokenizer (make lora config)')
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

        print(f' [2] loading lora config)')
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path) # LlavaConfig
        lora_cfg_pretrained.mm_vision_tower = args.vision_tower

        print(f' [3.1] base model with lora')
        model = LlavaLlamaForCausalLM.from_pretrained(model_base,
                                                      low_cpu_mem_usage=True,
                                                      config=lora_cfg_pretrained,
                                                      **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print(f' [3.2] lora zero file')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(repo_id=repo_id,
                                             filename=filename,
                                             subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')
            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        print(f' [3.3] loading lora weights and merging')
        # parameter efficient
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        print('Model is loaded...')

    print(f' [4] tokenizer agumenting with patch token')
    image_processor = None
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    print(f' [5] vision model')
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    # what is device ?
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):

    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):

        # [1] get one line
        line = self.questions[index]

        # [2] get image file
        image_file = line["image"] # image_path

        # [3] human asking
        # DEFAULT_IMAGE_TOKEN = <image> --> What is the title of the chart?
        # DEFAULT_IM_START_TOKEN + (DEFAULT_IMAGE_TOKEN) + DEFAULT_IM_END_TOKEN
        qs = line["conversations"][0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # conv_mode = vicuna_v1 -> conv_vicuna_v1
        conv = conv_templates[args.conv_mode].copy()
        print(f'conv (conv_vicuna_v1) = {conv}')
        # conv.roles[0] = USER
        # conv.roles[1] =  ASSISTANT
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None) # no answer

        # [5] input image
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        # [4] get prompt
        prompt = conv.get_prompt()
        print(f'prompt = {prompt}')
        input_ids = tokenizer_image_token(prompt,
                                          self.tokenizer,
                                          IMAGE_TOKEN_INDEX,
                                          return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            # in case you load the whole model
            if 'mm_projector' in names:
                # on mm_projector, skip
                continue
            print(f'lora on {names[0]}')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def eval_model(args):

    print(f'\n step 1. parse arguments and dtype')
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    print(f'\n step 2. make model')
    print(f' (2.1) base model')
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:  # light model
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(
            dict(device_map={"": training_args.device}, load_in_4bit=training_args.bits == 4,
                 load_in_8bit=training_args.bits == 8,
                 quantization_config=BitsAndBytesConfig(load_in_4bit=training_args.bits == 4,
                                                        load_in_8bit=training_args.bits == 8,
                                                        llm_int8_threshold=6.0,
                                                        llm_int8_has_fp16_weight=False,
                                                        bnb_4bit_compute_dtype=compute_dtype,
                                                        bnb_4bit_use_double_quant=training_args.double_quant,
                                                        bnb_4bit_quant_type=training_args.quant_type)))
    if model_args.vision_tower is not None:  # llava model
        if 'mpt' in model_args.model_name_or_path:  # mpt version
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(model_args.model_name_or_path, config=config,
                                                        cache_dir=training_args.cache_dir,
                                                        **bnb_model_from_pretrained_args)
        else: # llava version
            model = LlavaLlamaForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                          cache_dir=training_args.cache_dir,
                                                          **bnb_model_from_pretrained_args)
    else:  # just lama model
        model = transformers.LlamaForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                              cache_dir=training_args.cache_dir,
                                                              **bnb_model_from_pretrained_args)
    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    print(f' (2.2) base model')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default='/mnt/private_yucheng/huggingface_hub/llava-v1.5-13b')
    parser.add_argument("--vision_tower", type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="/mnt/private_yucheng/chartgpt/LLaVA/playground/data")
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()
    eval_model(args)