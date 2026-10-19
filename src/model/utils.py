import os
import random
from typing import Any, Dict
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])
MODEL_DIR = os.path.join(ROOT_DIR, r'Llama-3.1-8B-Instruct')

__all__ = [
    'GENERAL_TEMPLATE',
    'render_chat',
    'build_tokenizer',
    'build_base_peft_model',
    'set_seed',
    'print_trainable_parameters',
]

GENERAL_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)

def render_chat(messages:list[Dict[str,str]],tokenizer,add_generation_prompt=False):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt = add_generation_prompt
    )

#优先使用自带模版
def build_tokenizer(
    args,
    chat_template:str = None,
    no_template:bool = False,
    padding_side:str = 'right',
    truncation_side: str = 'right')->AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = truncation_side
    tokenizer.model_max_length = 1536
    if getattr(tokenizer, 'chat_template', None) is None and chat_template is not None:
        tokenizer.chat_template = chat_template
    if no_template:
        tokenizer.chat_template =  "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    return tokenizer

def build_base_peft_model(args)->AutoModelForCausalLM:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype = torch.bfloat16,
        device_map = {"": local_rank}
    )
    base_model.config.use_cache = False
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
    #lora默认关闭base模型梯度，这里用新方法强制打开embedding 输出激活值的梯度(不更新)，如果transformer版本不够就用下面的方法
    else:
        def make_inputs_require_grad(_module, _inputs, output):
            output.requires_grad_(True)
        base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
		#这段是为了兼容 梯度检查点（gradient checkpointing） 和 PEFT。
    peft_config = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        target_modules = find_all_linear_names(base_model),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    base_model = get_peft_model(base_model, peft_config)
    print_trainable_parameters(base_model)
    return base_model

def find_all_linear_names(model)->list:
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module_name = name.split(".")[-1]
            if module_name != "lm_head":
                lora_module_names.add(module_name)
    return sorted(lora_module_names)
    
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} "
        f"|| trainable percentage: {100 * trainable_params / all_param:.3f} %"
    )
