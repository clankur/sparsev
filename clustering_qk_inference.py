# %%
import time
import torch
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama import LlamaModel
from transformers.models.gpt2 import GPT2Model
from types import SimpleNamespace
from typing import Union, Tuple
from einops import rearrange, reduce, einsum
import utils
import importlib
import os
from pathlib import Path
import argparse
from sklearn.cluster import KMeans
from modeling_llama import LlamaModel


# %%
importlib.reload(utils)
global DatasetTypes, ModelTypes, get_dataset, get_tokenizer_model
from utils import DatasetTypes, ModelTypes, get_dataset, get_tokenizer_model

# %%
model_type = ModelTypes.LLAMA
# %%
model_name = model_type.value
tokenizer = AutoTokenizer.from_pretrained(model_name)
state_dict = AutoModelForCausalLM.from_pretrained(
    model_name,
).state_dict()
model = LlamaModel.from_pretrained(model_name)
# %%
model.attention_intermediates = {}

# Create a mapping of modules to their full names
saved_projs = ["q_proj", "k_proj", "v_proj", "self_attn"]
module_to_name = {}
for name, module in model.named_modules():
    if any(proj in name.lower() for proj in saved_projs):
        module_to_name[module] = name


def attention_forward_hook(module, input, output):
    full_name = module_to_name[module]
    parts = full_name.split(".")
    proj_type = parts[-1]  # e.g., 'q_proj'
    layer_num = int(parts[2])  # e.g., '0' from 'model.layers.0.self_attn.q_proj'

    if proj_type not in model.attention_intermediates:
        model.attention_intermediates[proj_type] = []

    model.attention_intermediates[proj_type].append(output)
    return output


# Register hooks for all attention layers
for name, module in model.named_modules():
    if any(proj in name.lower() for proj in saved_projs):
        module.register_forward_hook(attention_forward_hook)
# %%
