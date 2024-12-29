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
from utils import (
    DatasetTypes,
    ModelTypes,
    get_dataset,
    get_tokenizer_model,
    get_model_state_dict,
)

# %%
model_type = ModelTypes.LLAMA
dataset_type = DatasetTypes.CODE
n_samples = 100
seq_len = 1024
batch_size = 1

# %%
model_name = model_type.value
tokenizer = AutoTokenizer.from_pretrained(model_name)
state_dict = get_model_state_dict(AutoModelForCausalLM.from_pretrained(model_name))
model = LlamaModel.from_pretrained(model_name)
model.load_state_dict(state_dict)
# %%
stream = get_dataset(dataset_type, tokenizer, seq_len, batch_size)
# %%
model.attention_intermediates = {}

# Create a mapping of modules to their full names
saved_projs = ["q_proj", "k_proj", "v_proj", "roped_q_proj", "roped_k_proj"]
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
activations = [{} for _ in range(n_samples)]
for i in range(n_samples):
    cur_seq_len = 0

    inputs = next(stream)
    inputs_sliced = {"input_ids": torch.stack(inputs).to(device)}

    with torch.no_grad():
        outputs = model(**inputs_sliced)
        if get_output_attentions:
            cpu_attentions = [att.cpu() for att in outputs.attentions]

    if get_output_attentions:
        activations[i]["att_wei"] = torch.stack(cpu_attentions).cpu()
    activations[i]["q_proj"] = torch.stack(
        model.attention_intermediates["q_proj"]
    ).cpu()
    activations[i]["k_proj"] = torch.stack(
        model.attention_intermediates["k_proj"]
    ).cpu()
    activations[i]["v_proj"] = torch.stack(
        model.attention_intermediates["v_proj"]
    ).cpu()
    model.attention_intermediates = {}
