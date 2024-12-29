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
import modeling_llama


device = torch.device(
    "cuda" if torch.cuda.is_available() else "tpu" if torch.backends.xla else "cpu"
)
print(f"Using device: {device}")


# %%
importlib.reload(utils)
importlib.reload(modeling_llama)
global DatasetTypes, ModelTypes, get_dataset, get_tokenizer_model
from utils import (
    DatasetTypes,
    ModelTypes,
    get_dataset,
    get_tokenizer_model,
)
from modeling_llama import LlamaForCausalLM

# %%
model_type = ModelTypes.LLAMA
dataset_type = DatasetTypes.CODE
n_samples = 1
seq_len = 10
batch_size = 1

# %%
model_name = model_type.value
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)
model = model.to(device)
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
    layer_num = parts[1]  # e.g., '0' from 'layers.0.self_attn.q_proj'
    if proj_type not in model.attention_intermediates:
        model.attention_intermediates[proj_type] = []
    print(proj_type, output.shape)
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

    model.generate(inputs_sliced["input_ids"], max_new_tokens=512)


# %%
model.attention_intermediates.keys()
