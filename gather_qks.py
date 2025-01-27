# %%
import time
import torch
from transformers import AutoTokenizer
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
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
seq_len = 1024
batch_size = 4
n_samples = 25
# %%
model_name = model_type.value
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)
stream = get_dataset(dataset_type, tokenizer, seq_len, batch_size)
# %%
model = LlamaForCausalLM.from_pretrained(model_name)
model = model.to(device)
config = model.config
config


# %%
sample_idx = 0
model.attention_intermediates = {}
n_kv_heads = model.config.num_key_value_heads
n_q_per_kv = model.config.num_attention_heads // model.config.num_key_value_heads
d_head = model.config.head_dim

saved_projs = ["q_proj", "k_proj", "v_proj", "roped_q_proj", "roped_k_proj", "logits"]
module_to_name = {}
for name, module in model.named_modules():
    if any(proj in name.lower() for proj in saved_projs):
        module_to_name[module] = name


def attention_forward_hook(module, input, output):
    global sample_idx
    full_name = module_to_name[module]
    parts = full_name.split(".")
    proj_type = parts[-1]  # e.g., 'q_proj'
    layer_num = parts[2]  # e.g., '0' from 'model.layers.0.self_attn.q_proj'
    if proj_type not in model.attention_intermediates:
        model.attention_intermediates[proj_type] = [
            [[] for _ in range((model.config.num_hidden_layers))]
            for _ in range(n_samples)
        ]

    if proj_type in saved_projs:
        model.attention_intermediates[proj_type][sample_idx][int(layer_num)].append(
            output.cpu()
        )
    return output


for name, module in model.named_modules():
    if any(proj in name.lower() for proj in saved_projs):
        module.register_forward_hook(attention_forward_hook)

# %%

model.eval()
while sample_idx < n_samples:
    past_key_values = None
    sequence_ids = torch.stack(next(stream)).to(device)
    with torch.no_grad():
        seq_index = 0
        while seq_index < sequence_ids.size(1):
            # Take the "next token" directly from sequence_ids
            next_token_id = sequence_ids[:, seq_index : seq_index + 1]

            # If there's no past yet, feed the entire context. Otherwise, just feed the single next token.
            input_ids = next_token_id

            outputs = model(
                input_ids=input_ids, past_key_values=past_key_values, use_cache=True
            )

            past_key_values = outputs.past_key_values
            # Append next_token_id to generated_ids for final decoding
            seq_index += 1
    sample_idx += 1


# %%
for proj_type, activation in model.attention_intermediates.items():
    if proj_type == "logits":
        continue
    model.attention_intermediates[proj_type] = torch.stack(
        [torch.stack([torch.stack(layer) for layer in sample]) for sample in activation]
    )
    print(proj_type, model.attention_intermediates[proj_type].shape)


# %%
# get last sample, last layer keys calulate logits for last query of same sample, layer
q = model.attention_intermediates["roped_q_proj"][-1][-1][-1]
k = model.attention_intermediates["roped_k_proj"][-1][-1]
q = rearrange(
    q,
    "B (n_kv n_q_per_kv) Qlen d_head -> B n_kv n_q_per_kv Qlen d_head",
    n_kv=n_kv_heads,
    n_q_per_kv=n_q_per_kv,
)
k = rearrange(
    k,
    "seq_len B n_kv Klen d_head -> B n_kv (seq_len Klen) d_head",
    n_kv=n_kv_heads,
    d_head=d_head,
)

print(f"{q.shape=}", f"{k.shape=}")
logits = einsum(
    q,
    k,
    "B n_kv n_q_per_kv Qlen d_head, B n_kv Klen d_head -> B n_kv n_q_per_kv Qlen Klen",
) / math.sqrt(d_head)
# %%
logits = rearrange(
    logits, "B n_kv n_q_per_kv Qlen Klen -> B (n_kv n_q_per_kv) Qlen Klen"
)
logits.shape
# %%
if "logits" in model.attention_intermediates:
    output_logits = model.attention_intermediates["logits"][-1][-1][-1]
    assert (
        logits.shape == output_logits.shape
    ), f"{logits.shape=} != {output_logits.shape=}"
    is_equal = torch.allclose(logits, output_logits, rtol=1e-5, atol=1e-5)
    max_diff = torch.max(torch.abs(logits - output_logits))
    print(f"Attention weights equal: {is_equal}")
    print(f"Maximum difference: {max_diff:.2e}")

# %%
output_dir = Path("/mnt/HDD/datsets/attention_intermediates")
# Remove directory if it exists and recreate it
if output_dir.exists():
    import shutil

    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Save each type of projection for each layer
for proj_type, layers in model.attention_intermediates.items():
    if proj_type == "logits":
        continue
    for layer_idx, tensors in enumerate(layers):
        # Stack all tensors for this layer and projection type

        layer_tensors = torch.stack(tensors)

        # Create filename
        filename = output_dir / f"layer_{layer_idx}_{proj_type}.pt"

        # Save tensor
        torch.save(layer_tensors, filename)

print(f"Saved attention intermediates to {output_dir}")

# %%
