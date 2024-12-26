# %%
import time
import torch
from transformers import PreTrainedModel
from transformers.models.llama import LlamaModel
from transformers.models.gpt2 import GPT2Model
from types import SimpleNamespace
from typing import Union
from einops import rearrange, reduce, einsum
import utils
import importlib
import os
from pathlib import Path
import argparse

# %%
# %%
device = torch.device(
    "cuda" if torch.cuda.is_available() else "tpu" if torch.backends.xla else "cpu"
)
print(f"Using device: {device}")

# %%
importlib.reload(utils)
global DatasetTypes, ModelTypes, get_dataset, get_tokenizer_model
from utils import DatasetTypes, ModelTypes, get_dataset, get_tokenizer_model

# %%
model_type = ModelTypes.LLAMA
dataset_type = DatasetTypes.CODE
tokenizer, model = get_tokenizer_model(model_type, get_intermediates=True)
model = model.to(device)
model.config
# %%
n_samples = 2
seq_len = 128
batch_size = 1
# %%
stream = get_dataset(dataset_type, tokenizer, seq_len, batch_size)
# %%
activations = [{} for _ in range(n_samples)]
# %%
for i in range(n_samples):
    cur_seq_len = 0

    inputs = next(stream)
    inputs_sliced = {"input_ids": torch.stack(inputs).to(device)}

    with torch.no_grad():
        outputs = model(**inputs_sliced)

    attentions = torch.stack(outputs.attentions)
    activations[i]["att_wei"] = attentions
    activations[i]["q_proj"] = torch.stack(model.attention_intermediates["q_proj"])
    activations[i]["k_proj"] = torch.stack(model.attention_intermediates["k_proj"])
    activations[i]["v_proj"] = torch.stack(model.attention_intermediates["v_proj"])
    model.attention_intermediates = {}
# %%
activations[1]["q_proj"].shape, activations[1]["k_proj"].shape, activations[1][
    "v_proj"
].shape, activations[1]["att_wei"].shape
# %%
layer_intermediates = activations[1]
n_q_per_kv = model.config.num_attention_heads // model.config.num_key_value_heads
q_0 = rearrange(
    layer_intermediates["q_proj"],
    "L B Qlen (n_kv n_q_per_kv d_head) -> L B Qlen n_kv n_q_per_kv d_head",
    n_kv=model.config.num_key_value_heads,
    n_q_per_kv=n_q_per_kv,
)[0]
k_0 = rearrange(
    layer_intermediates["k_proj"],
    "L B Klen (n_kv d_head) -> L B Klen n_kv d_head",
    n_kv=model.config.num_key_value_heads,
)[0]
logits_0 = rearrange(
    layer_intermediates["att_wei"],
    "layers B (n_kv n_q) Qlen Klen -> layers B n_kv n_q Qlen Klen",
    n_kv=model.config.num_key_value_heads,
)[0]
q_0.shape, k_0.shape, logits_0.shape

# %%
k_clusters = einsum(
    logits_0,
    k_0,
    "B n_kv n_q_per_kv Qlen Klen, B Klen n_kv d_head -> B n_kv n_q_per_kv d_head Qlen",
)
q_clusters = einsum(
    logits_0,
    q_0,
    "B n_kv n_q_per_kv Qlen Klen, B Qlen n_kv n_q_per_kv d_head -> B n_kv n_q_per_kv d_head Klen",
)
# %%
# essentially we find for each head and for each batch, which k we should use, by getting the argmax of the q_clusters
q_labels = torch.argmax(q_clusters, dim=-1, keepdim=True)
k_labels = torch.argmax(k_clusters, dim=-1, keepdim=True)
"""

q_labels = torch.argmax(q_clusters, dim=-2, keepdim=True)
tensor([[  0,   0,   0,   0,   0, 127, 127,   0, 127, 127, 127, 127,   0,   0,
           0,   0, 127, 127, 127,   4,  83,   4,   4,   4,   4, 120,   0,   0,
           0, 127,   0, 127,  24, 127,   0, 127, 127,   0,   0,   0, 127,   0,
           0,   0,   0, 127,   0,   0,   0,   0,   0,   0,   0,   4,   4,   0,
          24,   0,   0,   4,  83,   0, 127,   0]], device='cuda:0')
this pretty much says for each feature in the head, which k_0 to use

OR
q_labels = torch.argmax(q_clusters, dim=-1)
tensor([52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52,
         52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52,
         52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52,
         52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52,
         52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52,
         52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52,
         52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52,
         52, 52], device='cuda:0'),
for each query this says what feature is the most important

"""
# %%
q_clusters.shape, q_labels.shape
# %%
# causal mask?
k_avg = reduce(k_0, "B Klen n_kv d_head -> B 1 n_kv d_head", "mean")
q_avg = reduce(
    q_0, "B Qlen n_kv n_q_per_kv d_head -> B 1 n_kv n_q_per_kv d_head", "mean"
)
# %%
cluster_alignment = einsum(
    q_0,
    k_avg,
    "B Qlen n_kv n_q_per_kv d_head, B Klen n_kv d_head -> B n_kv n_q_per_kv Qlen Klen",
)
cluster_alignment

# %%
logits_0.shape, cluster_alignment.shape

# %%
