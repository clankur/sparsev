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
seq_len = 1024
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
# %%
# essentially we find for each head and for each batch, which k we should use, by getting the argmax of the q_clusters
k_labels = torch.argmax(k_clusters, dim=-1)
# %%
B, n_kv, n_q_per_kv, d_head, Klen = k_clusters.shape
batch_indices = torch.arange(B).view(B, 1, 1, 1).expand(B, n_kv, n_q_per_kv, d_head)
kv_indices = torch.arange(n_kv).view(1, n_kv, 1, 1).expand(B, n_kv, n_q_per_kv, d_head)
q_indices = (
    torch.arange(n_q_per_kv)
    .view(1, 1, n_q_per_kv, 1)
    .expand(B, n_kv, n_q_per_kv, d_head)
)
head_indices = (
    torch.arange(d_head).view(1, 1, 1, d_head).expand(B, n_kv, n_q_per_kv, d_head)
)
k_approx = k_clusters[batch_indices, kv_indices, q_indices, head_indices, k_labels]
# we have the most important k and q for each head
k_approx.shape
# %%
# what we can do is use k_clusters/q_clusters to ground some centroids
# and use k means to adjust them over each example
# should we use k_approx or a
# %%
# simplify and work with 1 head
k_0_0 = k_0[0, :, 0, :]  #  Klen x d_head
q_0_0 = q_0[0, :, 0, 0, :]  # Qlen x d_head
logits_0_0 = logits_0[0, 0, 0, :, :]  # Qlen x Klen
k_0_0, k_0_0.shape, q_0_0.shape, logits_0_0.shape

# TODO: maybe want to compare using non-roped q and k
k_0_clusters = einsum(
    logits_0_0,
    k_0_0,
    "Qlen Klen, Klen d_head -> Qlen d_head",
)


# %%
def kmeans(data, n_clusters, max_iters=50, tolerance=1e-4, seed=42):
    torch.manual_seed(seed)
    indices = torch.randperm(data.shape[0])[:n_clusters]
    centroids = data[indices]

    prev_assignments = None
    for iter_num in range(max_iters):
        distances = torch.cdist(data, centroids)
        cluster_assignments = torch.argmin(distances, dim=1)

        if prev_assignments is not None and torch.all(
            cluster_assignments == prev_assignments
        ):
            print(f"Converged after {iter_num} iterations")
            break

        new_centroids = torch.stack(
            [
                (
                    data[cluster_assignments == i].mean(dim=0)
                    if (cluster_assignments == i).any()
                    else centroids[i]
                )
                for i in range(n_clusters)
            ]
        )

        if torch.norm(new_centroids - centroids, p=2) < tolerance:
            print(f"Converged after {iter_num} iterations (tolerance reached)")
            break
        centroids = new_centroids
        prev_assignments = cluster_assignments
    return centroids, cluster_assignments


# %%
n_clusters = 10
centroids, cluster_assignments = kmeans(k_0_clusters, n_clusters)

# %%
cluster_assignments.shape, k_0_clusters.shape, centroids.shape, cluster_assignments
# %%
# need to apply rope
aligned = einsum(
    q_0_0,
    centroids,
    "Qlen d_head, n_clusters d_head -> Qlen n_clusters",
)
argmax_cluster = torch.argmax(aligned, dim=-1)
argmax_cluster.shape, argmax_cluster, aligned
# %%
centroids[5], k_0_clusters[cluster_assignments == 5].shape
# %%

# %%
# TODO: apply rope to the ks we retrieve from the clusters
# need k_retrieved

cluster_alignment = einsum(
    q_0,
    k_approx,
    "B Qlen n_kv n_q_per_kv d_head, B n_kv n_q_per_kv d_head -> B n_kv n_q_per_kv Qlen",
)
cluster_alignment
