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
from sklearn.cluster import KMeans

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
n_samples = 5
seq_len = 256
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
    activations[i]["att_wei"] = attentions.cpu()
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
# %%
activations[1]["q_proj"].shape, activations[1]["k_proj"].shape, activations[1][
    "v_proj"
].shape, activations[1]["att_wei"].shape
# %%
layer_idx = 3

n_q_per_kv = model.config.num_attention_heads // model.config.num_key_value_heads
q_0 = rearrange(
    torch.stack([layer_intermediates["q_proj"] for layer_intermediates in activations]),
    "n_samples L B Qlen (n_kv n_q_per_kv d_head) ->  L ( n_samples B )  Qlen n_kv n_q_per_kv d_head",
    n_kv=model.config.num_key_value_heads,
    n_q_per_kv=n_q_per_kv,
)[layer_idx]
k_0 = rearrange(
    torch.stack([layer_intermediates["k_proj"] for layer_intermediates in activations]),
    "n_samples L B Klen (n_kv d_head) -> L ( n_samples B ) Klen n_kv d_head",
    n_kv=model.config.num_key_value_heads,
)[layer_idx]
logits_0 = rearrange(
    torch.stack(
        [layer_intermediates["att_wei"] for layer_intermediates in activations]
    ),
    "n_samples L B (n_kv n_q) Qlen Klen -> L ( n_samples B ) n_kv n_q Qlen Klen",
    n_kv=model.config.num_key_value_heads,
)[layer_idx]
q_0.shape, k_0.shape, logits_0.shape

# %%
# k_clusters = einsum(
#     logits_0,
#     k_0,
#     "n_samples B n_kv n_q_per_kv Qlen Klen, n_samples B Klen n_kv d_head -> n_samples B n_kv n_q_per_kv d_head Qlen",
# )
# %%
# essentially we find for each head and for each batch, which k we should use, by getting the argmax of the q_clusters
# k_labels = torch.argmax(k_clusters, dim=-1)
# %%
# _, B, n_kv, n_q_per_kv, d_head, Klen = k_clusters.shape
# sample_indices = (
#     torch.arange(n_samples)
#     .view(n_samples, 1, 1, 1, 1)
#     .expand(n_samples, B, n_kv, n_q_per_kv, d_head)
# )
# batch_indices = (
#     torch.arange(B).view(B, 1, 1, 1).expand(n_samples, B, n_kv, n_q_per_kv, d_head)
# )
# kv_indices = (
#     torch.arange(n_kv)
#     .view(1, n_kv, 1, 1)
#     .expand(n_samples, B, n_kv, n_q_per_kv, d_head)
# )
# q_indices = (
#     torch.arange(n_q_per_kv)
#     .view(1, 1, n_q_per_kv, 1)
#     .expand(n_samples, B, n_kv, n_q_per_kv, d_head)
# )
# head_indices = (
#     torch.arange(d_head)
#     .view(1, 1, 1, d_head)
#     .expand(n_samples, B, n_kv, n_q_per_kv, d_head)
# )
# k_approx = k_clusters[
#     sample_indices, batch_indices, kv_indices, q_indices, head_indices, k_labels
# ]
# we have the most important k and q for each head
# k_approx.shape
# %%
# what we can do is use k_clusters/q_clusters to ground some centroids
# and use k means to adjust them over each example
# should we use k_approx or a
# %%
# simplify and work with 1 head
head_idx = 0
q_idx = 0
k_0_0 = k_0[:, 0, :, head_idx, :]  #  Klen x d_head
q_0_0 = q_0[:, 0, :, head_idx, q_idx, :]  # Qlen x d_head
logits_0_0 = logits_0[:, 0, head_idx, q_idx, :, :]  # Qlen x Klen
k_0_0, k_0_0.shape, q_0_0.shape, logits_0_0.shape

# TODO: maybe want to compare using non-roped q and k
k_0_clusters = einsum(
    logits_0_0,
    k_0_0,
    "n_samples Qlen Klen, n_samples Klen d_head -> n_samples Qlen d_head",
)


# %%
def kmeans(data, n_clusters, max_iters=50, tolerance=1e-4, seed=42):
    """
    Performs K-Means clustering using scikit-learn.

    Args:
        data (torch.Tensor): Input tensor of shape (n_samples, Qlen, d_head)
        n_clusters (int): Number of clusters
    Returns:
        centroids (torch.Tensor): shape (n_samples, n_clusters, d_head)
        cluster_assignments (torch.Tensor): shape (n_samples, Qlen)
    """
    n_samples, Qlen, d_head = data.shape
    device = data.device

    data_cpu = data.cpu().numpy()

    # Initialize outputs
    centroids = torch.zeros(n_samples, n_clusters, d_head, device=device)
    cluster_assignments = torch.zeros(n_samples, Qlen, dtype=torch.long, device=device)

    # Perform kmeans for each sample
    for i in range(n_samples):
        kmeans = KMeans(
            n_clusters=n_clusters, max_iter=max_iters, tol=tolerance, random_state=seed
        )
        cluster_assignments[i] = torch.from_numpy(kmeans.fit_predict(data_cpu[i])).to(
            device
        )
        centroids[i] = torch.from_numpy(kmeans.cluster_centers_).to(device)

    return centroids, cluster_assignments


# %%
n_clusters = 10
centroids, cluster_assignments = kmeans(k_0_clusters, n_clusters)
for i in range(n_clusters):
    print(centroids[:, i, :].shape)

# %%
cluster_assignments.shape, k_0_clusters.shape, centroids.shape, cluster_assignments
# %%
# need to apply rope
aligned = einsum(
    q_0_0,
    centroids,
    "n_samples Qlen d_head, n_samples n_clusters d_head -> n_samples Qlen n_clusters",
)
# for each query, getting the cluster that aligns the most
argmax_cluster = torch.argmax(aligned, dim=2)
print(argmax_cluster.shape, argmax_cluster, aligned)
# %%
cumsum_attwei, sorted_indices = logits_0_0.sort(dim=-1, descending=True)
cumsum_attwei = torch.cumsum(cumsum_attwei, dim=-1)
cumsum_attwei, cumsum_attwei.shape
# %%
sorted_indices

# %%
