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
layer_idx = 5

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
# simplify and work with 1 head
head_idx = 0
q_idx = 0
k_0_0 = k_0[:, :, head_idx, :]  #  B x Klen x d_head
q_0_0 = q_0[:, :, head_idx, q_idx, :]  # B x Qlen x d_head
logits_0_0 = logits_0[:, head_idx, q_idx, :, :]  # B x Qlen x Klen
k_0_0, k_0_0.shape, q_0_0.shape, logits_0_0.shape

# %%
# this is the average weighted key for each query
avg_wei_k = einsum(
    logits_0_0,
    k_0_0,
    "B Qlen Klen, B Klen d_head -> B Qlen d_head",
)


# %%
def kmeans(data, n_clusters, max_iters=50, tolerance=1e-4, seed=42):
    """
    Performs K-Means clustering using scikit-learn.

    Args:
        data (torch.Tensor): Input tensor of shape (B, Qlen, d_head)
        n_clusters (int): Number of clusters
    Returns:
        centroids (torch.Tensor): shape (B, n_clusters, d_head)
        cluster_assignments (torch.Tensor): For each item in data, it gives what cluster it is in. shape (B, Qlen)
    """
    B, Qlen, d_head = data.shape
    device = data.device

    data_cpu = data.cpu().numpy()

    # Initialize outputs
    centroids = torch.zeros(B, n_clusters, d_head, device=device)
    cluster_assignments = torch.zeros(B, Qlen, dtype=torch.long, device=device)

    # Perform kmeans for each sample
    for i in range(B):
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
centroids, cluster_assignments = kmeans(avg_wei_k, n_clusters)
for i in range(n_clusters):
    print(centroids[:, i, :].shape)

# %%
cluster_assignments.shape, avg_wei_k.shape, centroids.shape, cluster_assignments
# %%
cluster_alignment = einsum(
    q_0_0,
    centroids,
    "B Qlen d_head, B n_clusters d_head -> B Qlen n_clusters",
)
# for each query, getting the cluster that aligns the most
argmax_cluster = torch.argmax(cluster_alignment, dim=2)
# argmax_cluster = reduce(aligned, "n_samples Qlen n_clusters -> n_samples Qlen", "max")
argmax_cluster.shape, "argmax_cluster", argmax_cluster
# %%
argmax_cluster.shape, cluster_assignments.shape, k_0_0.shape,
# %%
# lookup the tokens in k_0_0 based on the cluster_assignments and argmax_cluster
# (torch.Size([5, 256]), torch.Size([5, 256]), torch.Size([5, 256, 64]))
# argmax_cluster, cluster_assignments, k_0_0

# %%
# Shape annotations:
# argmax_cluster: [B, Qlen] - for each query position, which cluster it best aligns with
# cluster_assignments: [B, Qlen] - for each query position, which cluster its weighted key combination belongs to
# k_0_0: [B, Klen, d_head] - the key vectors

# Since both argmax_cluster and cluster_assignments are [B, Qlen],
# we can directly compare them
matches = argmax_cluster == cluster_assignments  # [B, Qlen]
matched_vectors = avg_wei_k[matches]
matched_vectors.shape, matched_vectors
matches_per_batch = matches.sum(dim=1)  # [B]
# for each batch, we have the indices of the average weighted keys that are relevant to the query
matching_indices = matches.nonzero()
indices_per_batch = [
    matching_indices[matching_indices[:, 0] == b][:, 1]
    for b in range(batch_size * n_samples)
]

k_approx = torch.split(matched_vectors, matches_per_batch.tolist())
matching_indices
# %%
for i in range(len(k_approx)):
    print(
        f"fetched {k_approx[i].shape[0]} out of {seq_len} keys ({k_approx[i].shape[0]/seq_len*100:.2f}%)"
    )
# %%
# %%
# Shape annotations:
# logits_0_0: [B, Qlen, Klen] - attention weights
B, Qlen, Klen = logits_0_0.shape
# Sort attention weights and get cumulative sum
sorted_weights, sorted_indices = logits_0_0.sort(
    dim=-1, descending=True
)  # Sort each query's attention weights
cumsum_weights = torch.cumsum(
    sorted_weights, dim=-1
)  # Cumulative sum of sorted weights
top_weight_mask = cumsum_weights <= 0.8  # [B, Qlen, Klen]
top_weight_mask[..., 0] = True

# %%
relevant_keys = [[] for _ in range(B)]
print(len(relevant_keys), len(relevant_keys[0]))
for i in range(B):
    for j in range(Qlen):
        key_indices = sorted_indices[i, j, :][top_weight_mask[i, j, :]]
        relevant_keys[i].append(key_indices)
for i in range(B):
    print(len(relevant_keys[i]))
# %%
# for each batch, for each query this has the indices of the keys that are relevant to get att_wei > 0.8
relevant_keys  # B x Qlen x variable n_relevant_keys

# %%
