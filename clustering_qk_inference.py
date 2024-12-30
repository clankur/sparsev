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
initial_seq_len = 256
max_seq_len = 1024
batch_size = 2
n_samples = max(1, 10 // batch_size)
n_clusters = 50
clustering_with = "q"  # ["q", "independent_q_k", "avg_wei_k"]

# %%
model_name = model_type.value
tokenizer = AutoTokenizer.from_pretrained(model_name)
stream = get_dataset(dataset_type, tokenizer, initial_seq_len, batch_size)
# %%
model = LlamaForCausalLM.from_pretrained(model_name)
model = model.to(device)
config = model.config
config
# %%


# %%
def kmeans(
    activation, activation_type, n_clusters, max_iters=50, tolerance=1e-4, seed=42
):
    """
    Performs K-Means clustering using scikit-learn.

    Args:
        activation (torch.Tensor): Input tensor of shape (B, n_kv, n_q_per_kv, L, d_head) or (B, n_kv, L, d_head)
        activation_type (str): "q_proj" or "k_proj"
        n_clusters (int): Number of clusters
    Returns:
        centroids (torch.Tensor): shape (B, n_kv, n_q_per_kv, n_clusters, d_head) or (B, n_kv, n_clusters, d_head)
        cluster_assignments (torch.Tensor): For each item in data, it gives what cluster it is in. shape (B, n_kv, n_q_per_kv, L) or (B, n_kv, L)
    """
    if activation_type == "q_proj":
        B, n_kv, n_q_per_kv, L, d_head = activation.shape
        centroids = torch.zeros(B, n_kv, n_q_per_kv, n_clusters, d_head, device=device)
        cluster_assignments = torch.zeros(
            B, n_kv, n_q_per_kv, L, dtype=torch.long, device=device
        )

    elif activation_type == "k_proj":
        B, n_kv, L, d_head = activation.shape
        centroids = torch.zeros(B, n_kv, n_clusters, d_head, device=device)
        cluster_assignments = torch.zeros(B, n_kv, L, dtype=torch.long, device=device)

    device = activation.device

    data_cpu = activation.cpu().numpy()

    # Initialize outputs
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
model.attention_intermediates = {}
n_kv_heads = model.config.num_key_value_heads
n_q_per_kv = model.config.num_attention_heads // model.config.num_key_value_heads
d_head = model.config.head_dim

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

    if output.shape[2] % 256 == 0:
        if proj_type in saved_projs:
            model.attention_intermediates[proj_type].append(output.cpu())
        if proj_type == "roped_q_proj":
            print(f"{proj_type}={output.shape}")
        if proj_type == "roped_k_proj":
            q = model.attention_intermediates["roped_q_proj"][-1].cuda()
            k = model.attention_intermediates["roped_k_proj"][-1].cuda()
            q = rearrange(
                q,
                "B (n_kv n_q_per_kv) Qlen d_head -> B n_kv n_q_per_kv Qlen d_head",
                n_kv=n_kv_heads,
                n_q_per_kv=n_q_per_kv,
            )
            print(f"{q.shape=}", f"{k.shape=}")
            logits = einsum(
                q,
                k,
                "B n_kv n_q_per_kv Qlen d_head, B n_kv Klen d_head -> B n_kv n_q_per_kv Qlen Klen",
            ) / (d_head**0.5)
            # no causal mask because this is inference
            att_wei = torch.softmax(logits, dim=-1)
            if "att_wei" not in model.attention_intermediates:
                model.attention_intermediates["att_wei"] = []
            model.attention_intermediates["att_wei"].append(att_wei.cpu())

    return output


def cluster_qk(att_wei, k, q):
    global n_clusters, clustering_with
    if clustering_with == "avg_wei_k":
        # Calculate average weighted key for each query
        avg_wei_k = einsum(
            att_wei,
            k,
            "B n_kv n_q_per_kv Qlen Klen, B n_kv Klen d_head -> B n_kv n_q_per_kv Qlen d_head",
        )
        centroids, cluster_assignments = kmeans(
            avg_wei_k, "q_proj", n_clusters=n_clusters
        )
        cluster_alignment = einsum(
            q,
            centroids,
            "B n_kv n_q_per_kv Qlen d_head, B n_kv n_q_per_kv n_clusters d_head -> B n_kv n_q_per_kv Qlen n_clusters",
        )
    elif clustering_with == "q":
        centroids, cluster_assignments = kmeans(q, "q_proj", n_clusters=n_clusters)
        q_cluster_alignment = einsum(
            q,
            centroids,
            "B n_kv n_q_per_kv Qlen d_head, B n_kv n_q_per_kv n_clusters d_head -> B n_kv n_q_per_kv Qlen n_clusters",
        )
        cluster_alignment = einsum(
            k,
            centroids,
            "B n_kv Klen d_head, B n_kv n_clusters d_head -> B n_kv Klen n_clusters",
        )
    elif clustering_with == "independent_q_k":
        q_centroids, q_cluster_assignments = kmeans(q, "q_proj", n_clusters=n_clusters)
        k_centroids, k_cluster_assignments = kmeans(k, "k_proj", n_clusters=n_clusters)
        q_cluster_alignment = einsum(
            q,
            k_centroids,
            "B n_kv n_q_per_kv Qlen d_head, B n_kv n_clusters d_head -> B n_kv n_q_per_kv Qlen n_clusters",
        )
        k_cluster_alignment = einsum(
            k,
            q_centroids,
            "B n_kv Klen d_head, B n_kv n_q_per_kv Qlen d_head -> B n_kv n_q_per_kv Klen n_clusters",
        )
        return q_cluster_alignment, k_cluster_alignment
    else:
        raise ValueError(f"Invalid clustering_with: {clustering_with}")

    return cluster_alignment, cluster_assignments


# Register hooks for all attention layers
for name, module in model.named_modules():
    if any(proj in name.lower() for proj in saved_projs):
        module.register_forward_hook(attention_forward_hook)


# %%
def calculate_cluster_alignment(cluster_alignment, cluster_assignments, att_wei):
    global results
    # Calculate cluster alignment
    argmax_cluster = torch.argmax(cluster_alignment, dim=-1)

    # Get dimensions
    B, n_kv, n_q_per_kv, L = argmax_cluster.shape

    # Calculate cluster relevant indices
    cluster_relevant_indices = [[] for _ in range(B)]
    total_probs = torch.zeros(B, n_kv, n_q_per_kv, L)

    # Calculate attention-based relevant keys
    sorted_weights, sorted_indices = att_wei.sort(dim=-1, descending=True)
    cumsum_weights = torch.cumsum(sorted_weights, dim=-1)
    top_weight_mask = cumsum_weights <= 0.8
    top_weight_mask[..., 0] = True
    relevant_keys = [[[] for _ in range(n_q_per_kv)] for _ in range(n_kv)] * B

    precision_scores = torch.zeros(B, n_kv, n_q_per_kv, L)
    recall_scores = torch.zeros(B, n_kv, n_q_per_kv, L)

    for b in range(B):
        for q in range(L):
            pred_cluster = argmax_cluster[b, q]
            key_indices = torch.where(cluster_assignments[b] == pred_cluster)[0]
            cluster_relevant_indices[b].append(key_indices)
            total_probs[b, :, :, q] = att_wei[b, :, :, q, key_indices].sum(dim=-1)

            key_indices = sorted_indices[b, q, :][top_weight_mask[b, q, :]]
            relevant_keys[b].append(key_indices)

    # Calculate precision and recall
    for b in range(B):
        for q in range(L):
            true_relevant = set(relevant_keys[b][q].tolist())
            pred_relevant = set(cluster_relevant_indices[b][q].tolist())

            if len(pred_relevant) == 0 or len(true_relevant) == 0:
                continue

            correct_predictions = len(true_relevant.intersection(pred_relevant))
            precision = correct_predictions / len(pred_relevant)
            recall = correct_predictions / len(true_relevant)

            precision_scores[b, :, :, q] = precision
            recall_scores[b, :, :, q] = recall

    # Store average metrics for this layer, head, and query
    avg_precision = precision_scores.mean().item()
    avg_recall = recall_scores.mean().item()
    f1_score = (
        2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0
    )

    # results["precision"][layer_idx, head_idx, q_idx] = avg_precision
    # results["recall"][layer_idx, head_idx, q_idx] = avg_recall
    # results["f1"][layer_idx, head_idx, q_idx] = f1_score

    # print(f"Layer {layer_idx}, KV Head {head_idx}, Query Idx {q_idx}:")
    print(f"  Precision: {avg_precision:.3f}")
    print(f"  Recall: {avg_recall:.3f}")
    print(f"  F1: {f1_score:.3f}")

    # Store the average total_prob for this layer, head, and query
    # results["total_prob"][layer_idx, head_idx, q_idx, :] = total_probs.mean(dim=0)
    print(f"  Average Prob across all queries: {total_probs.mean().item():.3f}")
    print(f"  Min Prob across all queries: {total_probs.min().item():.3f}")
    print(f"  Max Prob across all queries: {total_probs.max().item():.3f}")


# %%
activations = [{} for _ in range(n_samples)]
for i in range(n_samples):
    cur_seq_len = 0

    inputs = next(stream)
    inputs_sliced = {"input_ids": torch.stack(inputs).to(device)}

    model.generate(inputs_sliced["input_ids"], max_new_tokens=max_seq_len + 1)


# %%
model.attention_intermediates.keys()

# %%
model.attention_intermediates["roped_k_proj"][-1].shape, model.attention_intermediates[
    "roped_q_proj"
][-1].shape
# %%
for att_wei in model.attention_intermediates["att_wei"]:
    print(att_wei.shape)

# %%
