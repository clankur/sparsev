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
n_samples = 30
seq_len = 512
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
# Setup dimensions
n_layers = len(activations[0]["q_proj"])
n_kv_heads = model.config.num_key_value_heads
n_q_per_kv = model.config.num_attention_heads // model.config.num_key_value_heads

# Initialize results storage
results = {
    "precision": torch.zeros(n_layers, n_kv_heads, n_q_per_kv),
    "recall": torch.zeros(n_layers, n_kv_heads, n_q_per_kv),
    "f1": torch.zeros(n_layers, n_kv_heads, n_q_per_kv),
}
# %%
for layer_idx in range(n_layers):
    # Rearrange activations for this layer
    q_0 = rearrange(
        torch.stack(
            [layer_intermediates["q_proj"] for layer_intermediates in activations]
        ),
        "n_samples L B Qlen (n_kv n_q_per_kv d_head) ->  L ( n_samples B )  Qlen n_kv n_q_per_kv d_head",
        n_kv=n_kv_heads,
        n_q_per_kv=n_q_per_kv,
    )[layer_idx]

    k_0 = rearrange(
        torch.stack(
            [layer_intermediates["k_proj"] for layer_intermediates in activations]
        ),
        "n_samples L B Klen (n_kv d_head) -> L ( n_samples B ) Klen n_kv d_head",
        n_kv=n_kv_heads,
    )[layer_idx]

    logits_0 = rearrange(
        torch.stack(
            [layer_intermediates["att_wei"] for layer_intermediates in activations]
        ),
        "n_samples L B (n_kv n_q) Qlen Klen -> L ( n_samples B ) n_kv n_q Qlen Klen",
        n_kv=n_kv_heads,
    )[layer_idx]

    for head_idx in range(n_kv_heads):
        for q_idx in range(n_q_per_kv):
            # Extract specific head and query data
            k_0_0 = k_0[:, :, head_idx, :]  # B x Klen x d_head
            q_0_0 = q_0[:, :, head_idx, q_idx, :]  # B x Qlen x d_head
            logits_0_0 = logits_0[:, head_idx, q_idx, :, :]  # B x Qlen x Klen

            # Calculate average weighted key for each query
            avg_wei_k = einsum(
                logits_0_0,
                k_0_0,
                "B Qlen Klen, B Klen d_head -> B Qlen d_head",
            )

            # Calculate kmeans
            n_clusters = 10
            centroids, cluster_assignments = kmeans(avg_wei_k, n_clusters)

            # Calculate cluster alignment
            cluster_alignment = einsum(
                q_0_0,
                centroids,
                "B Qlen d_head, B n_clusters d_head -> B Qlen n_clusters",
            )
            argmax_cluster = torch.argmax(cluster_alignment, dim=2)

            # Get dimensions
            B, Qlen = argmax_cluster.shape

            # Calculate cluster relevant indices
            cluster_relevant_indices = [[] for _ in range(B)]
            for b in range(B):
                for q in range(Qlen):
                    pred_cluster = argmax_cluster[b, q]
                    key_indices = torch.where(cluster_assignments[b] == pred_cluster)[0]
                    cluster_relevant_indices[b].append(key_indices)

            # Calculate attention-based relevant keys
            sorted_weights, sorted_indices = logits_0_0.sort(dim=-1, descending=True)
            cumsum_weights = torch.cumsum(sorted_weights, dim=-1)
            top_weight_mask = cumsum_weights <= 0.8
            top_weight_mask[..., 0] = True

            relevant_keys = [[] for _ in range(B)]
            for b in range(B):
                for q in range(Qlen):
                    key_indices = sorted_indices[b, q, :][top_weight_mask[b, q, :]]
                    relevant_keys[b].append(key_indices)

            # Calculate precision and recall
            precision_scores = torch.zeros(B, Qlen)
            recall_scores = torch.zeros(B, Qlen)
            for b in range(B):
                for q in range(Qlen):
                    true_relevant = set(relevant_keys[b][q].tolist())
                    pred_relevant = set(cluster_relevant_indices[b][q].tolist())

                    if len(pred_relevant) == 0 or len(true_relevant) == 0:
                        continue

                    correct_predictions = len(true_relevant.intersection(pred_relevant))
                    precision = correct_predictions / len(pred_relevant)
                    recall = correct_predictions / len(true_relevant)

                    precision_scores[b, q] = precision
                    recall_scores[b, q] = recall

            # Store average metrics for this layer, head, and query
            avg_precision = precision_scores.mean().item()
            avg_recall = recall_scores.mean().item()
            f1_score = (
                2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                if (avg_precision + avg_recall) > 0
                else 0
            )

            results["precision"][layer_idx, head_idx, q_idx] = avg_precision
            results["recall"][layer_idx, head_idx, q_idx] = avg_recall
            results["f1"][layer_idx, head_idx, q_idx] = f1_score

            print(f"Layer {layer_idx}, Head {head_idx}, Query {q_idx}:")
            print(f"  Precision: {avg_precision:.3f}")
            print(f"  Recall: {avg_recall:.3f}")
            print(f"  F1: {f1_score:.3f}")

# %%
# After the main loop, calculate overall metrics
overall_precision = results["precision"].mean().item()
overall_recall = results["recall"].mean().item()
overall_f1 = results["f1"].mean().item()

print("\nOverall Metrics:")
print(f"Average Precision across all layers/heads: {overall_precision:.3f}")
print(f"Average Recall across all layers/heads: {overall_recall:.3f}")
print(f"Average F1 Score across all layers/heads: {overall_f1:.3f}")

# Optional: You can also look at per-layer averages
per_layer_precision = results["precision"].mean(dim=(1, 2))  # Average across heads
per_layer_recall = results["recall"].mean(dim=(1, 2))
per_layer_f1 = results["f1"].mean(dim=(1, 2))

print("\nPer-layer averages:")
for layer_idx in range(n_layers):
    print(f"Layer {layer_idx}:")
    print(f"  Precision: {per_layer_precision[layer_idx]:.3f}")
    print(f"  Recall: {per_layer_recall[layer_idx]:.3f}")
    print(f"  F1: {per_layer_f1[layer_idx]:.3f}")
