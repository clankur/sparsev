# %%
import math
import torch
from einops import einsum, rearrange
from sklearn.cluster import KMeans
import utils
import importlib
from modeling_llama import LlamaForCausalLM
import torch.nn.functional as F
import numpy as np

# %%
importlib.reload(utils)
from utils import load_layer_projection, ModelTypes

# %%
model_type = ModelTypes.LLAMA
model = LlamaForCausalLM.from_pretrained(model_type.value)
config = model.config
# %%
n_clusters = 50
clustering_with = "k"  #  OPTIONS = ["k", "independent_q_k", "avg_wei_k"]
layer_idx = 15
# %%
n_kv_heads = model.config.num_key_value_heads
n_q_per_kv = model.config.num_attention_heads // model.config.num_key_value_heads
d_head = model.config.head_dim

q = load_layer_projection(layer_idx, "q_proj")
k = load_layer_projection(layer_idx, "k_proj")
q = rearrange(
    q,
    "seq_len B Qlen (n_heads d_head) -> B n_heads (seq_len Qlen) d_head",
    n_heads=n_kv_heads * n_q_per_kv,
    d_head=d_head,
)

k = rearrange(
    k,
    "seq_len B Klen (n_kv d_head) -> B n_kv (seq_len Klen)  d_head",
    n_kv=n_kv_heads,
    d_head=d_head,
)

q = rearrange(
    q,
    "B (n_kv n_q_per_kv) Qlen d_head -> B n_kv n_q_per_kv Qlen d_head",
    n_kv=n_kv_heads,
    n_q_per_kv=n_q_per_kv,
)


logits = einsum(
    q,
    k,
    "B n_kv n_q_per_kv Qlen d_head, B n_kv Klen d_head -> B n_kv n_q_per_kv Qlen Klen",
) / math.sqrt(d_head)

att_wei = F.softmax(logits, dim=-1)


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
        centroids, cluster_assignments: Tensors with appropriate shapes based on activation_type
    """
    device = activation.device
    shape = activation.shape
    data_cpu = activation.cpu().numpy()

    if activation_type == "q_proj":
        B, n_kv, n_q_per_kv, L, d_head = shape
        dims_to_iterate = (B, n_kv, n_q_per_kv)
        centroids = torch.zeros(B, n_kv, n_q_per_kv, n_clusters, d_head, device=device)
        cluster_assignments = torch.zeros(
            B, n_kv, n_q_per_kv, L, dtype=torch.long, device=device
        )
    else:  # k_proj
        B, n_kv, L, d_head = shape
        dims_to_iterate = (B, n_kv)
        centroids = torch.zeros(B, n_kv, n_clusters, d_head, device=device)
        cluster_assignments = torch.zeros(B, n_kv, L, dtype=torch.long, device=device)

    for idx in np.ndindex(*dims_to_iterate):
        data_2d = data_cpu[idx]  # This will automatically get the correct slice
        kmeans_model = KMeans(
            n_clusters=n_clusters, max_iter=max_iters, tol=tolerance, random_state=seed
        )
        cluster_assignments[idx] = torch.from_numpy(
            kmeans_model.fit_predict(data_2d)
        ).to(device)
        centroids[idx] = torch.from_numpy(kmeans_model.cluster_centers_).to(device)

    return centroids, cluster_assignments


# %%
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
        # B, n_kv, n_q_per_kv, n_clusters, d_head
    elif clustering_with == "k":
        centroids, cluster_assignments = kmeans(k, "k_proj", n_clusters=n_clusters)
        cluster_alignment = einsum(
            q,
            centroids,
            "B n_kv n_q_per_kv Qlen d_head, B n_kv n_clusters d_head -> B n_kv n_q_per_kv Qlen n_clusters",
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


# %%
cluster_alignment, cluster_assignments = cluster_qk(att_wei, k, q)


# TODO: for k and avg_wei_k:
# for each query, find the top X k-clusters the align most with it
#   for that query, calculate the probability given those keys from each cluster
# do this also with the top X avg_wei_k-clusters

# TODO: for independent_q_k,
# for each q-cluster find the top X k-clusters that aligns most with it
#   grab the selection of queries from the q-cluster
#   for each of the top X k-clusters, calculate the probability given those keys from each cluster for each of the queries in the q-cluster


# %%
def calculate_cluster_alignment(cluster_alignment, cluster_assignments, att_wei):
    global results
    # Calculate cluster alignment
    argmax_cluster = torch.argmax(cluster_alignment, dim=-1)

    # Get dimensions
    B, n_kv, n_q_per_kv, L = argmax_cluster.shape

    # B n_kv n_q_per_kv Klen n_clusters

    # Calculate attention-based relevant keys
    sorted_weights, sorted_indices = att_wei.sort(dim=-1, descending=True)
    cumsum_weights = torch.cumsum(sorted_weights, dim=-1)
    top_relevant_keys = cumsum_weights <= 0.8
    top_relevant_keys[..., 0] = True

    total_probs = torch.zeros(B, n_kv, n_q_per_kv, L)
    precision_scores = torch.zeros(B, n_kv, n_q_per_kv, L)
    recall_scores = torch.zeros(B, n_kv, n_q_per_kv, L)

    # Calculate cluster relevant indices
    cluster_relevant_keys = torch.zeros_like(att_wei, dtype=torch.bool)
    b_indices = torch.arange(B)
    kv_indices = torch.arange(n_kv)
    q_indices = torch.arange(n_q_per_kv)
    q_indices = torch.arange(L)
    pred_cluster = argmax_cluster[b_indices, q_indices]

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
