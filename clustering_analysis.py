# %%
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from collections import defaultdict
import json

# %% [markdown]
# ### TODO: setup
# - Capture the k, v ,q projections
# - determine a way to setup


# %%
class DatasetTypes(Enum):
    WIKI = ("wikitext", "wikitext-2-raw-v1")
    INTERNET = ("allenai/c4", "en")
    CODE = "bigcode/starcoderdata"
    ASSISTANT = "HuggingFaceH4/ultrachat_200k"


def get_dataset(dataset_type: DatasetTypes):
    if dataset_type == DatasetTypes.WIKI or dataset_type == DatasetTypes.INTERNET:
        return load_dataset(
            dataset_type.value[0], dataset_type.value[1], streaming=True
        )
    return load_dataset(dataset_type.value, streaming=True)


# %%
dataset = get_dataset(DatasetTypes.INTERNET)

# %%
# Load tokenizer and model
num_samples = 2
max_clusters = 10
num_heads = 12
head_dim = 64
min_seq_len = 1024

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
# %%
all_intermediates = defaultdict(list)


def get_activation(name):
    def hook(module, input, output):
        all_intermediates[name].append(output.detach())

    return hook


# Register hooks for each attention layer
for name, module in model.named_modules():
    if "attn.c_attn" in name:
        module.register_forward_hook(get_activation(name))

# %%
stream = iter(dataset["train"])

# %%
for i in range(num_samples):
    input_text = next(stream)["text"]
    combined_text = input_text
    inputs = tokenizer(combined_text, return_tensors="pt")
    sequence_length = inputs.input_ids.shape[1]

    while sequence_length < min_seq_len:
        combined_text += " " + input_text
        inputs = tokenizer(combined_text, return_tensors="pt")
        sequence_length = inputs.input_ids.shape[1]

    # cap length of input to min sequence length tokens
    inputs_sliced = {
        "input_ids": inputs.input_ids[:, :min_seq_len],
        "attention_mask": inputs.attention_mask[:, :min_seq_len],
    }

    sequence_length = inputs_sliced["input_ids"].shape[1]

    # Run the model
    with torch.no_grad():
        outputs = model(**inputs_sliced)
# %%
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# %%
projs_per_layer = [[[]] * num_heads] * model.config.n_layer
# %%
for i, (name, samples_out) in enumerate(all_intermediates.items()):
    for qkv in samples_out:
        qkv = qkv.squeeze(0)
        # The c_attn output contains q, k, v concatenated
        split_size = qkv.size(-1) // 3
        assert split_size == head_dim * num_heads
        q, k, v = torch.split(qkv, split_size, dim=-1)

        q = q.view(-1, num_heads, head_dim)
        for j in range(num_heads):
            q_head = q[:, j, :]
            projs_per_layer[i][j].append(q_head.numpy())
# %%
import multiprocessing as mp

# %%


def process_head(args):
    i, j, k, q_head = args
    print(
        f"Processing query projections for layer {i} head {j} for cluster of size {k}"
    )
    q_proj = np.vstack(q_head)
    scaler = StandardScaler()
    q_proj_scaled = scaler.fit_transform(q_proj)

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(q_proj_scaled)
    score = silhouette_score(q_proj_scaled, kmeans.labels_)

    return i, j, k, kmeans.inertia_, score, q_proj_scaled


def parallel_clustering(projs_per_layer, max_clusters):
    pool = mp.Pool(processes=mp.cpu_count())

    args = [
        (i, j, k, q_head)
        for i, layer_projs in enumerate(projs_per_layer)
        for j, q_head in enumerate(layer_projs)
        for k in range(2, max_clusters + 1)
    ]

    results = pool.map(process_head, args)

    pool.close()
    pool.join()

    organized_results = defaultdict(
        lambda: {"inertias": [], "scores": [], "optimal_scores": float("-inf")}
    )
    for i, j, k, inertia, score, q_proj_scaled in results:
        organized_results[(i, j)]["inertias"].append(inertia)
        organized_results[(i, j)]["scores"].append(score)
        if organized_results[(i, j)]["optimal_score"] < score:
            organized_results[(i, j)]["optimal_score"] = score
            organized_results[(i, j)]["optimal_k"] = k
        if "data" not in organized_results[(i, j)]:
            organized_results[(i, j)]["data"] = q_proj_scaled

    for i, j in organized_results.keys():
        optimal = organized_results[(i, j)]["optimal_k"]
        kmeans = KMeans(n_clusters=optimal, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(q_proj_scaled)

    pca = PCA(n_components=2)
    queries_2d = pca.fit_transform(q_proj_scaled)


# %%
for i, layer_projs in enumerate(projs_per_layer):
    for j, q_head in enumerate(layer_projs):
        print(f"query projections for layer {i} head {j}")
        q_proj = np.vstack(q_head)
        # Standardize the data
        scaler = StandardScaler()
        q_proj_scaled = scaler.fit_transform(q_proj)

        # Compute inertias and silhouette scores for different numbers of clusters
        inertias = []
        kmeans_silhouette_scores = []

        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(q_proj_scaled)
            inertias.append(kmeans.inertia_)
            kmeans_silhouette_scores.append(
                silhouette_score(q_proj_scaled, kmeans.labels_)
            )

        # Plot elbow curve
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(range(2, max_clusters + 1), inertias, marker="o")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.title("Elbow Method")

        # Plot silhouette scores
        plt.subplot(122)
        plt.plot(
            range(2, max_clusters + 1),
            kmeans_silhouette_scores,
            marker="o",
            label="K-means",
        )

        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Scores")
        plt.legend()
        plt.show()

        plt.tight_layout()
        plt.show()

        # Determine optimal number of clusters for each method
        kmeans_optimal = (
            kmeans_silhouette_scores.index(max(kmeans_silhouette_scores)) + 2
        )

        print(f"Optimal clusters (K-means): {kmeans_optimal}")

        # Perform clustering with optimal number and visualize
        kmeans = KMeans(n_clusters=kmeans_optimal, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(q_proj_scaled)

        # Use PCA to reduce to 2D for visualization
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        queries_2d = pca.fit_transform(q_proj_scaled)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            queries_2d[:, 0], queries_2d[:, 1], c=kmeans_labels, cmap="viridis"
        )
        plt.colorbar(scatter)
        plt.title(
            f"Query Clusters for layer {i} head {j}\n(Optimal clusters: {kmeans_optimal})"
        )
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.show()

# %%
1024 * 768

# %%
