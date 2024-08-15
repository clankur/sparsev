# %%
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# %%
class DatasetTypes(Enum):
    WIKI = ("wikitext", "wikitext-2-raw-v1")
    INTERNET = ("allenai/c4", "en")
    CODE = "bigcode/starcoderdata"
    ASSISTANT = "HuggingFaceH4/ultrachat_200k"


def get_dataset(dataset_type: DatasetTypes):
    if dataset_type in [DatasetTypes.WIKI, DatasetTypes.INTERNET]:
        return load_dataset(
            dataset_type.value[0], dataset_type.value[1], streaming=True
        )
    return load_dataset(dataset_type.value, streaming=True)


def get_activation(name, all_intermediates):
    def hook(module, input, output):
        all_intermediates[name].append(output.detach())

    return hook


def process_dataset(dataset, tokenizer, model, num_samples, min_seq_len):
    all_intermediates = defaultdict(list)

    for name, module in model.named_modules():
        if "attn.c_attn" in name:
            module.register_forward_hook(get_activation(name, all_intermediates))

    stream = iter(dataset["train"])
    for _ in range(num_samples):
        combined_text = next(stream)["text"]
        inputs = tokenizer(combined_text, return_tensors="pt")

        while inputs.input_ids.shape[1] < min_seq_len:
            combined_text += " " + next(stream)["text"]
            inputs = tokenizer(combined_text, return_tensors="pt")

        inputs_sliced = {
            "input_ids": inputs.input_ids[:, :min_seq_len],
            "attention_mask": inputs.attention_mask[:, :min_seq_len],
        }

        with torch.no_grad():
            model(**inputs_sliced)

    return all_intermediates


# %%
def update_ema(current_value, new_value, alpha):
    return (1 - alpha) * current_value + alpha * new_value


# %%
def process_intermediates(
    all_intermediates,
    random_vectors,
    ema_alpha,
    num_heads,
    head_dim,
    num_random_vectors,
):
    q_dots = defaultdict(list)
    k_dots = defaultdict(list)
    query_data = defaultdict(list)
    key_heads = defaultdict()

    cum_vectors = torch.zeros_like(random_vectors)
    count_vectors = torch.zeros(
        random_vectors.shape[0], random_vectors.shape[1], dtype=torch.long
    )

    for layer_idx, (name, samples_out) in enumerate(all_intermediates.items()):
        for qkv in samples_out:
            qkv = qkv.squeeze(0)
            split_size = qkv.size(-1) // 3
            q, k, v = torch.split(qkv, split_size, dim=-1)
            q, k = [tensor.view(-1, num_heads, head_dim) for tensor in (q, k)]

            for head_idx in range(num_heads):
                q_head, k_head = [tensor[:, head_idx, :] for tensor in (q, k)]

                dots = torch.matmul(q_head, random_vectors[layer_idx].T)
                max_dots, max_indices = torch.max(dots, dim=1)
                for vec_idx in range(num_random_vectors):
                    mask = max_indices == vec_idx
                    if mask.any():
                        # update = torch.mean(q_head[mask], dim=0)
                        cum_vectors[layer_idx, vec_idx] += torch.sum(
                            q_head[mask], dim=0
                        )
                        count_vectors[layer_idx, vec_idx] += torch.sum(mask)

                q_dots[(layer_idx, head_idx)].extend(max_dots.tolist())
                query_data[(layer_idx, head_idx)].append(
                    (q_head.detach().cpu().numpy(), max_indices.detach().cpu().numpy())
                )
                key_heads[(layer_idx, head_idx)] = k_head

    mask = count_vectors > 0
    random_vectors[mask] = cum_vectors[mask] / count_vectors[mask].unsqueeze(-1)

    key_data = defaultdict(list)
    for (layer_idx, head_idx), k_head in key_heads.items():
        dots = torch.matmul(k_head, random_vectors[layer_idx].T)
        max_dots, max_indices = torch.max(dots, dim=1)
        k_dots[(layer_idx, head_idx)].extend(max_dots.tolist())
        key_data[(layer_idx, head_idx)].append(
            (k_head.detach().cpu().numpy(), max_indices.detach().cpu().numpy())
        )

    return q_dots, k_dots, query_data, key_data


# %%
def plot_query_cluster(layer_idx, head_idx, query_data, num_clusters):
    queries = []
    alignments = []
    for q, a in query_data[(layer_idx, head_idx)]:
        queries.append(q)
        alignments.append(a)

    queries = np.vstack(queries)
    alignments = np.concatenate(alignments)

    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    queries_2d = pca.fit_transform(queries)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        queries_2d[:, 0],
        queries_2d[:, 1],
        c=alignments,
        cmap="viridis",
        alpha=0.6,
    )
    plt.colorbar(scatter, label="Aligned Vector Index")
    plt.title(f"Query Clusters - Layer {layer_idx}, Head {head_idx}")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")

    # Add a legend
    handles = [
        plt.scatter(
            [],
            [],
            c=[plt.cm.viridis(i / num_clusters)],
            label=f"Vector {i}",
        )
        for i in range(num_clusters)
    ]
    plt.legend(
        handles=handles,
        title="Aligned Vector",
    )

    plt.tight_layout()
    plt.show()


def plot_3d_query_cluster(layer_idx, head_idx, query_data, num_clusters):
    import plotly.graph_objects as go
    import os

    def ensure_figs_folder():
        if not os.path.exists("figs"):
            os.makedirs("figs")

    ensure_figs_folder()

    queries = []
    alignments = []
    for q, a in query_data[(layer_idx, head_idx)]:
        queries.append(q)
        alignments.append(a)

    queries = np.vstack(queries)
    alignments = np.concatenate(alignments)

    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=3)
    queries_3d = pca.fit_transform(queries)

    max_points = 100000  # Adjust this value based on your system's capabilities
    indices = np.random.choice(
        queries_3d.shape[0], min(max_points, queries_3d.shape[0]), replace=False
    )
    queries_3d_sample = queries_3d[indices]

    fig = go.Figure()
    for i in range(num_clusters):
        mask = alignments == i
        fig.add_trace(
            go.Scatter3d(
                x=queries_3d_sample[mask, 0],
                y=queries_3d_sample[mask, 1],
                z=queries_3d_sample[mask, 2],
                mode="markers",
                name=f"Vector {i}",
                marker=dict(size=5),
            )
        )

    fig.update_layout(
        title=f"Query Clusters for layer {layer_idx} head {head_idx})",
        scene=dict(
            xaxis_title="First Principal Component",
            yaxis_title="Second Principal Component",
            zaxis_title="Third Principal Component",
        ),
        width=900,
        height=700,
    )

    filename = f"figs/cluster_plot_layer_{layer_idx}_head_{head_idx}.html"
    fig.write_html(filename)
    print(f"Plot saved as {filename}")
    fig.show()


def plot_query_clusters(query_data, num_random_vectors):
    for layer_idx, head_idx in query_data.keys():
        plot_3d_query_cluster(layer_idx, head_idx, query_data, num_random_vectors)
        # plot_query_cluster(layer_idx, head_idx, query_data, num_random_vectors)


import os


# %%
def plot_3d_cluster(layer_idx, head_idx, data, num_clusters, data_type):
    def ensure_figs_folder():
        if not os.path.exists("figs"):
            os.makedirs("figs")

    ensure_figs_folder()

    vectors = []
    alignments = []
    for v, a in data[(layer_idx, head_idx)]:
        vectors.append(v)
        alignments.append(a)

    vectors = np.vstack(vectors)
    alignments = np.concatenate(alignments)

    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors)

    max_points = 100000
    print(min(max_points, vectors_3d.shape[0]))
    indices = np.random.choice(
        vectors_3d.shape[0], min(max_points, vectors_3d.shape[0]), replace=False
    )
    vectors_3d_sample = vectors_3d[indices]
    alignments_sample = alignments[indices]
    # Choose color scheme based on data_type
    if data_type == "Query":
        color_scheme = px.colors.sequential.Viridis
    else:  # Key
        color_scheme = px.colors.sequential.Plasma

    for cluster in range(num_clusters):
        mask = alignments_sample == cluster

        if np.sum(mask) == 0:
            print(
                f"No points for cluster {cluster} in {data_type} data, layer {layer_idx}, head {head_idx}"
            )
            continue

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=vectors_3d_sample[mask, 0],
                y=vectors_3d_sample[mask, 1],
                z=vectors_3d_sample[mask, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=vectors_3d_sample[mask, 2],  # Color based on z-axis
                    colorscale=color_scheme,
                    opacity=0.8,
                ),
            )
        )

        fig.update_layout(
            title=f"{data_type} Cluster {cluster} for layer {layer_idx} head {head_idx}",
            scene=dict(
                xaxis_title="First Principal Component",
                yaxis_title="Second Principal Component",
                zaxis_title="Third Principal Component",
            ),
            width=900,
            height=700,
        )

        filename = f"figs/{data_type.lower()}_cluster_{cluster}_plot_layer_{layer_idx}_head_{head_idx}.html"
        fig.write_html(filename)
        print(f"Plot saved as {filename}")
        fig.show()


def plot_clusters(query_data, key_data, num_random_vectors):
    for layer_idx, head_idx in query_data.keys():
        plot_3d_cluster(layer_idx, head_idx, key_data, num_random_vectors, "Key")


# ... (keep the existing code for model loading and data processing)

# Plot clusters for both query and key data

# %%
num_samples = 10
num_random_vectors = 4
num_heads = 12
head_dim = 64
min_seq_len = 1024
ema_alpha = 0.1

# %%
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)

random_vectors = torch.randn(model.config.n_layer, num_random_vectors, head_dim)

dataset = get_dataset(DatasetTypes.INTERNET)
all_intermediates = process_dataset(dataset, tokenizer, model, num_samples, min_seq_len)
# %%
q_dots, k_dots, query_data, key_data = process_intermediates(
    all_intermediates,
    random_vectors,
    ema_alpha,
    num_heads,
    head_dim,
    num_random_vectors,
)
# sampled_data = random.sample(sorted(query_data), min(12, len(query_data)))
# sample = {key: query_data[key] for key in sorted(sampled_data)}


plot_3d_cluster(2, 0, query_data, num_random_vectors, "Query")
plot_3d_cluster(2, 0, key_data, num_random_vectors, "Key")
# %%
plot_3d_cluster(9, 8, query_data, num_random_vectors, "Query")
plot_3d_cluster(9, 8, key_data, num_random_vectors, "Key")

# %%
