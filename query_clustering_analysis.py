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
    dot_products = defaultdict(list)
    query_data = defaultdict(list)
    for layer_idx, (name, samples_out) in enumerate(all_intermediates.items()):
        for qkv in samples_out:
            qkv = qkv.squeeze(0)
            split_size = qkv.size(-1) // 3
            q, k, v = torch.split(qkv, split_size, dim=-1)
            q = q.view(-1, num_heads, head_dim)

            for head_idx in range(num_heads):
                q_head = q[:, head_idx, :]

                dots = torch.matmul(q_head, random_vectors[layer_idx].T)
                max_dots, max_indices = torch.max(dots, dim=1)
                for vec_idx in range(num_random_vectors):
                    mask = max_indices == vec_idx
                    if mask.any():
                        update = torch.mean(q_head[mask], dim=0)
                        random_vectors[layer_idx, vec_idx] = update_ema(
                            random_vectors[layer_idx, vec_idx],
                            update,
                            ema_alpha,
                        )

                dot_products[(layer_idx, head_idx)].extend(max_dots.tolist())
                query_data[(layer_idx, head_idx)].append(
                    (q_head.detach().cpu().numpy(), max_indices.detach().cpu().numpy())
                )

    return dot_products, query_data


# %%
def plot_cluster(layer_idx, head_idx, query_data, num_clusters):
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


def plot_query_clusters(query_data, model, num_heads, num_random_vectors):
    for layer_idx, head_idx in query_data.keys():
        plot_cluster(layer_idx, head_idx, query_data, num_random_vectors)


# %%
def main():
    num_samples = 10
    num_random_vectors = 4
    num_heads = 12
    head_dim = 64
    min_seq_len = 1024
    ema_alpha = 0.1

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)

    random_vectors = torch.randn(model.config.n_layer, num_random_vectors, head_dim)

    dataset = get_dataset(DatasetTypes.INTERNET)
    all_intermediates = process_dataset(
        dataset, tokenizer, model, num_samples, min_seq_len
    )
    dot_products, query_data = process_intermediates(
        all_intermediates,
        random_vectors,
        ema_alpha,
        num_heads,
        head_dim,
        num_random_vectors,
    )
    sampled_data = random.sample(sorted(query_data), min(24, len(query_data)))
    query_data = {key: query_data[key] for key in sorted(sampled_data)}
    plot_query_clusters(query_data, model, num_heads, num_random_vectors)


# %%
if __name__ == "__main__":
    main()

# %%
