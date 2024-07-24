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
# ### TODO: Load in datasets to see how to sparsify v for
# - One coding dataset
# - One wiki dataset
# - One general internet dataset
# - One dialogue/assistant dataset
#
# ### TODO: Report the percentage of logits needed for cumsum to be 80, 90, 95, 99%
# - if you keep the top n% for n in 1, 5, 10, 20, what are their percentages
# ### TODO: Run metrics across different sequence lengths
# - Find max seq length and use prefixes of it
#   -  also filter out sequences under a certain length
# - report metrics at intermediate sequence lengths
#   - ex). on __ sequences, these are the metrics for __ sequence length
# - Track worst case (sparsity) - when the data is not sparse b/c all the logits are close
#   - more evenly distrubuted logits
#


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
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
# %%
for name, module in model.named_modules():
    print(name)
# %%
num_of_samples = 150

# %%
stream = iter(dataset["train"])

# %%
min_seq_len = 1024
top_ks = [1, 5, 10, 20]
metric_cumsums = [80, 90, 95, 99]
samples_metrics = list()
cumsum_metrics = [
    [
        {
            "best": torch.zeros((min_seq_len)),
            "avg": torch.zeros((min_seq_len)),
            "worst": torch.ones((min_seq_len)),
        }
        for _ in range(model.config.n_head)
    ]
    for _ in range(model.config.n_layer)
]


# %%
# Encode input and create tensors
for i in range(1):
    input_text = next(stream)["text"]
    inputs = tokenizer(input_text, return_tensors="pt")
    sequence_length = inputs.input_ids.shape[1]

    while sequence_length <= min_seq_len:
        inputs = tokenizer(input_text, return_tensors="pt")
        input_text = next(stream)["text"]
        sequence_length = inputs.input_ids.shape[1]

    # cap length of input to min sequence length tokens
    inputs_sliced = {
        "input_ids": inputs.input_ids[:, :min_seq_len],
        "attention_mask": inputs.attention_mask[:, :min_seq_len],
    }

    sequence_length = inputs_sliced["input_ids"].shape[1]

    metrics = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
    )

    # Run the model
    with torch.no_grad():
        outputs = model(**inputs_sliced)

    # Get the attentions
    attentions = outputs.attentions  # Tuple of tensors: one for each layer

    for l, layer_attention in enumerate(attentions):
        for h in range(model.config.n_head):
            att = layer_attention[0, h, -1, :].sort().values
            reverse_att = att.flip(0)
            total_prob = att.cumsum(0)
            reverse_total_prob = reverse_att.cumsum(0)

            cumsum_metrics[l][h]["avg"] += reverse_total_prob
            cumsum_metrics[l][h]["worst"] = torch.min(
                cumsum_metrics[l][h]["worst"], reverse_total_prob
            )
            cumsum_metrics[l][h]["best"] = torch.max(
                cumsum_metrics[l][h]["best"], reverse_total_prob
            )

            for k, total_sum in zip(top_ks, metric_cumsums):
                k_percentile, desired_sum = k / 100, total_sum / 100

                min_elements = torch.nonzero(reverse_total_prob > desired_sum)[0]

                top_percentile_prob = total_prob[-int(k_percentile * sequence_length)]

                prev_precentile = metrics[l][h]["top_percentile_prob"]
                prev_total_sum = metrics[l][h]["min_logits_for_cumsum"]
                prev_precentile[f"{k}%"] = 1 - top_percentile_prob.item()
                prev_total_sum[f"{total_sum}%"] = 1 + min_elements.item()

    samples_metrics.append(metrics)

# %%
print(json.dumps(samples_metrics, indent=4))

# %%
for l in range(model.config.n_layer):
    for h in range(model.config.n_head):
        plt.figure()
        plt.plot(cumsum_metrics[l][h]["worst"], label="worst")
        plt.plot(cumsum_metrics[l][h]["best"], label="best")
        plt.plot(cumsum_metrics[l][h]["avg"] / num_of_samples, label="avg")
        plt.xlabel("number of tokens token")
        plt.ylabel("cumsum of att scores")
        plt.title(f"Cum Att Scores for Layer {l} Head {h}")
        plt.legend()


# %%
def analyze_cases(data):
    """
    Analyze the samples and return the average and worst case values for the top_percentile_prob and min_logits_for_cumsum metrics
    for each layer and head in the model.
    """
    # Initialize dictionaries to store the results
    top_percentile_prob_avg = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0.0))
    )
    top_percentile_prob_worst = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 1.0))
    )
    min_logits_for_cumsum_avg = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0.0))
    )
    min_logits_for_cumsum_worst = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0.0))
    )

    # Iterate over each sample
    for sample in data:
        # Iterate over each layer in the sample
        for layer in sample:
            # Iterate over each head in the layer
            for head in sample[layer]:
                # Extract top_percentile_prob and min_logits_for_cumsum data
                top_percentile_prob_data = sample[layer][head]["top_percentile_prob"]
                min_logits_for_cumsum_data = sample[layer][head][
                    "min_logits_for_cumsum"
                ]

                # for each head, update the average and worst case values
                for key in top_percentile_prob_data:
                    top_percentile_prob_avg[layer][head][
                        key
                    ] += top_percentile_prob_data[key]
                    top_percentile_prob_worst[layer][head][key] = min(
                        top_percentile_prob_worst[layer][head][key],
                        top_percentile_prob_data[key],
                    )

                for key in min_logits_for_cumsum_data:
                    min_logits_for_cumsum_avg[layer][head][
                        key
                    ] += min_logits_for_cumsum_data[key]
                    min_logits_for_cumsum_worst[layer][head][key] = max(
                        min_logits_for_cumsum_worst[layer][head][key],
                        min_logits_for_cumsum_data[key],
                    )

    # Calculate the average values
    for layer in top_percentile_prob_avg:
        for head in top_percentile_prob_avg[layer]:
            for k1, k2 in zip(
                top_percentile_prob_avg[layer][head],
                min_logits_for_cumsum_avg[layer][head],
            ):
                top_percentile_prob_avg[layer][head][k1] /= len(data)
                min_logits_for_cumsum_avg[layer][head][k2] /= len(data)

    return (
        top_percentile_prob_avg,
        top_percentile_prob_worst,
        min_logits_for_cumsum_avg,
        min_logits_for_cumsum_worst,
    )


# %%
(
    top_percentile_prob_avg,
    top_percentile_prob_worst,
    min_logits_for_cumsum_avg,
    min_logits_for_cumsum_worst,
) = analyze_cases(samples_metrics)

# %%
print(json.dumps(top_percentile_prob_avg[0][0], indent=4))
print(json.dumps(top_percentile_prob_worst[0][0], indent=4))
print(json.dumps(min_logits_for_cumsum_avg[0][0], indent=4))
print(json.dumps(min_logits_for_cumsum_worst[0][0], indent=4))


# %%
def plot_histograms(data_avg, data_worst, xlabel, title_template=None):
    # Collect all values for each percentile across all layers and heads
    percentiles = defaultdict(lambda: {"avg": [], "worst": []})
    for layer in data_avg:
        for head in data_avg[layer]:
            for percentile in data_avg[layer][head]:
                percentiles[percentile]["avg"].append(data_avg[layer][head][percentile])
                percentiles[percentile]["worst"].append(
                    data_worst[layer][head][percentile]
                )

    print(json.dumps(percentiles, indent=4))

    # Plot histograms for each percentile
    for percentile in percentiles:
        plt.figure(figsize=(10, 5))
        plt.hist(percentiles[percentile]["avg"], bins=100, alpha=0.5, label="Average")
        plt.hist(percentiles[percentile]["worst"], bins=100, alpha=0.5, label="Worst")
        if title_template:
            plt.title(title_template.format(percentile))
        plt.xlabel(f"{xlabel} {percentile}")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


# %%
plot_histograms(
    top_percentile_prob_avg,
    top_percentile_prob_worst,
    "Largest {} of Logits",
    "Cumulative attention score for largest",
)


# %%
plot_histograms(
    min_logits_for_cumsum_avg,
    min_logits_for_cumsum_worst,
    "Minimum logits for cumulative attention score of",
)
