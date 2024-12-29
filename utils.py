import torch
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from enum import Enum
from typing import Tuple, Dict, Any
import functools
from torch.utils.data import DataLoader
import numpy as np
from einops import rearrange, einops


class DatasetTypes(Enum):
    WIKI = ("Salesforce/wikitext", "wikitext-103-raw-v1")
    INTERNET = ("allenai/c4", "en")
    CODE = "bigcode/starcoderdata"
    ASSISTANT = "HuggingFaceH4/ultrachat_200k"
    SLIM_PAJAMA = "cerebras/SlimPajama-627B"


class ModelTypes(Enum):
    GPT2 = "gpt2"
    LLAMA = "meta-llama/Llama-3.2-1B"
    GEMMA = "google/gemma-2-2b"
    TINY_LLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class SeqLenFilterDataLoader(DataLoader):
    def __init__(self, dataset, min_seq_len, true_batch_size, *args, **kwargs):
        self.min_seq_len = min_seq_len
        self.true_batch_size = true_batch_size
        super().__init__(dataset, *args, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        filtered_batch = [
            item for item in batch if item["input_ids"].size(1) >= self.min_seq_len
        ]

        if not filtered_batch:
            return None

        input_ids = [item["input_ids"].squeeze(0) for item in filtered_batch]
        return {"input_ids": input_ids}

    def __iter__(self):
        itr = super().__iter__()
        batch = []
        try:
            while True:
                while len(batch) < self.true_batch_size:
                    new_batch = next(itr)
                    if new_batch:
                        new_batch = new_batch["input_ids"]
                        batch.extend(new_batch)
                yield batch[: self.true_batch_size]
                batch = batch[self.true_batch_size :]
        except StopIteration:
            print("out of elements in dataset")


def get_tokenizer_model(
    model_type: ModelTypes,
    get_intermediates: bool = False,
    get_output_attentions: bool = True,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Get the tokenizer and model for a given model type."""
    model_name = model_type.value
    model_name = model_type.value
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=get_output_attentions,
        return_dict_in_generate=True,
    )

    if not get_intermediates:
        return tokenizer, model

    # Dictionary to store Q/K/V projections for each layer
    model.attention_intermediates = {}

    # Create a mapping of modules to their full names
    saved_projs = ["q_proj", "k_proj", "v_proj"]
    module_to_name = {}
    for name, module in model.named_modules():
        if any(proj in name.lower() for proj in saved_projs):
            module_to_name[module] = name

    def attention_forward_hook(module, input, output):
        full_name = module_to_name[module]
        parts = full_name.split(".")
        proj_type = parts[-1]  # e.g., 'q_proj'
        layer_num = int(parts[2])  # e.g., '0' from 'model.layers.0.self_attn.q_proj'

        if proj_type not in model.attention_intermediates:
            model.attention_intermediates[proj_type] = []

        model.attention_intermediates[proj_type].append(output.cpu())

        return output

    # Register hooks for all attention layers
    for name, module in model.named_modules():
        if any(proj in name.lower() for proj in saved_projs):
            module.register_forward_hook(attention_forward_hook)

    return tokenizer, model


def get_dataset(
    dataset_type: DatasetTypes, tokenizer, seq_len: int, batch_size: int, seed: int = 42
) -> DataLoader:
    """Get a dataset iterator with proper tokenization and filtering."""
    if dataset_type == DatasetTypes.WIKI or dataset_type == DatasetTypes.INTERNET:
        dataset = load_dataset(
            dataset_type.value[0], dataset_type.value[1], streaming=True, split="train"
        )
    else:
        dataset = load_dataset(dataset_type.value, streaming=True, split="train")

    tokenize = functools.partial(
        tokenizer,
        padding=False,
        truncation=True,
        max_length=seq_len,
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_tensors="pt",
    )

    dataset = dataset.shuffle(seed=seed)
    column_name = "content" if dataset_type == DatasetTypes.CODE else "text"

    tokenized = dataset.select_columns([column_name]).map(
        tokenize, input_columns=[column_name], remove_columns=[column_name]
    )

    dataloader = SeqLenFilterDataLoader(
        tokenized, seq_len, true_batch_size=batch_size, batch_size=256
    )
    return iter(dataloader)


def get_activation(name: str, all_intermediates: Dict[str, Any]):
    """Create a hook function to capture intermediate activations."""

    def hook(module, input, output):
        all_intermediates[name].append(output.detach())

    return hook


def ensure_figs_folder():
    """Ensure the figures output directory exists."""
    import os

    if not os.path.exists("figs"):
        os.makedirs("figs")
