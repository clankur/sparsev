import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum
from typing import Tuple, Dict, Any
import functools
from torch.utils.data import DataLoader


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
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Get the tokenizer and model for a given model type."""
    model_name = model_type.value
    return AutoTokenizer.from_pretrained(
        model_name
    ), AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)


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
    if dataset_type == DatasetTypes.CODE:
        tokenized = dataset.select_columns(["content"]).map(
            tokenize, input_columns=["content"], remove_columns=["content"]
        )
    else:
        tokenized = dataset.select_columns(["text"]).map(
            tokenize, input_columns=["text"], remove_columns=["text"]
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
