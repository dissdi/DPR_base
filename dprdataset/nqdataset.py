import torch
import random
from datasets import load_from_disk
from torch.utils.data import Dataset


def _batch_to_tensor(xs: list, dtype=torch.long):
    if torch.is_tensor(xs[0]):
        return torch.stack(xs)

    return torch.tensor(xs, dtype=dtype)

def collate_fn(batch):
    q_input_ids= _batch_to_tensor([item["query"]["input_ids"] for item in batch])
    q_attention_mask = _batch_to_tensor([item["query"]["attention_mask"] for item in batch])

    chosen = [random.choice(item["positive_passages"]) for item in batch]

    p_input_ids = _batch_to_tensor([item["token"]["input_ids"] for item in chosen])
    p_attention_mask = _batch_to_tensor([item["token"]["attention_mask"] for item in chosen])

    chosen = [random.choice(item["negative_passages"]) for item in batch]

    hn_input_ids = _batch_to_tensor([item["token"]["input_ids"] for item in chosen])
    hn_attention_mask = _batch_to_tensor([item["token"]["attention_mask"] for item in chosen])

    return {
        "q_input_ids": q_input_ids,
        "q_attention_mask": q_attention_mask,
        "p_input_ids": p_input_ids,
        "p_attention_mask": p_attention_mask,
        "hn_input_ids": hn_input_ids,
        "hn_attention_mask": hn_attention_mask,
    }

def load_nq_dataset(file_path: str) -> Dataset:
    return load_from_disk(file_path)