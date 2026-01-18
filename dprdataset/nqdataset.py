import torch
import random
from datasets import load_from_disk
from torch.utils.data import Dataset
import csv

def read_tsv(path, batch_size):
    batch = []
    with open(path,"r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            batch.append({
                "id": int(row["id"]),
                "title" : row["title"],
                "text": row["text"]
            })
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _batch_to_tensor(xs: list, dtype=torch.long):
    if torch.is_tensor(xs[0]):
        return torch.stack(xs)

    return torch.tensor(xs, dtype=dtype)

def collate_fn(batch):
    q_input_ids= _batch_to_tensor([item["query"]["input_ids"] for item in batch])
    q_token_type_ids= _batch_to_tensor([item["query"]["token_type_ids"] for item in batch])
    q_attention_mask = _batch_to_tensor([item["query"]["attention_mask"] for item in batch])

    chosen = [random.choice(item["positive_passages"]) for item in batch]

    p_input_ids = _batch_to_tensor([item["token"]["input_ids"] for item in chosen])
    p_token_type_ids= _batch_to_tensor([item["token"]["token_type_ids"] for item in chosen])
    p_attention_mask = _batch_to_tensor([item["token"]["attention_mask"] for item in chosen])

    chosen = [random.choice(item["negative_passages"]) for item in batch]

    hn_input_ids = _batch_to_tensor([item["token"]["input_ids"] for item in chosen])
    hn_token_type_ids = _batch_to_tensor([item["token"]["token_type_ids"] for item in chosen])
    hn_attention_mask = _batch_to_tensor([item["token"]["attention_mask"] for item in chosen])

    return {
        "q_input_ids": q_input_ids,
        "q_attention_mask": q_attention_mask,
        "q_token_type_ids": q_token_type_ids,

        "p_input_ids": p_input_ids,
        "p_attention_mask": p_attention_mask,
        "p_token_type_ids": p_token_type_ids,

        "hn_input_ids": hn_input_ids,
        "hn_attention_mask": hn_attention_mask,
        "hn_token_type_ids": hn_token_type_ids,
    }

def valid_collate_fn(batch):
    q_input_ids = _batch_to_tensor([item["query"]["input_ids"] for item in batch])
    q_token_type_ids = _batch_to_tensor([item["query"]["token_type_ids"] for item in batch])
    q_attention_mask = _batch_to_tensor([item["query"]["attention_mask"] for item in batch])

    passage_ids = [[int(passage["passage_id"]) for passage in item["positive_passages"]] for item in batch]
    padded_passage_ids = [sublist + [-1] * (128 - len(sublist)) for sublist in passage_ids]

    return {
        "q_input_ids": q_input_ids,
        "q_attention_mask": q_attention_mask,
        "q_token_type_ids": q_token_type_ids,
        "passage_ids": padded_passage_ids,
    }

def load_nq_dataset(file_path: str) -> Dataset:
    return load_from_disk(file_path)