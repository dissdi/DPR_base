import numpy as np
import torch
import tqdm
import faiss
from safetensors.torch import load_model

from dprdataset.nqdataset import load_nq_dataset, valid_collate_fn
from torch.utils.data import DataLoader

from models import DPR


def benchmark_recall_k(model, index, dataset_path, k = 1, batch_size = 256):
    valid_dataset = load_nq_dataset(dataset_path)
    dataloader = DataLoader(valid_dataset, collate_fn=valid_collate_fn, batch_size=batch_size, shuffle=False, num_workers=6,
                            pin_memory=True,
                            prefetch_factor=6, persistent_workers=True)
    model.eval()
    with torch.no_grad():
        recall = 0
        N = 0
        for step, batch in enumerate(tqdm.tqdm(dataloader, desc=f"Benchmark Recall@{k}", unit="batch")):
            #print(batch.keys())
            q_ids = batch["q_input_ids"].to("cuda")
            q_mask = batch["q_attention_mask"].to("cuda")
            q_token_ids = batch["q_token_type_ids"].to("cuda")

            query = model.encode_query(q_ids, q_mask, q_token_ids).cpu().numpy().astype(np.float32)

            dist, indices = index.search(query, k)

            labels = batch["passage_ids"]

            for result, label in zip(indices, labels):
                N += 1
                if set(result) & set(label):
                    recall += 1

        return recall / N
from models.DPRModel_alcls import DPR_alcls 


if __name__ == "__main__":
    FAISS_INDEX_PATH = 'faiss/faiss.index'
    MODEL_PATH = "outputs/2026-01-20/20-42-34/checkpoint-13800/model.safetensors"
    DATASET_PATH = "downloads/data/nq-dev"
    BATCH_SIZE = 256
    Ks = [1, 5, 20, 100]
    NPROBE = 64 # IVF parameter

    print("Benchmark Recall@K")

    index = faiss.read_index(FAISS_INDEX_PATH)
    index.nprobe = NPROBE

    model = DPR_alcls()
    load_model(model, MODEL_PATH)
    model.to("cuda")

    results = []

    for k in Ks:
        results.append(benchmark_recall_k(model, index, DATASET_PATH, k = k, batch_size = BATCH_SIZE))

    for k, recall in zip(Ks, results):
        print(f"Recall@{k}: {recall:.3f}", flush=True)