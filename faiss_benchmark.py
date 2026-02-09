from pathlib import Path
import numpy as np
import torch
import tqdm
import faiss
from safetensors.torch import load_model

from dprdataset.nqdataset import load_nq_dataset, valid_collate_fn
from torch.utils.data import DataLoader

from models import *
import logging
log = logging.getLogger(__name__)


def benchmark_recall_k(model, index, dataset_path, k=1, batch_size=512):
    valid_dataset = load_nq_dataset(dataset_path)
    dataloader = DataLoader(valid_dataset, collate_fn=valid_collate_fn, batch_size=batch_size, shuffle=False, num_workers=6,
                            pin_memory=True,
                            prefetch_factor=6, persistent_workers=True)
    model.eval()
    with torch.no_grad():
        recall = 0
        N = 0
        for step, batch in enumerate(tqdm.tqdm(dataloader, desc=f"Benchmark Recall@{k}", unit="batch")):
            # print(batch.keys())
            q_ids = batch["q_input_ids"].to("cuda")
            q_mask = batch["q_attention_mask"].to("cuda")
            q_token_ids = batch["q_token_type_ids"].to("cuda")

            query = model.encode_query(
                q_ids, q_mask, q_token_ids).cpu().numpy().astype(np.float32)

            dist, indices = index.search(query, k)

            labels = batch["passage_ids"]

            for result, label in zip(indices, labels):
                N += 1
                if set(result) & set(label):
                    recall += 1

        return recall / N


def benchmark(checkout_dir: Path = None, DATASET_PATH: str = "downloads/data/nq-dev", BATCH_SIZE: int = 256, NPROBE: int = 64, model_config=None):
    log.info(f"Start benchmark faiss index at {checkout_dir}")
    FAISS_INDEX_PATH = checkout_dir / 'faiss' / "faiss.index"
    MODEL_PATH = checkout_dir / "model.safetensors"
    Ks = [1, 5, 20, 100]

    log.info("Benchmark Recall@K")
    log.info(f"FAISS_INDEX_PATH: {FAISS_INDEX_PATH}")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    index.nprobe = NPROBE

    model = DPR_mixcls(**model_config) if model_config is not None else DPR_mixcls()
    load_model(model, MODEL_PATH)
    model.to("cuda")

    results = {}

    for k in Ks:
        results[k] = benchmark_recall_k(
            model, index, DATASET_PATH, k=k, batch_size=BATCH_SIZE)

    for k, recall in results.items():
        log.info(f"Recall@{k}: {recall:.3f}")
    print(f"Benchmark results at {checkout_dir}: {results}")
    return results

if __name__ == "__main__":
    import sys
    # 로그가 콘솔에 찍히도록 기본 설정
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    log.info("Starting faiss benchmark script...")
    print("Running faiss benchmark...", flush=True)
    result = benchmark(Path("projects/dpr_mixcls/2026-02-04/11-37-27/checkpoint-13800"), model_config={"mix_layer":8}, BATCH_SIZE=256)
    print(result, flush=True)