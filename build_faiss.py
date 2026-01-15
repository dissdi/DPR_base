import numpy as np
import torch
import faiss
from tqdm import tqdm
from dprdataset.nqdataset import read_tsv
from models import DPR, BaseTokenizer
from safetensors.torch import load_model
from datetime import datetime

def log(msg: str) -> None:
    ts = datetime.now().strftime("%m/%d %H:%M:%S")
    print(f"[{ts}] : {msg}", flush=True)

if __name__ == "__main__":
    BATCH_SIZE = 256
    STEP = 150
    ivf_data_path = 'faiss/ivfdata'
    loader = read_tsv(r"downloads\data\wikipedia_split\psgs_w100.tsv", BATCH_SIZE)

    model = DPR()
    load_model(model, r"output\b32_small_hn\checkpoint-4680\model.safetensors")
    model.to("cuda")

    model.eval()
    model = torch.compile(model)

    log("Start encoding passages for train")

    with torch.no_grad():
        p_embs = []
        for steps, batch in enumerate(tqdm(loader, desc="Encoding passages for train", total=STEP, unit="batch"), start=1):
            p_token = BaseTokenizer([item["text"] for item in batch], max_length=256, padding="max_length", truncation=True,
                                    return_tensors="pt").to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb = model.encode_passage(**p_token)
            p_embs.append(emb.float().cpu())
            if steps == STEP:
                break
        loader.close()
    p_embs = torch.cat(p_embs, dim=0)

    quant = faiss.IndexFlatIP(768)
    nlist = 512
    m = 64
    bits = 8
    index = faiss.IndexIVFFlat(quant, 768, nlist, faiss.METRIC_INNER_PRODUCT)
    
    log("Train faiss IVFPQ")

    p_embs_np = p_embs.numpy().astype(np.float32, copy=False)
    index.train(p_embs_np)

    log("Train done.")
    
    #ram이 아닌 disk에서 faiss 실행
    invlists = faiss.OnDiskInvertedLists(index.nlist, index.code_size, ivf_data_path)

    log("Build faiss IVFPQ")

    log("Start encoding passages")

    loader = read_tsv(r"downloads\data\wikipedia_split\psgs_w100.tsv", BATCH_SIZE)
    with torch.no_grad():
        for steps, batch in enumerate(tqdm(loader, desc="Encoding passages", unit="batch"),
                                      start=1):
            p_token = BaseTokenizer([item["text"] for item in batch], max_length=256, padding="max_length",
                                    truncation=True,
                                    return_tensors="pt").to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb = model.encode_passage(**p_token)
            X = emb.float().cpu()
            I = np.asarray([item["id"] for item in batch], dtype=np.int64)
            index.add_with_ids(X, I)
        loader.close()

    faiss.write_index(index, "./data/b32_small_hn.index")
    log("Save faiss index to disk")