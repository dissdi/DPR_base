from pathlib import Path
import numpy as np
import torch
import faiss
from tqdm import tqdm
from dprdataset.nqdataset import read_tsv
from models import BaseTokenizer
from safetensors.torch import load_model
from models.DPRModel_alcls import DPR_alcls 

if __name__ == "__main__":
    output_path = Path("faiss")
    BATCH_SIZE = 512 # Passage encode batch size
    STEP = 800 # Total training sample count is calculated by STEP * BATCH_SIZE.
    MODEL_PATH = 'outputs/2026-01-20/20-42-34/checkpoint-13800/model.safetensors'
    FAISS_INDEX_PATH = output_path / "faiss.index" # To save path
    PSGS_PATH = "downloads/data/wikipedia_split/psgs_w100.tsv" # Passages path (should be tsv file)
    nlist = 4096 # IVF parameter


    model = DPR_alcls()
    load_model(model, MODEL_PATH)
    model.to("cuda")

    model.eval()

    print(f"Start encode passages for train")

    qaunt = faiss.IndexFlatIP(768) # Bert CLS Token dim \wo L2 norm (IP = Inner Product)
    index = faiss.IndexIVFFlat(qaunt, 768, nlist, faiss.METRIC_INNER_PRODUCT) # IVF needs train

    loader = read_tsv(PSGS_PATH, BATCH_SIZE)
    with torch.no_grad():
        p_embs = []
        for steps, batch in enumerate(tqdm(loader, desc="Encoding passages for train", total=STEP, unit="batch"), start=0):
            titles = [item["title"] for item in batch]
            texts = [item["text"] for item in batch]
            p_token = BaseTokenizer(titles, texts, max_length=256, padding="max_length", truncation=True,
                                    return_tensors="pt").to("cuda")
            emb = model.encode_passage(**p_token)
            p_embs.append(emb.cpu())
            if steps + 1 >= STEP:
                break
        loader.close()

    print("Start IVF index train")

    p_embs = torch.cat(p_embs, dim=0)
    x_train = p_embs.numpy().astype(np.float32)
    index.train(x_train)

    print("IVF index train done.")

    print("Start build IVF index")

    # \w ChatGPT
    inv = faiss.OnDiskInvertedLists(nlist, index.code_size, str(output_path / "ivf_lists.ondisk"))
    index.replace_invlists(inv)

    loader = read_tsv(PSGS_PATH, BATCH_SIZE)
    with torch.no_grad():
        for steps, batch in enumerate(tqdm(loader, desc="Encoding passages", unit="batch"),
                                      start=0):
            titles = [item["title"] for item in batch]
            texts = [item["text"] for item in batch]
            p_token = BaseTokenizer(titles, texts, max_length=256, padding="max_length", truncation=True,
                                    return_tensors="pt").to("cuda")

            emb = model.encode_passage(**p_token)
            X = emb.cpu().numpy().astype(np.float32)
            I = np.asarray([item["id"] for item in batch], dtype=np.int64)
            index.add_with_ids(X, I)
        loader.close()

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print("Save faiss index to disk")