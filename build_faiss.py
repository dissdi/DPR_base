from pathlib import Path
import numpy as np
import torch
import faiss
from tqdm import tqdm
from dprdataset.nqdataset import read_tsv
from models import DPR, BaseTokenizer
from safetensors.torch import load_model
from logging import getLogger
log = getLogger(__name__)

def build_faiss_index(check_point_dir: Path, BATCH_SIZE=512, STEP=800, PSGS_PATH="downloads/data/wikipedia_split/psgs_w100.tsv", nlist=4096):
    log.info(f'Build faiss index at {check_point_dir}')

    output_path = check_point_dir/"faiss"
    output_path.mkdir(parents=True, exist_ok=True)
    
    MODEL_PATH = check_point_dir / "model.safetensors"
    FAISS_INDEX_PATH = output_path / "faiss.index" # To save path


    model = DPR()
    load_model(model, MODEL_PATH)
    model.to("cuda")

    model.eval()

    log.info("Start encode passages for train")

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

    log.info("Start IVF index train")

    p_embs = torch.cat(p_embs, dim=0)
    x_train = p_embs.numpy().astype(np.float32)
    index.train(x_train)

    log.info("IVF index train done.")

    log.info("Start build IVF index")

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
    log.info("Save faiss index to disk")

    return FAISS_INDEX_PATH