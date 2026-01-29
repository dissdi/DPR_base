from pathlib import Path
import numpy as np
import torch
import faiss
from tqdm import tqdm
from dprdataset.nqdataset import read_tsv
from models.RiskMCLS_model import DPR
from models.RiskMCLS_tokenizer import RiskMCLS_tokenizer
from safetensors.torch import load_model

def build_faiss_index(check_point_dir: Path, BATCH_SIZE=512, STEP=800, PSGS_PATH="downloads/data/wikipedia_split/psgs_w100.tsv", nlist=4096):

    output_path = check_point_dir/"faiss"
    output_path.mkdir(parents=True, exist_ok=True)
    
    MODEL_PATH = check_point_dir / "model.safetensors"
    FAISS_INDEX_PATH = output_path / "faiss.index" # To save path


    model = DPR()
    load_model(model, MODEL_PATH)
    model.to("cuda")

    model.eval()

    tokenizer = RiskMCLS_tokenizer()

    print(f"Start encode passages for train")

    qaunt = faiss.IndexFlatIP(768) # Bert CLS Token dim \wo L2 norm (IP = Inner Product)
    index = faiss.IndexIVFFlat(qaunt, 768, nlist, faiss.METRIC_INNER_PRODUCT) # IVF needs train

    loader = read_tsv(PSGS_PATH, BATCH_SIZE)
    with torch.no_grad():
        p_embs = []
        for steps, batch in enumerate(tqdm(loader, desc="Encoding passages for train", total=STEP, unit="batch"), start=0):
            embs = []
            # tokenize one by one, as my tokenizer doesnt support batch tokenize
            for item in batch:
                p_token = tokenizer.p_tokenize(item["title"], item["text"])
                
                # tensor all item
                p_token = {
                    "input_ids": torch.tensor(p_token["input_ids"], dtype=torch.long).unsqueeze(0).to("cuda"),
                    "attention_mask": torch.tensor(p_token["attention_mask"], dtype=torch.long).unsqueeze(0).to("cuda"),
                    "token_type_ids": torch.tensor(p_token["token_type_ids"], dtype=torch.long).unsqueeze(0).to("cuda"),
                    "mcls_positions": p_token["mcls_positions"].to("cuda"),
                    "mcls_mask": p_token["mcls_mask"].to("cuda"),
                }
                mcls, mcls_mask = model.encode_passage(**p_token)
                
                mcls_mask_ = mcls_mask
                if mcls_mask_.dim() == 1:
                    mcls_mask_ = mcls_mask_.unsqueeze(0)     # (M,) -> (1, M)
                mcls_mask_ = mcls_mask_.unsqueeze(-1).to(mcls.device)  # (1, M, 1)
                
                denom = mcls_mask_.sum(dim=1).clamp(min=1)          # (1, 1)
                pooled = (mcls * mcls_mask_).sum(dim=1) / denom     # (1, H)   
                             
                embs.append(pooled)
            emb = torch.cat(embs, dim=0)
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
        for steps, batch in enumerate(tqdm(loader, desc="Encoding passages", unit="batch"), start=0):
            embs = []
            for item in batch:
                p_token = tokenizer.p_tokenize(item["title"], item["text"])
                # tensor all item
                p_token = {
                    "input_ids": torch.tensor(p_token["input_ids"], dtype=torch.long).unsqueeze(0).to("cuda"),
                    "attention_mask": torch.tensor(p_token["attention_mask"], dtype=torch.long).unsqueeze(0).to("cuda"),
                    "token_type_ids": torch.tensor(p_token["token_type_ids"], dtype=torch.long).unsqueeze(0).to("cuda"),
                    "mcls_positions": p_token["mcls_positions"].to("cuda"),
                    "mcls_mask": p_token["mcls_mask"].to("cuda"),
                }
                mcls, mcls_mask = model.encode_passage(**p_token)

                mcls_mask_ = mcls_mask
                if mcls_mask_.dim() == 1:
                    mcls_mask_ = mcls_mask_.unsqueeze(0)     # (M,) -> (1, M)
                mcls_mask_ = mcls_mask_.unsqueeze(-1).to(mcls.device)  # (1, M, 1)
             
                denom = mcls_mask_.sum(dim=1).clamp(min=1)          # (1, 1)
                pooled = (mcls * mcls_mask_).sum(dim=1) / denom     # (1, H)

                embs.append(pooled)
            emb = torch.cat(embs, dim=0)
            X = emb.cpu().numpy().astype(np.float32)
            I = np.asarray([item["id"] for item in batch], dtype=np.int64)
            index.add_with_ids(X, I)
        loader.close()

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print("Save faiss index to disk")

    return FAISS_INDEX_PATH

if __name__ == "__main__":
    build_faiss_index(Path("./projects/risk_mcls/2026-01-25/23-06-31/checkpoint-13800"), STEP=800)