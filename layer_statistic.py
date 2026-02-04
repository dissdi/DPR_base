from pathlib import Path
import random
import csv
import torch
from torch.utils.data import Dataset
from safetensors.torch import load_model
import tqdm

from models import DPR_mixcls

class PassageCollator:
    def __init__(self, tokenizer, max_length=256, with_title=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_title = with_title

    def __call__(self, batch):
        if self.with_title:
            texts = [b["title"] + " " + b["text"] for b in batch]
        else:
            texts = [b["text"] for b in batch]

        toks = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "p_input_ids": toks["input_ids"],
            "p_attention_mask": toks["attention_mask"],
            "p_token_type_ids": toks.get("token_type_ids", torch.zeros_like(toks["input_ids"])),
        }

def make_sample_w100(path, n, seed=42):
    path = Path(path)
    out_path = path.with_name(f"{path.stem}.sample{n}{path.suffix}")

    rng = random.Random(seed)

    with path.open("r", encoding="utf-8", newline="") as fin, out_path.open("w", encoding="utf-8", newline="") as fout:
        header = fin.readline()
        fout.write(header)

        reservoir = []
        seen = 0
        for line in fin:
            seen += 1
            if len(reservoir) < n:
                reservoir.append(line)
            else:
                j = rng.randrange(seen)
                if j < n:
                    reservoir[j] = line

        fout.writelines(reservoir)

    return str(out_path)


class W100TSVDataset(Dataset):
    def __init__(self, tsv_path):
        self.tsv_path = Path(tsv_path)

        with self.tsv_path.open("r", encoding="utf-8", newline="") as f:
            header = f.readline().rstrip("\n").split("\t")
        self.col = {name: i for i, name in enumerate(header)}

        self.offsets = []
        with self.tsv_path.open("r", encoding="utf-8", newline="") as f:
            f.readline()
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                self.offsets.append(pos)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        off = self.offsets[idx]
        with self.tsv_path.open("r", encoding="utf-8", newline="") as f:
            f.seek(off)
            line = f.readline().rstrip("\n")
        parts = line.split("\t")

        return {
            "pid": parts[self.col["id"]],
            "title": parts[self.col["title"]] if "title" in self.col else "",
            "text": parts[self.col["text"]],
        }


def get_layer_statistics(checkout_dir, dataloader, device="cuda"):
    checkout_dir = Path(checkout_dir)
    model_path = checkout_dir / "model.safetensors"

    model = DPR_mixcls()
    load_model(model, model_path)
    model.to(device)
    model.eval()

    sum_var = None
    best_counts = None
    n_total = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Layer statistics", unit="batch"):
            ids = batch["p_input_ids"].to(device)
            mask = batch["p_attention_mask"].to(device)
            tt = batch["p_token_type_ids"].to(device)

            _, cls_layers = model.encode_passage(ids, mask, tt, statistic_mode=True)  # cls_layers: list of (B,H)
            cls_stack = torch.stack(cls_layers, dim=1)  # (B,L,H)

            var_bl = cls_stack.var(dim=-1, unbiased=False)  # (B,L)
            best = var_bl.argmax(dim=-1)                    # (B,)

            B, L = var_bl.shape
            if sum_var is None:
                sum_var = torch.zeros(L, dtype=torch.float64)
                best_counts = torch.zeros(L, dtype=torch.long)

            sum_var += var_bl.double().sum(dim=0).cpu()
            best_counts += torch.bincount(best.cpu(), minlength=L)
            n_total += B

    mean_var = (sum_var / n_total).tolist()
    best_ratio = (best_counts.double() / n_total).tolist()

    return [
        {"layer": i, "mean_cls_var": float(mean_var[i]), "best_ratio": float(best_ratio[i]), "n": int(n_total)}
        for i in range(len(mean_var))
    ]


def save_result(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer_statistics.csv"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result[0].keys()))
        w.writeheader()
        w.writerows(result)

    return str(out_path)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    
    print("sampling w100")
    sample_path = make_sample_w100(r"downloads\data\wikipedia_split\psgs_w100.tsv", 10000)
    # sample_path = r"downloads\data\wikipedia_split\psgs_w100.sample10000.tsv"
    print("sampling complete")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = W100TSVDataset(sample_path)
    collate_fn = PassageCollator(tokenizer, max_length=256, with_title=False)
    
    dl = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    print("get layer statistic")
    res = get_layer_statistics(r"output", dl)
    print("getting layer statistic complete")
    
    print("saving result")
    save_result(res, "analysis_outputs")
    print("complete")