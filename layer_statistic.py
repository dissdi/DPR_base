from pathlib import Path
import random
import csv
import torch
from torch.utils.data import Dataset
from safetensors.torch import load_model
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    sum_var_lh = None   # (L-1, H)
    n_steps = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Residual layer statistics", unit="batch"):
            ids = batch["p_input_ids"].to(device)
            mask = batch["p_attention_mask"].to(device)
            tt = batch["p_token_type_ids"].to(device)

            # cls_layers: list of (B, H)
            outputs = model.p_encoder(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=tt,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[1:]      # 12개 레이어 (embedding 제외)
            cls_layers = [h[:, 0, :] for h in hidden_states]  # 각 (B, H)
            cls_stack = torch.stack(cls_layers, dim=1)     # (B, 12, H)
            
            # (B, L, H)
            cls_stack = torch.stack(cls_layers, dim=1)

            # residual: (B, L-1, H) where delta[:, i] = CLS_{i+1} - CLS_{i}
            delta = cls_stack[:, 1:, :] - cls_stack[:, :-1, :]

            # variance over batch dimension -> (L-1, H)
            var_lh = delta.var(dim=0, unbiased=False).detach().cpu()

            if sum_var_lh is None:
                sum_var_lh = torch.zeros_like(var_lh)

            sum_var_lh += var_lh
            n_steps += 1

    mean_var_lh = (sum_var_lh / n_steps)  # (L-1, H)

    results = []
    Lm1, H = mean_var_lh.shape
    for i in range(Lm1):
        for h in range(H):
            results.append({
                "delta_layer": i,   # i=0 means layer1-layer0
                "dim": h,
                "var": float(mean_var_lh[i, h]),
            })

    return results

def save_result(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer_statistics.csv"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result[0].keys()))
        w.writeheader()
        w.writerows(result)

    return str(out_path)


def make_heatmap_line_robust(
    csv_path="analysis_outputs/layer_statistics.csv",
    out_png="analysis_outputs/var_heatmap_layersxH_robust.png",
    n_layers=11,   # residual이면 11개가 맞음
    H=768,
    layer_col="delta_layer",
    var_col="var",
    low_q=1.0,
    high_q=99.0,
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    # robust vmin/vmax
    vals = df[var_col].to_numpy(dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    vmin = np.percentile(vals, low_q)
    vmax = np.percentile(vals, high_q)

    # (n_layers, H) 매트릭스로 모으기
    mat = np.full((n_layers, H), np.nan, dtype=np.float32)
    for l in range(n_layers):
        sub = df[df[layer_col] == l]
        if sub.empty:
            continue
        dims = sub["dim"].to_numpy(dtype=int)
        mat[l, dims] = sub[var_col].to_numpy(dtype=np.float32)

    # nan은 vmin으로 채우거나 그대로 두고 clip
    mat = np.nan_to_num(mat, nan=vmin)
    mat = np.clip(mat, vmin, vmax)

    # 그림: 세로 11줄, 가로 768
    fig, ax = plt.subplots(figsize=(18, 6))  # 가로 길게
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)

    ax.set_title("Per-dim variance (residual) — layers × hidden-dim (robust)")
    ax.set_xlabel("hidden dim (0..H-1)")
    ax.set_ylabel("delta layer (CLS_{t} - CLS_{t-1})")

    # y축 눈금: 0..10
    ax.set_yticks(np.arange(n_layers))

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(f"Var (clipped {low_q}–{high_q} pct)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print("saved:", out_png, "| vmin:", vmin, "vmax:", vmax)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    
    print("sampling w100")
    # sample_path = make_sample_w100(r"downloads\data\wikipedia_split\psgs_w100.tsv", 10000)
    sample_path = r"downloads\data\wikipedia_split\psgs_w100.sample10000.tsv"
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
    
    print("make heatmap")
    make_heatmap_line_robust()