from datasets import load_dataset, Features, Value, Sequence
from transformers import AutoTokenizer, DataCollatorWithPadding
import random

from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def load_w100_dataset():
    return load_dataset(
        'csv',
        data_files='downloads/data/wikipedia_split/psgs_w100.tsv',
        sep='\t',
        streaming=True
    )

# dataset 구조
# [
#   {
# 	"question": "....",
# 	"answers": ["...", "...", "..."],
# 	"positive_ctxs": [{
# 		"title": "...",
# 		"text": "...."
# 	}],
# 	"negative_ctxs": ["..."],
# 	"hard_negative_ctxs": ["..."]
#   },
#   ...
# ]

def load_nq_dataset(path):
    ctx = Features({
    "passage_id": Value("string"),
    "score": Value("float64"),       # score는 전부 float로 통일 권장 (int도 float로 캐스팅 가능)
    "text": Value("large_string"),
    "title": Value("string"),
    "title_score": Value("int64"),
    })

    features = Features({
        "dataset": Value("string"),
        "question": Value("string"),
        "answers": Sequence(Value("string")),
        "positive_ctxs": Sequence(ctx),
        "negative_ctxs": Sequence(ctx),
        "hard_negative_ctxs": Sequence(ctx),
    })

    return load_dataset(
        'json', 
        data_files=path, 
        split='train',
        features=features
    )

def collate_fn(batch):
    questions = [item["question"] for item in batch]

    positives = []
    for item in batch:
        selected_p = random.choice(item["positive_ctxs"])
        formatted_p = f"{selected_p['title']}{tokenizer.sep_token}{selected_p['text']}"
        positives.append(formatted_p)

    hard_negatives = []
    for item in batch:
        pool = item["hard_negative_ctxs"] if item["hard_negative_ctxs"] else item["negative_ctxs"]
        selected_hn = random.choice(pool)
        
        formatted_hn = f"{selected_hn['title']}{tokenizer.sep_token}{selected_hn['text']}"
        hard_negatives.append(formatted_hn)

    q_inputs = tokenizer(
        questions,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    p_inputs = tokenizer(
        positives,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    hn_inputs = tokenizer(
        hard_negatives,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    return {
        "q_input_ids": q_inputs['input_ids'],
        "q_attention_mask": q_inputs['attention_mask'],
        
        "p_input_ids": p_inputs['input_ids'],
        "p_attention_mask": p_inputs['attention_mask'], 
        
        "hn_input_ids": hn_inputs['input_ids'],        
        "hn_attention_mask": hn_inputs['attention_mask']
    }
