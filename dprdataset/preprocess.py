import random

import ijson
from datasets import Dataset

from models import BaseTokenizer
from models.RiskMCLS_tokenizer import RiskMCLS_tokenizer

tokenizer = RiskMCLS_tokenizer()

def nq_generator(file_path: str):
    with open(file_path, 'rb') as f:
        for item in ijson.items(f, 'item'):
            pos_ctxs = []
            hard_neg_ctxs = []
            for ctx in item["positive_ctxs"]:
                pos_ctxs.append({
                    "title": ctx["title"],
                    "text": ctx["text"],
                    "passage_id": ctx["passage_id"],
                })

            for ctx in item["hard_negative_ctxs"]:
                hard_neg_ctxs.append({
                    "title": ctx["title"],
                    "text": ctx["text"],
                    "passage_id": ctx["passage_id"],
                    "score": float(ctx["score"]),
                })

            if len(hard_neg_ctxs) == 0:
                ctx = random.choice(item["negative_ctxs"])
                hard_neg_ctxs.append({
                    "title": ctx["title"],
                    "text": ctx["text"],
                    "passage_id": ctx["passage_id"],
                    "score": float(ctx["score"]),
                })

            # Sort negatives in descending order (hard negatives with high scores are more challenging)
            hard_neg_ctxs.sort(key=lambda x: x["score"], reverse=True)

            yield {
                "question": item["question"],
                "positive_ctxs": pos_ctxs,
                "hard_negative_ctxs": hard_neg_ctxs,
            }


def nq_preprocess(batch):
    questions = batch["question"]
    p_ctxs_batch = batch["positive_ctxs"]
    hn_ctxs_batch = batch["hard_negative_ctxs"]

    queries = []
    pos_passages_batch = []
    neg_passages_batch = []

    for q, pos_ctxs, neg_ctxs in zip(questions, p_ctxs_batch, hn_ctxs_batch):
        q_token = BaseTokenizer(q, max_length=64, padding="max_length", truncation=True)

        pos_passages = []
        neg_passages = []

        # for sanity check
        # pos_ctxs = pos_ctxs[:1]
        # neg_ctxs = neg_ctxs[:min(len(neg_ctxs), 4)]

        for ctx in pos_ctxs:
            p_token = BaseTokenizer(ctx["title"], ctx["text"], max_length=256, padding="max_length", truncation=True)
            pos_passages.append({
                "passage_id": ctx["passage_id"],
                "token": p_token
            })

        for ctx in neg_ctxs:
            p_token = tokenizer.p_tokenize(ctx["title"], ctx["text"], max_length=256, padding="max_length", truncation=True) # changed
            neg_passages.append({
                "passage_id": ctx["passage_id"],
                "token": p_token
            })

        queries.append(q_token)
        pos_passages_batch.append(pos_passages)
        neg_passages_batch.append(neg_passages)

    return {
        "query": queries,
        "positive_passages": pos_passages_batch,
        "negative_passages": neg_passages_batch,
    }


if __name__ == "__main__":
    #python -m datasets.preprocess
    tragets = ['nq-dev', 'nq-train']
    for target in tragets:
        print(f"Preprocessing {target} dataset...")
        ds = Dataset.from_generator(nq_generator, gen_kwargs={
                                    "file_path": f'downloads/data/retriever/{target}.json'})

        ds = ds.map(nq_preprocess,
                    batched=True,
                    batch_size=512,
                    num_proc=6,
                    remove_columns=ds.column_names)

        ds.save_to_disk(f'downloads/data/risk/{target}')

    print("Done.")  
