import torch
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from models import DPR
from dprdataset.nqdataset import load_nq_dataset, collate_fn
from transformers import Trainer, TrainingArguments

log = logging.getLogger(__name__)
results = []  # multi run시 결과 한눈에 보기 위해 사용

#재현성을 위한 seed 고정
def set_seed(seed):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 멀티 GPU 사용 시
    np.random.seed(seed)
    random.seed(seed)
    # 결정론적 연산을 위한 설정 (필요 시)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


@hydra.main(config_path="configs", version_base=None)
def run(config):
    device = torch.device(config.device)
    log.info(f"Using device: {device}")
    set_seed(config.seed)

    model = DPR()
    model.to(device)
    log.info("Model built successfully.")

    args = TrainingArguments(
        **config["train"],
        report_to=[]
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=load_nq_dataset(r"downloads\\data\\retriever\\nq-train.json"),
        # eval_dataset=load_nq_dataset(r"downloads\\data\\retriever\\nq-dev.json"),
        data_collator=collate_fn,
    )

    trainer.train()

if __name__ == "__main__":
    run()
    log.info(results)
