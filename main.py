from pathlib import Path
import faiss
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
import logging

import transformers

from models import DPR
from dprdataset.nqdataset import load_nq_dataset, collate_fn
from transformers import Trainer, TrainingArguments

from transformers.trainer_utils import get_last_checkpoint

log = logging.getLogger(__name__)
results = []  # multi run시 결과 한눈에 보기 위해 사용

def with_benchmark(checkout_dir):
    from build_faiss import build_faiss_index
    faiss_index_path = build_faiss_index(checkout_dir)
    from faiss_benchmark import benchmark
    results.append(benchmark(checkout_dir=checkout_dir))

@hydra.main(config_path="configs", version_base=None)
def run(config):
    tf_logger = logging.getLogger("transformers")
    tf_logger.setLevel(logging.INFO)
    tf_logger.propagate = True
    
    device = torch.device(config.device)
    log.info(f"Using device: {device}")

    model = DPR()
    model.to(device)
    log.info("Model built successfully.")
    
    output_dir = HydraConfig.get().runtime.output_dir

    args = TrainingArguments(
        **config["train"],
        output_dir=output_dir,
        report_to=[],
        log_level="info",
        seed=config.seed # instead of set_seed func
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=load_nq_dataset(config["dataset_path"]),
        data_collator=collate_fn,
    )

    last_checkpoint = None
    trainer.train()
    last_checkpoint = get_last_checkpoint(output_dir)
    log.info(f"Training completed. Last checkpoint: {last_checkpoint}")

    if config.benchmark:
        del model
        del trainer
        del args
        torch.cuda.empty_cache()
        
        with_benchmark(Path(last_checkpoint))


if __name__ == "__main__":
    checkpoint = run()
    log.info(results)
