from pathlib import Path
from pyexpat import model
import faiss
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
import logging

import transformers
from models import * 
from dprdataset.nqdataset import load_nq_dataset, collate_fn
from transformers import Trainer, TrainerCallback, TrainingArguments

from transformers.trainer_utils import get_last_checkpoint

log = logging.getLogger(__name__)
results = []  # multi run시 결과 한눈에 보기 위해 사용

def with_benchmark(checkout_dir, model_config=None):
    from build_faiss import build_faiss_index
    faiss_index_path = build_faiss_index(checkout_dir, model_config=model_config)
    from faiss_benchmark import benchmark
    results.append(benchmark(checkout_dir=checkout_dir, model_config=model_config))
    
class LayerWLoggingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            layer_w = model.layer_mix.layer_w.data.cpu().numpy()
            log.info(f"Epoch {int(state.epoch)} - layer_w: {layer_w}")


@hydra.main(config_path="configs", version_base=None)
def run(config):
    # tf_logger = logging.getLogger("transformers")
    # tf_logger.setLevel(logging.INFO)
    # tf_logger.propagate = True
    
    device = torch.device(config.device)
    log.info(f"Using device: {device}")

    model = DPR_mixcls(**config.model)
    model.to(device)
    log.info("Model built successfully.")
    
    output_dir = HydraConfig.get().runtime.output_dir

    args = TrainingArguments(
        **config["train"],
        output_dir=output_dir,
        report_to=[],
        log_level="info",
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=load_nq_dataset(config["dataset_path"]),
        data_collator=collate_fn,
        callbacks=[LayerWLoggingCallback()]
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
        
        with_benchmark(Path(last_checkpoint), model_config=config.model)


if __name__ == "__main__":
    checkpoint = run()
    log.info(results)
