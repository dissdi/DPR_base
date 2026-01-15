from .DPRModel import DPR
from transformers import AutoTokenizer

BaseTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")