from .DPRModel import DPR
from transformers import AutoTokenizer
from .ReRankerModel import ReRankerDPR

BaseTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")