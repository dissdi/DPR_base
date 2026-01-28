from .DPRModel import DPR
from .DPRModel_alcls import DPR_alcls
from .DPRModel_mixcls import DPR_mixcls
from .DPRModel_CosSim import DPR_CosSim
from transformers import AutoTokenizer

BaseTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")