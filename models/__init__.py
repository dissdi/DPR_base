from .DPRModel import DPR
from .FastTrackModel import FastDPR, TinyModel
from transformers import AutoTokenizer

BaseTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")