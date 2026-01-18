import torch
import torch.nn as nn
from transformers import AutoModel


class DPR(nn.Module):
    def __init__(self, loss_fn = None):
        super().__init__()
        self.q_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.p_encoder = AutoModel.from_pretrained("bert-base-uncased")

        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    def encode_query(self, input_ids = None, attention_mask = None, token_type_ids = None):
        outputs = self.q_encoder(input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True)
        return outputs.last_hidden_state[:, 0, :]

    def encode_passage(self, input_ids = None, attention_mask = None, token_type_ids = None):
        outputs =self.p_encoder(input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True)
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, 
                q_input_ids = None, q_attention_mask = None, q_token_type_ids = None,
                p_input_ids = None, p_attention_mask = None, p_token_type_ids = None,
                hn_input_ids = None, hn_attention_mask = None, hn_token_type_ids = None,
                 labels = None, return_loss=True):

        # query, passage encode
        q_emb = self.encode_query(q_input_ids, q_attention_mask, q_token_type_ids)
        p_emb = self.encode_passage(p_input_ids, p_attention_mask, p_token_type_ids)

        # if hard-negative is given encode negative and concat p_emb, hn_emb
        if hn_input_ids is not None:
            hn_emb = self.encode_passage(hn_input_ids, hn_attention_mask, hn_token_type_ids)
            p_emb = torch.cat([p_emb, hn_emb], dim=0)

        # dot product
        sim_matrix = q_emb @ p_emb.T

        if labels is None:
            labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        loss = self.loss_fn(sim_matrix, labels) if return_loss else None

        return {
            "loss": loss,
            "logits": sim_matrix
        }

