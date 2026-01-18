import torch
import torch.nn as nn
from transformers import AutoModel


class DPR_alcls(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.q_encoder = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.p_encoder = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

        self.layer_attn = LayerAttention(hidden_size=768,
                                         num_heads=8,
                                         ffn_mult=4,
                                         dropout=0.1)

        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    def encode_query(self, input_ids=None, attention_mask=None, token_type_ids=None):
        layers = self.q_encoder(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict=True).hidden_states[1:13]  # bert-base is 12 layer

        cls_layers = [h[:, 0, :] for h in layers]
        return self.layer_attn(cls_layers)


    def encode_passage(self, input_ids=None, attention_mask=None, token_type_ids=None):
        layers = self.p_encoder(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict=True).hidden_states[1:13]  # bert-base is 12 layer

        cls_layers = [h[:, 0, :] for h in layers]
        return self.layer_attn(cls_layers)

    def forward(self,
                q_input_ids=None, q_attention_mask=None, q_token_type_ids=None,
                p_input_ids=None, p_attention_mask=None, p_token_type_ids=None,
                hn_input_ids=None, hn_attention_mask=None, hn_token_type_ids=None,
                labels=None, return_loss=True):

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


#
class LayerAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.H = hidden_size

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_mult * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * hidden_size, hidden_size),
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.info = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, cls_layers):
        # cls_layers List[[B, H]]
        X = torch.stack(cls_layers, dim=1)
        B, L, H = X.shape
        info_tok = self.info.expand(B, 1, self.H)

        Z = torch.cat([info_tok, X], dim=1)

        attn_out, _ = self.attn(Z, Z, Z)

        Z = self.norm1(Z + self.dropout(attn_out))
        ffn_out = self.ffn(Z)
        Z = self.norm2(Z + self.dropout(ffn_out))

        return Z[:, 0, :]
