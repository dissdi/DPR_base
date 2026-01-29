import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel



# LayerAttention -> ScalarMix
class DPR_mixcls(nn.Module):
    def __init__(self, loss_fn=None, tau = 0.07, mix_layer = 8, projection = True):
        super().__init__()
        self.q_encoder = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.p_encoder = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

        # q, p 를 나눠서 학습하면 공간 자체가 틀어질 가능성
        self.layer_mix = ScalarMixHead(num_layers=mix_layer, proj=projection)

        assert 1 < mix_layer <= 12, "mix_layer must be <= 12 and > 1"
        self.mix_layer = mix_layer

        self.tau = tau

        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    def encode_query(self, input_ids=None, attention_mask=None, token_type_ids=None):
        layers = self.q_encoder(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict=True).hidden_states[(13 - self.mix_layer):13]  # bert-base is 12 layer

        cls_layers = [h[:, 0, :] for h in layers]
        
        mixed_cls, _ = self.layer_mix(cls_layers)
        
        return F.normalize(mixed_cls, dim=1)


    def encode_passage(self, input_ids=None, attention_mask=None, token_type_ids=None):
        layers = self.p_encoder(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict=True).hidden_states[(13 - self.mix_layer):13]  # bert-base is 12 layer

        cls_layers = [h[:, 0, :] for h in layers]
        mixed_cls, _ = self.layer_mix(cls_layers)
        
        return F.normalize(mixed_cls, dim=1)

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

        sim_matrix = (q_emb @ p_emb.T) / self.tau

        if labels is None:
            labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        loss = self.loss_fn(sim_matrix, labels) if return_loss else None

        return {
            "loss": loss,
            "logits": sim_matrix
        }


#
class ScalarMixHead(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, proj=True, dropout=0.1, init_weight=0.65):
        super().__init__()

        # 각 레이어의 가중치를 결정하는 layer_w
        self.layer_w = nn.Parameter(torch.zeros(num_layers))

        # Dropout Layer
        self.dropout = nn.Dropout(dropout)

        # Layer Norm
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Projection Layer (가중합으로 합친 후 적절한 공간으로 다시 투영[정렬])
        self.proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU()) if proj else nn.Identity()

        # 초기 layer_w를 랜덤하게 사용하면 학습이 불안정해 질 수 있으므로 마지막 레이어의 가중치를 init_weight으로 초기화
        with torch.no_grad():
            self.layer_w.data.fill_(0.0)
            self.layer_w.data[-1] = math.log((init_weight * (num_layers - 1)) / (1.0 - init_weight))

    def forward(self, cls_layers):  # list of [B,H]
        X = torch.stack(cls_layers, dim=1)          # [B,L,H]
        alpha = F.softmax(self.layer_w, dim=0)        # [L]
        z = (X * alpha.view(1, -1, 1)).sum(dim=1)   # [B,H]
        z = self.proj(z)
        z = self.layer_norm(z)
        z = self.dropout(z)
        return z, alpha