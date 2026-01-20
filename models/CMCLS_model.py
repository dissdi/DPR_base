import torch
import torch.nn as nn
from transformers import AutoModel


class DPR(nn.Module):
    def __init__(self, loss_fn = None):
        super().__init__()
        self.q_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.p_encoder = AutoModel.from_pretrained("bert-base-uncased")
        
        self.alpha_raw = nn.Parameter(torch.tensor(1.0))
        self.beta_raw = nn.Parameter(torch.tensor(0.1))
        
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    
    def get_alpha_beta(self):
        alpha = F.softplus(self.alpha_raw)
        beta = F.softplus(self.beta_raw)
        return alpha, beta
    
    def encode_query(self, input_ids = None, attention_mask = None, token_type_ids = None, mcls_positions = None):
        out = self.q_encoder(input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True)
        mcls = out.last_hidden_state[:, mcls_positions, :]
        return mcls
        
    def encode_passage(self, input_ids = None, attention_mask = None, token_type_ids = None, mcls_positions = None):
        out = self.p_encoder(input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True)
        mcls = out.last_hidden_state[:, mcls_positions, :]
        return mcls

    def forward(self, 
                q_input_ids = None, q_attention_mask = None, q_token_type_ids = None,
                p_input_ids = None, p_attention_mask = None, p_token_type_ids = None,
                hn_input_ids = None, hn_attention_mask = None, hn_token_type_ids = None,
                mcls_positions = None, 
                labels = None, return_loss=True):

        # query, passage encode
        q_mcls = self.encode_query(q_input_ids, q_attention_mask, q_token_type_ids, mcls_positions)
        p_mcls = self.encode_passage(p_input_ids, p_attention_mask, p_token_type_ids, mcls_positions)

        # if hard-negative is given encode negative and concat p_emb, hn_emb
        if hn_input_ids is not None:
            hn_emb = self.encode_passage(hn_input_ids, hn_attention_mask, hn_token_type_ids, mcls_positions)
            p_emb = torch.cat([p_emb, hn_emb], dim=0)

        # dot product
        sim_list = []
        for q_cls in q_mcls:
            q_cls = q_cls.squeeze()
            for p_cls in p_mcls:
                p_cls = p_cls.squeeze()
                sim_list.append(q_cls @ p_cls.T)
        sim_list, _ = torch.sort(torch.stack(sim_list, dim=0), descending=True)
        
        # good score and bad score
        n = sim_list.numel()
        k = max(3*n//10, 1)
        l = min(8*n//10, n-1)
        top = sim_list[:k].sum()
        bottom = sim_list[l:].sum()

        # add top and bottom
        alpha, beta = self.get_alpha_beta()
        sim_matrix = alpha*top - beta*bottom

        if labels is None:
            labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        loss = self.loss_fn(sim_matrix, labels) if return_loss else None

        return {
            "loss": loss,
            "logits": sim_matrix
        }
