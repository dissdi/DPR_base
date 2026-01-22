import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F

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
    
    def gather_positions(self, out, mcls_positions):
        B, L, H = out.last_hidden_state.shape
        
        # (M,), (1,) -> (1, M), (1, 1)
        if mcls_positions.dim() == 0:
            mcls_positions = mcls_positions.view(1, 1)
        elif mcls_positions.dim() == 1:
            if mcls_positions.numel() == B:
                mcls_positions = mcls_positions.view(B, 1)   # (B,) -> (B,1)
            else:
                mcls_positions = mcls_positions.unsqueeze(0)  # (M,) -> (1,M)
        
        # match batch (1, M) -> (B, M)
        pos_B, M = mcls_positions.shape
        if pos_B == 1 and B > 1:
            mcls_positions = mcls_positions.expand(B, M)
            pos_B, M = mcls_positions.shape
        elif pos_B != B and pos_B != 1:
            raise ValueError
        
        mcls_positions = mcls_positions.to(device=out.last_hidden_state.device, dtype=torch.long)
        
        # match shape
        mcls_positions = torch.unsqueeze(mcls_positions, dim=-1).expand(B, M, H)
        
        return out.last_hidden_state.gather(dim=1, index=mcls_positions)

    def _masked_topk_sum(self, S_flat, mask_flat, k_pair, largest: bool):
        B, P, D = S_flat.shape
        device = S_flat.device
        dtype = S_flat.dtype
    
        max_k = int(k_pair.max().item())
        max_k = min(max_k, D)
    
        if largest:
            filled = S_flat.masked_fill(~mask_flat, float("-inf"))
            vals = torch.topk(filled, k=max_k, dim=2, largest=True, sorted=False).values  # (B,P,max_k)
        else:
            filled = S_flat.masked_fill(~mask_flat, float("inf"))
            vals = torch.topk(filled, k=max_k, dim=2, largest=False, sorted=False).values  # (B,P,max_k)
    
        rank = torch.arange(max_k, device=device).view(1, 1, max_k)               # (1,1,max_k)
        keep = rank < k_pair.clamp(min=1, max=max_k).unsqueeze(-1)                # (B,P,max_k) bool
        return (vals.masked_fill(~keep, 0.0)).sum(dim=2)                          # (B,P)
    
        
    def encode_query(self, input_ids = None, attention_mask = None, token_type_ids = None, mcls_positions = None, mcls_mask = None):
        out = self.q_encoder(input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True)
        mcls = self.gather_positions(out, mcls_positions)
        mcls_mask = mcls_mask.to(device=out.last_hidden_state.device, dtype=torch.bool)
        return mcls, mcls_mask
        
    def encode_passage(self, input_ids = None, attention_mask = None, token_type_ids = None, mcls_positions = None, mcls_mask = None):
        out = self.p_encoder(input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True)
        mcls = self.gather_positions(out, mcls_positions)
        mcls_mask = mcls_mask.to(device=out.last_hidden_state.device, dtype=torch.bool)
        return mcls, mcls_mask

    def forward(self, 
                q_input_ids = None, q_attention_mask = None, q_token_type_ids = None, q_mcls_positions = None, q_mcls_mask = None,
                p_input_ids = None, p_attention_mask = None, p_token_type_ids = None, p_mcls_positions = None, p_mcls_mask = None,
                hn_input_ids = None, hn_attention_mask = None, hn_token_type_ids = None, hn_mcls_positions = None, hn_mcls_mask = None,
                labels = None, return_loss=True):

        # query, passage encode
        q_emb, q_mask = self.encode_query(q_input_ids, q_attention_mask, q_token_type_ids, q_mcls_positions, q_mcls_mask)
        p_emb, p_mask = self.encode_passage(p_input_ids, p_attention_mask, p_token_type_ids, p_mcls_positions, p_mcls_mask)

        # if hard-negative is given encode negative and concat p_emb, hn_emb
        if hn_input_ids is not None:
            hn_emb, hn_mask = self.encode_passage(hn_input_ids, hn_attention_mask, hn_token_type_ids, hn_mcls_positions, hn_mcls_mask)
            p_emb = torch.cat([p_emb, hn_emb], dim=0)
            p_mask = torch.cat([p_mask, hn_mask], dim=0)
        q_mask, p_mask = q_mask.to(q_emb.device), p_mask.to(q_emb.device)
        
        # batch, passage length
        B = q_emb.size(0)
        P = p_emb.size(0)
        
        # learning parameter
        alpha, beta = self.get_alpha_beta()
        alpha, beta = alpha.to(q_emb.device), beta.to(q_emb.device)
        
        # get similarity “각 (b,p) 쌍마다 Mq×Mp 점수표”
        S = torch.einsum("bih,pjh->bpij", q_emb, p_emb)  # (B,P,Mq,Mp)

        # make pair mask
        pair_mask = (q_mask[:, None, :, None] & p_mask[None, :, None, :])

        # flatten: (B,P,D)
        S_flat = S.reshape(B, P, -1)
        mask_flat = pair_mask.reshape(B, P, -1)
        
        
        nq = q_mask.sum(dim=1).to(torch.long)          # (B,)
        np_ = p_mask.sum(dim=1).to(torch.long)         # (P,)
        n_pair = (nq[:, None] * np_[None, :])          # (B,P)
        valid_pair = n_pair > 0 
        n_safe = n_pair.clamp(min=1) 

        k_pair = (4 * n_safe) // 10
        k_pair = torch.clamp(k_pair, min=1, max=S_flat.size(2))

        l_pair = (2 * n_safe) // 10
        l_pair = torch.clamp(l_pair, min=1, max=S_flat.size(2)) 

        count = mask_flat.sum(dim=2).clamp(min=1)          # (B,P)
        total_mean = (S_flat * mask_flat).sum(dim=2) / count  # (B,P)   
        
        top_mean = top_sum / k_pair.clamp(min=1).to(top_sum.dtype)          # (B,P)
        bottom_mean = bottom_sum / l_pair.clamp(min=1).to(bottom_sum.dtype) # (B,P)
        bottom_mean = max(0, -bottom_mean)
        
        # reinforce good score and bad score
        sim_list = total_mean + alpha * top_mean - beta * bottom_mean
        sim_list = sim_list.masked_fill(~valid_pair, -1e9)
    
        if labels is None:
            labels = torch.arange(sim_list.size(0), device=sim_list.device) 

        loss = self.loss_fn(sim_list, labels) if return_loss else None 

        return {
            "loss": loss,
            "logits": sim_list
        }
