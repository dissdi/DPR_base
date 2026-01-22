import torch
import torch.nn as nn
from transformers import AutoModel

class ReRanker(nn.Module):
    def __init__(self, emebdding_dim=768, nheads=8, dropout=0.1):
        super().__init__()
        self.embedding_dim = emebdding_dim
        self.nheads = nheads
        self.dropout = dropout
        
        self.MultiheadAttn = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.nheads, dropout=self.dropout, batch_first=True)
        self.fc = nn.Linear(self.embedding_dim, 1)
        
    def forward(self, query_emb, passage_cls_emb): # q: [B, L, D], p: [B, N, D]
        # expand query embedding to match passage embeddings
        B, N, D = passage_cls_emb.size()
        query_emb_expanded = query_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, D]
        
        attn_output, _ = self.MultiheadAttn(passage_cls_emb, query_emb_expanded, query_emb_expanded)  # [B, N, D]
        scores = self.fc(attn_output).squeeze(-1)  # [B, N]
        return scores  # [B, N]
    
class ReRankerDPR(nn.Module):
    def __init__(self, reranker: ReRanker, dpr_model: nn.Module, k: int = 5, dpr_loss_fn = None, reranker_loss_fn = None):
        super().__init__()
        self.reranker = reranker
        self.dpr_model = dpr_model
        
        self.k = k  # number of top passages to rerank
        self.dpr_loss_fn = dpr_loss_fn if dpr_loss_fn is not None else nn.CrossEntropyLoss()
        self.reranker_loss_fn = reranker_loss_fn if reranker_loss_fn is not None else nn.CrossEntropyLoss()
        
    def forward(self, 
                q_input_ids = None, q_attention_mask = None, q_token_type_ids = None,
                p_input_ids = None, p_attention_mask = None, p_token_type_ids = None,
                hn_input_ids = None, hn_attention_mask = None, hn_token_type_ids = None,
                 labels = None, return_loss=True):
        q_emb_all = self.dpr_model.p_encoder(input_ids=q_input_ids,
            attention_mask=q_attention_mask,
            token_type_ids=q_token_type_ids,
            return_dict=True).last_hidden_state
        
        q_emb = self.dpr_model.dropout(q_emb_all[:, 0, :])
        p_emb = self.dpr_model.encode_passage(p_input_ids, p_attention_mask, p_token_type_ids)

        # if hard-negative is given encode negative and concat p_emb, hn_emb
        if hn_input_ids is not None:
            hn_emb = self.dpr_model.encode_passage(hn_input_ids, hn_attention_mask, hn_token_type_ids)
            p_emb = torch.cat([p_emb, hn_emb], dim=0)

        # dot product
        sim_matrix = q_emb @ p_emb.T

        if labels is None:
            labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        dpr_loss = self.dpr_loss_fn(sim_matrix, labels) if return_loss else None

        #여기까지는 dpr와 동일
        
        # Reranking
        topk_scores, topk_indices = torch.topk(sim_matrix, self.k-1, dim=1)  # [B, k]
        
        selected_p_emb = p_emb[topk_indices]  # [B, k, D]
        reranked_scores_batch = self.reranker(q_emb, selected_p_emb)  # [B, k]
        if return_loss:
            batch_range = torch.arange(topk_indices.size(0), device=topk_indices.device)
            matches = (topk_indices == batch_range.unsqueeze(1))
            valid_mask = matches.any(dim=1)
            if valid_mask.sum() > 0:
                final_scores = reranked_scores_batch[valid_mask]  # [Valid_B, K]
                target_indices = matches[valid_mask].float().argmax(dim=1)  # [Valid_B]
                reranker_loss = self.reranker_loss_fn(final_scores, target_indices)
            else:
                reranker_loss = torch.tensor(0.0, device=topk_indices.device, requires_grad=True)
        else:
            reranker_loss = None
        loss = dpr_loss + reranker_loss if return_loss else None
        return {
            "loss": loss,
            "logits": sim_matrix
        }