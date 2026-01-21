import torch
import torch.nn as nn
from transformers import AutoModel, BitsAndBytesConfig, TrainerCallback

class LossLambdaScheduler(TrainerCallback):
    def __init__(self, max_lambda=1.0, warmup_steps=100):
        self.max_lambda = max_lambda
        self.warmup_steps = warmup_steps

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        current_step = state.global_step
        
        if current_step < self.warmup_steps:
            new_lambda = self.max_lambda * (current_step / self.warmup_steps)
        else:
            new_lambda = self.max_lambda
            
        if hasattr(model, 'module'):
            model.module.loss_lambda = new_lambda
        else:
            model.loss_lambda = new_lambda


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        quantization_config = BitsAndBytesConfig(load_in_8bit=True) 

        self.q_encoder = AutoModel.from_pretrained(
            "bert-base-uncased",
            quantization_config=quantization_config
        )
                
        self.q_restore_layer = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 768)
        )
        
    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None):
        quantized_outputs = self.q_encoder(input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True)
        quantized_output = quantized_outputs.last_hidden_state[:, 0, :].to(torch.float32)
        restored_output = self.q_restore_layer(quantized_output)
        return restored_output
    
class FastDPR(nn.Module):
    def __init__(self, loss_fn = None, base_model = None, tiny_model = None):
        super().__init__()
        self.base_model = base_model
        self.tiny_model = tiny_model
        self.loss_lambda = 0.0
        
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    
    def encode_query_origin(self, input_ids = None, attention_mask = None, token_type_ids = None):
        return self.base_model.encode_query(input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

    def encode_passage(self, input_ids = None, attention_mask = None, token_type_ids = None):
        return self.base_model.encode_passage(input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        
    def encode_query(self, input_ids = None, attention_mask = None, token_type_ids = None):
        return self.tiny_model(input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
    
    def forward(self, 
                q_input_ids = None, q_attention_mask = None, q_token_type_ids = None,
                p_input_ids = None, p_attention_mask = None, p_token_type_ids = None,
                hn_input_ids = None, hn_attention_mask = None, hn_token_type_ids = None,
                 labels = None, return_loss=True):
        
        q_emb = self.encode_query_origin(q_input_ids, q_attention_mask, q_token_type_ids)
        p_emb = self.encode_passage(p_input_ids, p_attention_mask, p_token_type_ids)
        q_tiny_emb = self.encode_query(q_input_ids, q_attention_mask, q_token_type_ids)
        
        if hn_input_ids is not None:
            hn_emb = self.encode_passage(hn_input_ids, hn_attention_mask, hn_token_type_ids)
            p_emb = torch.cat([p_emb, hn_emb], dim=0)
            
        sim_matrix = q_emb @ p_emb.T
        
        if labels is None:
            labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            
        sim_loss = self.loss_fn(sim_matrix, labels) 
        
        restore_loss = nn.MSELoss()(q_emb, q_tiny_emb)
        
        loss = sim_loss + restore_loss * self.loss_lambda
        return {
            "loss": loss,
            "logits": sim_matrix
        }
