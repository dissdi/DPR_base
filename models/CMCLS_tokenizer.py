import torch
import torch.nn as nn
from transformers import AutoTokenizer

def special_token_pos(input_ids, token_id):
    for i, val in enumerate(input_ids):
        if val == token_id:
            return i
    return len(input_ids)

class CMCLS_tokenizer(nn.Module):
    def __init__(self):
        super().__init__() 
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def q_tokenize(self, text_a, max_length=256, padding="max_length", truncation=True):
        tokens = self.tokenizer(
            text_a,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        input_ids = tokens["input_ids"].tolist()
        attention_mask = tokens["attention_mask"].tolist()
        token_type_ids = tokens["token_type_ids"].tolist()
        
        input_ids = input_ids[0][:max_length]
        attention_mask = attention_mask[0][:max_length]
        token_type_ids = token_type_ids[0][:max_length]
        
        # mcls values
        mcls_positions = [0]
        mcls_mask = torch.tensor([True], dtype=torch.bool)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids, "mcls_positions": mcls_positions, "mcls_mask": mcls_mask}
    
    def p_tokenize(self, text_a, text_b=None, max_length=256, padding="max_length", truncation=True):
        cls_id = self.tokenizer.cls_token_id
        pad_id = self.tokenizer.pad_token_id
        sep_id = self.tokenizer.sep_token_id
        
        # number of special tokens to be inserted
        indexes = [i*max_length//10 for i in range(1, 10, 1)]
        reserve = len(indexes)
        
        tokens = self.tokenizer(
            text_a, text_b,
            max_length=max_length-reserve,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        input_ids = tokens["input_ids"].tolist()
        attention_mask = tokens["attention_mask"].tolist()
        token_type_ids = tokens["token_type_ids"].tolist()
        
        # CLS token insert
        sep_token_pos = special_token_pos(input_ids[0], sep_id)
        for idx in indexes:
            input_ids[0].insert(idx, cls_id)
            attention_mask[0].insert(idx, 1)
            if idx > sep_token_pos:
                token_type_ids[0].insert(idx, 1)
            else:
                token_type_ids[0].insert(idx, 0)
            
        # truncate max length
        input_ids = input_ids[0][:max_length]
        attention_mask = attention_mask[0][:max_length]
        token_type_ids = token_type_ids[0][:max_length]
        
        # CLS after PAD should be 0
        pad_token_pos = special_token_pos(attention_mask, 0)
        for pos in indexes:
            if pos < pad_token_pos:
                attention_mask[pos] = 1
            else:
                attention_mask[pos] = 0
                
        # mcls values
        mcls_positions = [0] + indexes
        mcls_mask = [False for _ in range(len(mcls_positions))]
        for i, pos in enumerate(mcls_positions):
            if input_ids[pos] == cls_id and attention_mask[pos] == 1:
                mcls_mask[i] = True
        mcls_mask = torch.tensor(mcls_mask, dtype=torch.bool)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids, "mcls_positions": mcls_positions, "mcls_mask": mcls_mask}
    
if __name__ == "__main__":
    # Test code
    tokenizer = CMCLS_tokenizer()
    ta = "big little lies season 2 how many episodes"
    tb = "Does He Love You \"Does He Love You\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album \"Greatest Hits Volume Two\". It is one of country music's several songs about a love triangle. \"Does He Love You\" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members"

    out1 = tokenizer.q_tokenize(ta, max_length=256, padding="max_length", truncation=True)
    print(len(out1["input_ids"]), len(out1["attention_mask"]), len(out1["token_type_ids"]), len(out1["mcls_positions"]), len(out1["mcls_mask"]), sum(out1["attention_mask"]))
    print(out1["mcls_mask"])
    input_ids_list = out1["input_ids"]
    mcls_positions = out1["mcls_positions"]
    mcls_mask = out1["mcls_mask"]
    for i, pos in enumerate(mcls_positions):
        print(f"pos:{pos} token_id:{input_ids_list[pos]} mask:{mcls_mask[i]} ")
    
    
    print()
    out2 = tokenizer.p_tokenize(ta, tb, max_length=256, padding="max_length", truncation=True)
    print(len(out2["input_ids"]), len(out2["attention_mask"]), len(out2["token_type_ids"]), len(out2["mcls_positions"]), len(out2["mcls_mask"]), sum(out2["attention_mask"]))
    print(out2["mcls_mask"])
    input_ids_list = out2["input_ids"]
    mcls_positions = out2["mcls_positions"]
    mcls_mask = out2["mcls_mask"]
    for i, pos in enumerate(mcls_positions):
        print(f"pos:{pos} token_id:{input_ids_list[pos]} mask:{mcls_mask[i]} ")
            
