import torch
import torch.nn as nn
from transformers import AutoTokenizer

def get_last_position(input_ids, sep_id):
    for i, val in reversed(list(enumerate(input_ids))):
        if val == sep_id:
            return i
    return len(input_ids)-1

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
        mcls_positions = [0]
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids, "mcls_positions": mcls_positions}
    
    def p_tokenize(self, text_a, text_b=None, max_length=256, padding="max_length", truncation=True):
        cls_id = self.tokenizer.cls_token_id
        
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
        for idx in reversed(indexes):
            input_ids[0].insert(idx, cls_id)
            attention_mask[0].insert(idx, 1)
            if idx==0:
                token_type_ids[0].insert(idx, token_type_ids[0][0])
            else:
                token_type_ids[0].insert(idx, token_type_ids[0][idx-1])
            
        # truncate max length
        input_ids = input_ids[0][:max_length]
        attention_mask = attention_mask[0][:max_length]
        token_type_ids = token_type_ids[0][:max_length]
  
        last = get_last_position(input_ids, self.tokenizer.sep_token_id)
        mcls_positions = [pos for pos, val in enumerate(input_ids) if val == cls_id and pos <= last]
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids, "mcls_positions": mcls_positions}
    
if __name__ == "__main__":
    tokenizer = CMCLS_tokenizer()
    ta = "big little lies season 2 how many episodes"
    tb = "Does He Love You \"Does He Love You\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album \"Greatest Hits Volume Two\". It is one of country music's several songs about a love triangle. \"Does He Love You\" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members"
    
    out1 = tokenizer.q_tokenize(ta, max_length=256, padding="max_length", truncation=True)
    print(len(out1["input_ids"]), len(out1["attention_mask"]), len(out1["token_type_ids"]), len(out1["mcls_positions"]))
    input_ids_list = out1["input_ids"]
    mcls_positions = out1["mcls_positions"]
    for pos in mcls_positions:
        if tokenizer.tokenizer.cls_token_id != input_ids_list[pos]:
            print("ERROR, mcls pos does not match with other list")
        else:
            print(f"{pos}th element is CLS?: {tokenizer.tokenizer.cls_token_id == input_ids_list[pos]}")
    
    print()
    out2 = tokenizer.p_tokenize(ta, tb, max_length=256, padding="max_length", truncation=True)
    print(len(out2["input_ids"]), len(out2["attention_mask"]), len(out2["token_type_ids"]), len(out2["mcls_positions"]))
    input_ids_list = out2["input_ids"]
    mcls_positions = out2["mcls_positions"]
    for pos in mcls_positions:
        if tokenizer.tokenizer.cls_token_id != input_ids_list[pos]:
            print("ERROR, mcls pos does not match with other list")
        else:
            print(f"{pos}th element is CLS?: {tokenizer.tokenizer.cls_token_id == input_ids_list[pos]}")
       
    print("Test complete")