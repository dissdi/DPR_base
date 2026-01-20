import torch
import torch.nn as nn
from transformers import AutoTokenizer

class CMCLS_tokenizer(nn.Module):
    def __init__(self):
        super().__init__() 
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize(self, text_a, text_b=None, max_length=256, padding="max_length", truncation=True):
        # number of special tokens to be inserted
        indexes = [0, max_length//5, 2*max_length//5, 3*max_length//5, 4*max_length//5]
        reserve = len(indexes)
        
        tokens = self.tokenizer(
            text_a, text_b,         # NOTICE: no SEP between text_a and text_b
            max_length=max_length-reserve,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        
        input_ids = tokens["input_ids"].tolist()
        attention_mask = tokens["attention_mask"].tolist()
        token_type_ids = tokens["token_type_ids"].tolist()
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        # CLS token insert
        mcls_positions = []
        for idx in indexes:
            input_ids[0].insert(idx, cls_id)
            attention_mask[0].insert(idx, 1)
            token_type_ids[0].insert(idx, 0)
            mcls_positions.append(idx)
            
        # SEP token insert
        last = None
        for i, val in enumerate(input_ids[0]):
            if val == pad_id:
                last = i
                break
        if last == None:
            last = len(input_ids[0]) - 1
        input_ids[0][last] = sep_id
        attention_mask[0][last] = 1
        token_type_ids[0][last] = 0
                
        # truncate max length
        input_ids = input_ids[0][:max_length]
        attention_mask = attention_mask[0][:max_length]
        token_type_ids = token_type_ids[0][:max_length]
        mcls_positions = [pos for pos in mcls_positions if pos < last]
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids, "mcls_positions": mcls_positions, "last":last}
    
if __name__ == "__main__":
    tokenizer = CMCLS_tokenizer()
    ta = "tokenizer test 1 2 3 attention mask token type ids, mcls positions ta mcls_positions"
    tb = "dataloader_drop_last: True num_train_epochs: 1 logging_steps: 100 remove_unused_columns: False push_to_hub: False"
    out1 = tokenizer.tokenize(ta, max_length=256, padding="max_length", truncation=True)
    out2 = tokenizer.tokenize(ta, tb, max_length=256, padding="max_length", truncation=True)
    print(len(out1["input_ids"]), len(out1["attention_mask"]), len(out1["token_type_ids"]), len(out1["mcls_positions"]))
    print(len(out2["input_ids"]), len(out2["attention_mask"]), len(out2["token_type_ids"]), len(out2["mcls_positions"]))

    input_ids_list = out1["input_ids"]
    mcls_positions = out1["mcls_positions"]
    for pos in mcls_positions:
        if tokenizer.tokenizer.cls_token_id != input_ids_list[pos]:
            print("ERROR, mcls pos does not match with other list")
        else:
            print(f"{pos}th element matches")
    last = out1["last"]
    print(f"last is SEP = {tokenizer.tokenizer.sep_token_id == input_ids_list[last]}")
    
    input_ids_list = out2["input_ids"]
    mcls_positions = out2["mcls_positions"]
    for pos in mcls_positions:
        if tokenizer.tokenizer.cls_token_id != input_ids_list[pos]:
            print("ERROR, mcls pos does not match with other list")
        else:
            print(f"{pos}th element matches")
    last = out2["last"]
    print(f"last is SEP = {tokenizer.tokenizer.sep_token_id == input_ids_list[last]}")
       
    print("Test complete")