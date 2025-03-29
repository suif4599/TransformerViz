import torch
import os
from transformers import BertTokenizer, BertForMaskedLM

path = os.path.dirname(os.path.abspath(__file__))

name = "english"
name = "chinese"
model = BertForMaskedLM.from_pretrained(os.path.join(path, name))
tokenizer = BertTokenizer.from_pretrained(os.path.join(path, name))

def predict(text, color=True):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    mask_token_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
    if len(mask_token_index) > 0:
        mask_logits = logits[0, mask_token_index, :]
        predicted_token_ids = torch.argmax(mask_logits, dim=-1)
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids)
        for token in predicted_tokens:
            if color:
                token = f"\033[1;31;40m{token}\033[0m"
            text = text.replace("[MASK]", token, 1)
        return text
    return "No [MASK] token found."


while True:
    text = input("input text(with _): ")
    text = text.replace("_", "[MASK]")
    outputs = predict(text)
    print(outputs)
