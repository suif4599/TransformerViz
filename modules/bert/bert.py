import torch
import os
import gc
from transformers import BertTokenizer, BertForMaskedLM, logging
from core.abstract_module import AbstractModule
from typing import Optional, Tuple
import math

logging.set_verbosity_error()

DOCX = """
<!DOCTYPE html>
<html>
<head>
<style>
    .info-card {
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .comparison {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
</head>
<body>

<div class="info-card">
    <h2 style="color: #1976d2; margin-top: 0;">Bert %s</h2>
    <p>BERT model is designed for <span style="color: #d32f2f;">cloze test</span> and provide a basic text understanding model.</p>
</div>

<div class="info-card">
    <h3 style="color: #1976d2;">What It Understands</h3>
    <table style="width: 100%%;">
        <tr>
            <td>✅</td>
            <td>Daily conversations</td>
            <td>"Where's the nearest coffee shop?"</td>
        </tr>
        <tr>
            <td>✅</td>
            <td>Simple questions</td>
            <td>"What's the weather tomorrow?"</td>
        </tr>
        <tr>
            <td>⚠️</td>
            <td>Complex texts</td>
            <td>Technical manuals, legal documents</td>
        </tr>
    </table>
</div>

</body>
</html>
"""

class BertModule(AbstractModule):
    POSITION_MODE_LIST = ["encoder"]
    LAYER_MIX_MODE_LIST = ["first", "final", "average"]
    HEAD_MIX_MODE_LIST = ["all", "first", "average"]

    def __init__(self, language="english"):
        super().__init__()
        self.language = language.lower()
    
    @staticmethod
    def bert_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        use_cache = past_key_value is not None
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        return attention_scores.squeeze(0)

    def load(self):
        path = os.path.dirname(os.path.abspath(__file__))
        self.model = BertForMaskedLM.from_pretrained(os.path.join(path, self.language))
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(path, self.language))
        def gen_hook(index):
            def hook(module, input, output):
                self.buffer[index, :, :, :] = self.bert_attention_forward(module, *input)
            return hook
        for ind, layer in enumerate(self.model.bert.encoder.layer):
            layer.attention.self.register_forward_hook(gen_hook(ind))
    
    def unload(self):
        del self.model
        del self.tokenizer
        if hasattr(self, "buffer"):
            del self.buffer
        if hasattr(self, "input"):
            del self.input
        if hasattr(self, "output"):
            del self.output
        gc.collect()
        torch.cuda.empty_cache()

    def get_name(self):
        return f"BERT {self.language.capitalize()}"
    
    def get_description(self):
        return DOCX % (self.language.capitalize(), )
    
    def forward(self, sentence):
        inputs = self.tokenizer(sentence.replace("_", "[MASK]"), return_tensors="pt", padding=True, truncation=True, max_length=128)
        self.buffer = torch.zeros((12, 12, inputs["input_ids"].shape[1], inputs["input_ids"].shape[1]))
        self.input = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        mask_token_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
        if len(mask_token_index) > 0:
            mask_logits = logits[0, mask_token_index, :]
            predicted_token_ids = torch.argmax(mask_logits, dim=-1)
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_token_ids)
            for token in predicted_tokens:
                sentence = sentence.replace("_", token, 1)
            output = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
            self.output = self.tokenizer.convert_ids_to_tokens(output["input_ids"].squeeze(0))
        else:
            self.output = self.input
    
    def get_sentence(self, position_mode: str):
        if position_mode == "encoder":
            try:
                return self.output, self.input
            except AttributeError:
                raise RuntimeError("Please run forward() before get_sentence()")
        raise ValueError(f"Unsupported position mode: {position_mode}")
    
    def get_attention_weights(self, 
                              key: int, 
                              position_mode: str, 
                              layer_mix_mode: str, 
                              head_mix_mode: str, 
                              temperature: float):
        "k word need q word's attention"
        if head_mix_mode == "average":
            if layer_mix_mode == "first":
                res = self.buffer.mean(dim=1)[0, :, key]
            elif layer_mix_mode == "final":
                res = self.buffer.mean(dim=1)[-1, :, key]
            elif layer_mix_mode == "average":
                res = self.buffer.mean(dim=1).mean(dim=0)[:, key]
            else:
                raise ValueError(f"Unsupported layer mix mode: {layer_mix_mode}")
            return torch.nn.functional.softmax(res / temperature, dim=-1).tolist()
        elif head_mix_mode == "first":
            head = 0
        elif head_mix_mode == "all":
            head = slice(None)
        else:
            raise ValueError(f"Unsupported head mix mode: {head_mix_mode}")
        if layer_mix_mode == "first":
            res = self.buffer[0, head, :, key]
        elif layer_mix_mode == "final":
            res = self.buffer[-1, head, :, key]
        elif layer_mix_mode == "average":
            res = self.buffer.mean(dim=0)[head, :, key]
        else:
            raise ValueError(f"Unsupported layer mix mode: {layer_mix_mode}")
        return torch.nn.functional.softmax(res / temperature, dim=-1).tolist()
    
    def get_position_mode_list(self):
        return self.POSITION_MODE_LIST
    
    def get_layer_mix_mode_list(self):
        return self.LAYER_MIX_MODE_LIST
    
    def get_head_mix_mode_list(self):
        return self.HEAD_MIX_MODE_LIST

    def get_n_head(self, position_mode: str, layer_mix_mode: str, head_mix_mode: str):
        if head_mix_mode == "all":
            return 12
        return 1

