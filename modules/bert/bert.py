import torch
import os
import gc
from transformers import BertTokenizer, BertForMaskedLM, logging
from core.abstract_module import AbstractModule
from typing import Optional, Tuple
import math

logging.set_verbosity_error()

DOCX_ZH = """
<div style="font-family: 'Microsoft YaHei', Arial, sans-serif; max-width: 800px; margin: auto; color: #333;">
    <h1 style="color: #2C3E50; border-bottom: 2px solid #3498DB; padding-bottom: 8px;">BERT - 上下文理解专家</h1>
    
    <div style="padding: 20px; margin: 15px 0; border: 1px solid #eee; border-radius: 5px;">
        <h2 style="color: #3498DB;">▍ 核心特性</h2>
        <p>作为仅含编码器的模型：</p>
        <ul style="padding-left: 25px;">
            <li><strong>双向理解</strong>：同时分析前后文关系</li>
            <li><strong>预训练任务</strong>：通过遮盖词语（MLM）和句子关系判断（NSP）学习语言规律</li>
            <li><strong>通用性强</strong>：适用于问答/分类/搜索等多种任务</li>
        </ul>
    </div>

    <div style="padding: 20px; border: 1px solid #eee; border-radius: 5px; margin: 15px 0;">
        <h2 style="color: #27AE60;">▍ 可视化重点</h2>
        <ul style="columns: 2; padding-left: 20px;">
            <li>各层注意力热力图</li>
            <li>词语相似度矩阵</li>
            <li>上下文关联强度</li>
            <li>位置编码可视化</li>
        </ul>
    </div>

    <div style="padding: 20px; border: 1px solid #eee; border-radius: 5px;">
        <h2 style="color: #E74C3C;">▍ 技术细节</h2>
        <table style="width: 100%; border-collapse: collapse;">
            <tr><td style="padding: 8px; border: 1px solid #eee;">模型类型</td><td style="padding: 8px; border: 1px solid #eee;">仅编码器</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #eee;">参数量</td><td style="padding: 8px; border: 1px solid #eee;">1.1亿 (Base)</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #eee;">训练数据</td><td style="padding: 8px; border: 1px solid #eee;">维基百科 + 图书语料</td></tr>
        </table>
    </div>
</div>
"""

DOCX_EN = """
<div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: auto; color: #333;">
    <h1 style="color: #2C3E50; border-bottom: 2px solid #3498DB; padding-bottom: 8px;">BERT - Context Understanding Expert</h1>
    
    <div style="padding: 20px; margin: 15px 0; border: 1px solid #eee; border-radius: 5px;">
        <h2 style="color: #3498DB;">▍ Key Features</h2>
        <ul style="padding-left: 25px;">
            <li><strong>Bidirectional</strong>: Context analysis in both directions</li>
            <li><strong>Pre-training Tasks</strong>: Masked Language Modeling (MLM) & Next Sentence Prediction (NSP)</li>
            <li><strong>Versatile</strong>: Adaptable to QA/classification/search tasks</li>
        </ul>
    </div>

    <div style="padding: 20px; border: 1px solid #eee; border-radius: 5px; margin: 15px 0;">
        <h2 style="color: #27AE60;">▍ Visualization Focus</h2>
        <ul style="columns: 2; padding-left: 20px;">
            <li>Layer-wise attention heatmaps</li>
            <li>Word similarity matrices</li>
            <li>Context association strength</li>
            <li>Position encoding patterns</li>
        </ul>
    </div>

    <div style="padding: 20px; border: 1px solid #eee; border-radius: 5px;">
        <h2 style="color: #E74C3C;">▍ Technical Specs</h2>
        <table style="width: 100%; border-collapse: collapse;">
            <tr><td style="padding: 8px; border: 1px solid #eee;">Architecture</td><td style="padding: 8px; border: 1px solid #eee;">Encoder-only</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #eee;">Parameters</td><td style="padding: 8px; border: 1px solid #eee;">110M (Base)</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #eee;">Training Data</td><td style="padding: 8px; border: 1px solid #eee;">Wikipedia + BookCorpus</td></tr>
        </table>
    </div>
</div>
"""

class BertModule(AbstractModule):
    POSITION_MODE_LIST = ["encoder"]
    LAYER_MIX_MODE_LIST = ["first", "final", "average"]
    HEAD_MIX_MODE_LIST = ["all", "first", "average"]

    def __init__(self, language="english"):
        super().__init__()
        if language.lower() not in ["english", "chinese"]:
            raise ValueError(f"Unsupported language: {language}")
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_name(self):
        return f"BERT {self.language.capitalize()}"
    
    def get_description(self):
        # return DOCX % (self.language.capitalize(), )
        return DOCX_ZH if self.language == "chinese" else DOCX_EN
    
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

