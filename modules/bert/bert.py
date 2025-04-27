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
    <h1 style="color: #2C3E50; border-bottom: 2px solid #3498DB; padding-bottom: 8px;">BERT - 双向文本理解引擎</h1>
    
    <div style="padding: 20px; border: 1px solid #eee; border-radius: 5px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05)">
        <h2 style="color: #3498DB; margin-top: 0;">核心功能</h2>
        <p>BERT (Bidirectional Encoder Representations from Transformers) 能够：</p>
        <ul style="list-style-type: '- '; padding-left: 25px;">
            <li>理解词语的上下文关系</li>
            <li>捕捉句子深层语义</li>
            <li>识别文本中的隐含模式</li>
        </ul>
        <p>通过双向上下文分析，同时理解前后文本关联。</p>
    </div>

    <div style="margin-top: 25px; display: flex; gap: 15px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 250px; padding: 15px; border: 1px solid #E8F6FF; border-radius: 5px;">
            <h3 style="color: #2980B9; margin-top: 0;">技术特点</h3>
            <p>• 掩码语言建模<br>
            • 双向上下文处理<br>
            • 预训练+微调模式</p>
        </div>

        <div style="flex: 1; min-width: 250px; padding: 15px; border: 1px solid #EAFAF1; border-radius: 5px;">
            <h3 style="color: #27AE60; margin-top: 0;">主要优势</h3>
            <p>• 上下文感知能力<br>
            • 多任务适应性<br>
            • 深度语义理解</p>
        </div>
    </div>

    <div style="margin-top: 25px; padding: 20px; border: 1px solid #FEF5E7; border-radius: 5px;">
        <h2 style="color: #F39C12;">典型应用</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;">
            <div style="padding: 10px; border: 1px solid #eee; border-radius: 4px;">
                <h4 style="margin: 5px 0;">语义搜索</h4>
                <p>理解搜索意图</p>
            </div>
            <div style="padding: 10px; border: 1px solid #eee; border-radius: 4px;">
                <h4 style="margin: 5px 0;">文本分类</h4>
                <p>文档类型识别</p>
            </div>
            <div style="padding: 10px; border: 1px solid #eee; border-radius: 4px;">
                <h4 style="margin: 5px 0;">问答系统</h4>
                <p>上下文推理</p>
            </div>
        </div>
    </div>
</div>
"""

DOCX_EN = """
<!-- English Version -->
<div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: auto; color: #333;">
    <h1 style="color: #2C3E50; border-bottom: 2px solid #3498DB; padding-bottom: 8px;">BERT - Bidirectional Text Understanding</h1>
    
    <div style="padding: 20px; border: 1px solid #eee; border-radius: 5px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05)">
        <h2 style="color: #3498DB; margin-top: 0;">Core Capabilities</h2>
        <p>BERT (Bidirectional Encoder Representations from Transformers) can:</p>
        <ul style="list-style-type: '- '; padding-left: 25px;">
            <li>Understand contextual relationships</li>
            <li>Capture deep semantics</li>
            <li>Recognize textual patterns</li>
        </ul>
        <p>Processes text bidirectionally for full context awareness.</p>
    </div>

    <div style="margin-top: 25px; display: flex; gap: 15px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 250px; padding: 15px; border: 1px solid #E8F6FF; border-radius: 5px;">
            <h3 style="color: #2980B9; margin-top: 0;">Technical Features</h3>
            <p>• Masked language modeling<br>
            • Bidirectional processing<br>
            • Pretrain + fine-tune</p>
        </div>

        <div style="flex: 1; min-width: 250px; padding: 15px; border: 1px solid #EAFAF1; border-radius: 5px;">
            <h3 style="color: #27AE60; margin-top: 0;">Key Advantages</h3>
            <p>• Context-aware embeddings<br>
            • Multi-task adaptability<br>
            • Deep semantic analysis</p>
        </div>
    </div>

    <div style="margin-top: 25px; padding: 20px; border: 1px solid #FEF5E7; border-radius: 5px;">
        <h2 style="color: #F39C12;">Common Applications</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;">
            <div style="padding: 10px; border: 1px solid #eee; border-radius: 4px;">
                <h4 style="margin: 5px 0;">Semantic Search</h4>
                <p>Understanding search intent</p>
            </div>
            <div style="padding: 10px; border: 1px solid #eee; border-radius: 4px;">
                <h4 style="margin: 5px 0;">Text Classification</h4>
                <p>Document categorization</p>
            </div>
            <div style="padding: 10px; border: 1px solid #eee; border-radius: 4px;">
                <h4 style="margin: 5px 0;">QA Systems</h4>
                <p>Contextual reasoning</p>
            </div>
        </div>
    </div>
</div>
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

