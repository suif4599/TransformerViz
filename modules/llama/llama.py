import torch
import os
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, AutoConfig
from core.abstract_module import AbstractModule
from typing import Optional, Tuple
import math

logging.set_verbosity_error()

DOCX = """
<div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: auto; color: #333;">
    <h1 style="color: #2C3E50; border-bottom: 2px solid #E67E22; padding-bottom: 8px;">Llama2-7b - Autoregressive Generative Model</h1>
    
    <div style="padding: 20px; margin: 15px 0; border: 1px solid #eee; border-radius: 5px;">
        <h2 style="color: #E67E22;">▍ Core Architecture</h2>
        <ul style="padding-left: 25px;">
            <li><strong>Decoder-only Architecture</strong>: 32-layer Transformer decoder stack</li>
            <li><strong>Causal Attention Mask</strong>: Prevents information leakage by restricting attention to preceding tokens</li>
            <li><strong>Pre-training Objective</strong>: Next-token prediction (autoregressive generation)</li>
        </ul>
    </div>

    <div style="padding: 20px; border: 1px solid #eee; border-radius: 5px; margin: 15px 0;">
        <h2 style="color: #27AE60;">▍ Technical Specifications</h2>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <tr><td style="padding: 8px; border: 1px solid #eee;">Parameters</td><td style="padding: 8px; border: 1px solid #eee;">7 Billion</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #eee;">Context Window</td><td style="padding: 8px; border: 1px solid #eee;">4,096 Tokens</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #eee;">Training Data</td><td style="padding: 8px; border: 1px solid #eee;">2 Trillion Tokens (mixed corpus)</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #eee;">Decoding Method</td><td style="padding: 8px; border: 1px solid #eee;">Top-p Sampling (nucleus sampling)</td></tr>
        </table>
    </div>

    <div style="padding: 20px; border: 1px solid #eee; border-radius: 5px;">
        <h2 style="color: #E74C3C;">▍ Attention Mechanisms</h2>
        <ul style="columns: 2; padding-left: 20px;">
            <li>Layer-wise triangular masking</li>
            <li>Long-range dependency tracking</li>
            <li>Positional bias patterns</li>
        </ul>
    </div>
</div>
"""
    

class LlamaModule(AbstractModule):
    POSITION_MODE_LIST = ["decoder"]
    LAYER_MIX_MODE_LIST = ["first", "final", "average"]
    HEAD_MIX_MODE_LIST = ["all", "first", "average"]

    def __init__(self):
        super().__init__()
        self.model_size = "7b"
        self.num_layers = 32  # LLaMA-7B的层数
        self.num_heads = 32    # 注意力头数量
    
    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def llama_attention_forward(
        self,
        self_attn,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[torch.Tensor] = None,
        num_heads:Optional[int] = None,
        head_dim:Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:

        query_states = (self_attn.q_proj(hidden_states))
        key_states = (self_attn.k_proj(hidden_states))

        query_states = query_states.view(
            hidden_states.shape[0], 
            hidden_states.shape[1], 
            num_heads, 
            head_dim
        ).transpose(1, 2)

        key_states = key_states.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            num_heads,
            head_dim
        ).transpose(1, 2)

        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

        cos, sin = rotary_emb(query_states, position_ids=position_ids)

        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos=cos,
            sin=sin
        )
        
        # 计算注意力分数
        attn_weights = torch.matmul(
            query_states, 
            key_states.transpose(2, 3)) / math.sqrt(head_dim)
        
        causal_mask = torch.full(
            (attn_weights.shape[2], attn_weights.shape[2]), 
            fill_value=-torch.inf, 
            device=hidden_states.device
        )
        causal_mask = torch.tril(causal_mask, diagonal=-1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attn_weights = attn_weights + attention_mask

        attn_weights = attn_weights + causal_mask
        return attn_weights.squeeze(0)

    def load(self):
        path = os.path.dirname(os.path.abspath(__file__))
        self.model = AutoModelForCausalLM.from_pretrained(os.path.join(path, self.model_size),torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, self.model_size))
        self.config = AutoConfig.from_pretrained(os.path.join(path, self.model_size))
        def gen_hook(index):
            def hook(module, input, output):
                input = (self.hidden_states, self.model.model.rotary_emb, self.num_heads, self.config.hidden_size // self.num_heads)
                self.buffer[index, :, :, :] = self.llama_attention_forward(module, *input)
            return hook
        def q_hook(module, input, output):
            self.hidden_states = input[0]
        
        for ind, layer in enumerate(self.model.model.layers):
            layer.self_attn.q_proj.register_forward_hook(q_hook)
            layer.self_attn.register_forward_hook(gen_hook(ind))
    
    def unload(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_name(self):
        return f"LLaMA-2-{self.model_size.upper()}"
    
    def get_description(self):
        return DOCX
    
    def forward(self, sentence):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(sentence,return_tensors="pt",return_attention_mask=True, padding=True, truncation=True, max_length=128)
        self.buffer = torch.zeros((self.num_layers, self.num_heads, inputs["input_ids"].shape[1], inputs["input_ids"].shape[1]))
        self.hidden_states = torch.zeros((inputs["input_ids"].shape[0], inputs["input_ids"].shape[1], self.config.hidden_size))
        self.input = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_token_ids = torch.argmax(outputs[0], dim=-1)
        self.output = self.tokenizer.convert_ids_to_tokens(predicted_token_ids.squeeze(0).tolist())

    
    def get_sentence(self, position_mode: str):
        if position_mode == "decoder":
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
            res = res[:key + 1]
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
        if head_mix_mode == "first":
            res = res[:key + 1]
        else:
            res = res[:, :key + 1]
        return torch.nn.functional.softmax(res / temperature, dim=-1).tolist()
    
    def get_position_mode_list(self):
        return self.POSITION_MODE_LIST
    
    def get_layer_mix_mode_list(self):
        return self.LAYER_MIX_MODE_LIST
    
    def get_head_mix_mode_list(self):
        return self.HEAD_MIX_MODE_LIST

    def get_n_head(self, position_mode: str, layer_mix_mode: str, head_mix_mode: str):
        if head_mix_mode == "all":
            return self.num_heads
        return 1
