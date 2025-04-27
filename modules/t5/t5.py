from transformers import T5ForConditionalGeneration, T5Tokenizer
from core.abstract_module import AbstractModule
import torch
import os
import gc

path = os.path.dirname(os.path.abspath(__file__))

DOCX = """
<div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: auto; color: #333;">
    <h1 style="color: #2C3E50; border-bottom: 2px solid #8E44AD; padding-bottom: 8px;">T5-translate-en-ru-zh - Translation Bridge</h1>
    
    <div style="padding: 20px; margin: 15px 0; border: 1px solid #eee; border-radius: 5px;">
        <h2 style="color: #8E44AD;">▍ Architecture Overview</h2>
        <ul style="padding-left: 25px;">
            <li><strong>Encoder-Decoder</strong>: Full transformer structure</li>
            <li><strong>Special Token</strong>: Uses [translate] prompts</li>
            <li><strong>Multilingual</strong>: Handles ZH/RU/EN translation</li>
        </ul>
    </div>

    <div style="padding: 20px; border: 1px solid #eee; border-radius: 5px; margin: 15px 0;">
        <h2 style="color: #27AE60;">▍ Translation Process</h2>
        <ol style="padding-left: 25px;">
            <li>Input: "Hello world!"</li>
            <li>Encoder analysis → Context understanding</li>
            <li>Decoder generation → Output translation</li>
        </ol>
    </div>
</div>
"""

class T5Module(AbstractModule):
    POSITION_MODE_LIST = ["encoder", "decoder", "encoder-decoder"]
    LAYER_MIX_MODE_LIST = ["first", "final", "average"]
    HEAD_MIX_MODE_LIST = ["all", "first", "average"]

    def __init__(self):
        self.target_language = "zh"

    @staticmethod
    def t5_attention_forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, seq_length, key_length) (causal decoder)
        batch_size, seq_length = hidden_states.shape[:2]

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            key_length = key_states.shape[-2]
            # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
            real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
            # real_seq_length = key_length # it may be wrong
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device, cache_position=cache_position
                )
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = position_bias + causal_mask

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        return scores.squeeze(0).transpose(-1, -2)

    def load(self):
        self.model = T5ForConditionalGeneration.from_pretrained(os.path.join(path, "large"))
        self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(path, "large"))
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        def gen_hook(position, index):
            if position == "encoder":
                def hook(module, args, kwargs, output):
                    if self.disable_hooks:
                        return
                    self.encoder_buffer[index, :, :, :] = self.t5_attention_forward(module, *args, **kwargs)
            elif position == "decoder":
                def hook(module, args, kwargs, output):
                    if self.disable_hooks or args[0].shape[1] != self.out_len:
                        return
                    # if index == 0:
                    #     print(index, args[0].shape, self.t5_attention_forward(module, *args, **kwargs).shape)
                    self.decoder_buffer[index, :, :, :] = self.t5_attention_forward(module, *args, **kwargs)
            elif position == "encoder-decoder":
                def hook(module, args, kwargs, output):
                    if self.disable_hooks or args[0].shape[1] != self.out_len:
                        return
                    # if index == 0:
                    #     print(index, args[0].shape, self.t5_attention_forward(module, *args, **kwargs).shape)
                    self.encoder_decoder_buffer[index, :, :, :] = self.t5_attention_forward(module, *args, **kwargs)
            else:
                raise ValueError(f"Unsupported position: {position}")
            return hook
        for ind, layer in enumerate(self.model.encoder.block):
            layer.layer[0].SelfAttention.register_forward_hook(gen_hook("encoder", ind), with_kwargs=True)
        for ind, layer in enumerate(self.model.decoder.block):
            layer.layer[0].SelfAttention.register_forward_hook(gen_hook("decoder", ind), with_kwargs=True)
            layer.layer[1].EncDecAttention.register_forward_hook(gen_hook("encoder-decoder", ind), with_kwargs=True)

    def unload(self):
        del self.model
        del self.tokenizer
        if hasattr(self, "encoder_buffer"):
            del self.encoder_buffer
        if hasattr(self, "decoder_buffer"):
            del self.decoder_buffer
        if hasattr(self, "encoder_decoder_buffer"):
            del self.encoder_decoder_buffer
        if hasattr(self, "input"):
            del self.input
        if hasattr(self, "output"):
            del self.output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_name(self):
        return "T5 Large"
    
    def get_description(self):
        return DOCX
    
    def forward(self, sentence):
        header = f"translate to {self.target_language}: "
        inputs = self.tokenizer(header + sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
        self.input = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        in_len = inputs["input_ids"].shape[1]
        self.encoder_buffer = torch.zeros(24, 16, in_len, in_len)
        with torch.no_grad():
            self.disable_hooks = True
            outputs = self.model.generate(
                **inputs.to("cuda" if torch.cuda.is_available() else "cpu"),
                num_beams=1,
                do_sample=False
            )
            self.output = self.tokenizer.convert_ids_to_tokens(outputs[0])
            self.out_len = len(self.output) - 1
            self.decoder_buffer = torch.zeros(24, 16, self.out_len, self.out_len)
            self.encoder_decoder_buffer = torch.zeros(24, 16, in_len, self.out_len)
            self.disable_hooks = False
            self.model.generate(
                **inputs.to("cuda" if torch.cuda.is_available() else "cpu"),
                num_beams=1,
                do_sample=False,
                use_cache=False
            )

    def get_sentence(self, position_mode: str):
        if position_mode == "encoder":
            try:
                return self.input, self.input
            except AttributeError:
                raise RuntimeError("Please run forward() before get_sentence()")
        elif position_mode == "decoder":
            try:
                return self.output[1:], self.output[:-1]
            except AttributeError:
                raise RuntimeError("Please run forward() before get_sentence()")
        elif position_mode == "encoder-decoder":
            try:
                return self.output[:-1], self.input
            except AttributeError:
                raise RuntimeError("Please run forward() before get_sentence()")
        raise ValueError(f"Unsupported position mode: {position_mode}")
    
    def get_position_mode_list(self):
        return self.POSITION_MODE_LIST
    
    def get_layer_mix_mode_list(self):
        return self.LAYER_MIX_MODE_LIST
    
    def get_head_mix_mode_list(self):
        return self.HEAD_MIX_MODE_LIST
    
    def get_n_head(self, position_mode, layer_mix_mode, head_mix_mode):
        if head_mix_mode == "all":
            return 16
        return 1
    
    def get_attention_weights(self,
                              key: int,
                              position_mode: str,
                              layer_mix_mode: str,
                              head_mix_mode: str,
                              temperature: float):
        if position_mode == "encoder":
            buffer = self.encoder_buffer
        elif position_mode == "decoder":
            buffer = self.decoder_buffer
        elif position_mode == "encoder-decoder":
            buffer = self.encoder_decoder_buffer
        else:
            raise ValueError(f"Unsupported position mode: {position_mode}")
        if head_mix_mode == "average":
            if layer_mix_mode == "first":
                res = buffer.mean(dim=1)[0, :, key]
            elif layer_mix_mode == "final":
                res = buffer.mean(dim=1)[-1, :, key]
            elif layer_mix_mode == "average":
                res = buffer.mean(dim=1).mean(dim=0)[:, key]
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
            res = buffer[0, head, :, key]
        elif layer_mix_mode == "final":
            res = buffer[-1, head, :, key]
        elif layer_mix_mode == "average":
            res = buffer.mean(dim=0)[head, :, key]
        else:
            raise ValueError(f"Unsupported layer mix mode: {layer_mix_mode}")
        # if position_mode == "decoder":
        #     res = res[:, 1:]
        return torch.nn.functional.softmax(res / temperature, dim=-1).tolist()
    
    def get_other_info(self):
        return {
            "Target Language": ["zh", "ru", "en"]
        }

    def set_other_info(self, info):
        self.target_language = info["Target Language"]