import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from core.abstract_module import AbstractModule
from typing import Optional, Tuple

from transformers.models.marian.modeling_marian import MarianAttention, MarianMTModel, GenerationMixin


class OpusModule(AbstractModule):
    POSITION_MODE_LIST = ["encoder", "decoder", "encoder-decoder"]
    LAYER_MIX_MODE_LIST = ["first", "final", "average"]
    HEAD_MIX_MODE_LIST = ["all", "first", "average"]

    def __init__(self, language="zh-en"):
        super().__init__()
        self.language = language.lower()
    
    @staticmethod
    def marian_attention_forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        print(self)
        return attn_output, attn_weights_reshaped, past_key_value

    def load(self):
        path = os.path.dirname(os.path.abspath(__file__))
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, f"opus-mt-{self.language}"))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(path, f"opus-mt-{self.language}"))
        self.model.eval()
        def gen_hook(position, index):
            if position == "encoder":
                def hook(module, input):
                    self.encoder_buffer[index, :, :, :] = self.marian_attention_forward(module.self_attn, *input)
                return hook
            elif position == "decoder":
                def hook(module, input):
                    self.decoder_buffer[index, :, :, :] = self.marian_attention_forward(module.self_attn, *input)
                return hook
            elif position == "encoder-decoder":
                def hook(module, input):
                    self.encoder_decoder_buffer[index, :, :, :] = self.marian_attention_forward(module, *input)
                return hook
        # for ind, layer in enumerate(self.model.model.encoder.layers):
        #     layer.register_forward_pre_hook(gen_hook("encoder", ind))
        # for ind, layer in enumerate(self.model.model.decoder.layers):
        #     layer.register_forward_pre_hook(gen_hook("decoder", ind))
        #     layer.encoder_attn.register_forward_pre_hook(gen_hook("encoder-decoder", ind))
    
    def unload(self):
        del self.model
        del self.tokenizer
        gc.collect()

    def get_name(self):
        return f"Helsinki-NLP/opus-mt-{self.language}"
    
    def get_description(self):
        return f"Translation model for {self.language.replace('-', ' to ')}"
    
    def forward(self, sentence):
        encoded = self.tokenizer([sentence], return_tensors="pt")
        self.encoder_buffer = torch.zeros((6, 8, encoded["input_ids"].shape[1], encoded["input_ids"].shape[1]))
        self.decoder_buffer = torch.zeros((6, 8, encoded["input_ids"].shape[1], encoded["input_ids"].shape[1]))
        self.encoder_decoder_buffer = torch.zeros((6, 8, encoded["input_ids"].shape[1], encoded["input_ids"].shape[1]))
        self.input = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"].squeeze(0))
        with torch.no_grad():
            translation = self.model.generate(**encoded)
        result = self.tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
        print(f"Translation: {result}")
    
    def get_sentence(self, position_mode):
        pass

    def get_position_mode_list(self):
        return self.POSITION_MODE_LIST
    
    def get_layer_mix_mode_list(self):
        return self.LAYER_MIX_MODE_LIST
    
    def get_head_mix_mode_list(self):
        return self.HEAD_MIX_MODE_LIST
    
    def get_n_head(self, position_mode, layer_mix_mode, head_mix_mode):
        pass

    def get_attention_weights(self, key, position_mode, layer_mix_mode, head_mix_mode, temperature):
        pass