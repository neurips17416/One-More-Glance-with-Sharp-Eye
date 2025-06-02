from typing import List, Optional, Tuple, Union
import random

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import (
    OPTForCausalLM, OPTModel, OPTDecoder, OPTDecoderLayer, OPTAttention, OptFlashAttention2
)

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.trainer import logger

from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions
)

from transformers.utils import is_flash_attn_greater_or_equal_2_10

_CONFIG_FOR_DOC = "LlavaOPTcrossattnConfig"

class OPTAttentionBottleNect(OPTAttention):
    """
        Multi-headed attention from 'Attention Is All You Need' paper
        For self-attention, the operation is the same to the original OPTAttention.
        For cross-attnetion, the hidden dim of projector is reduced.
    """

    def __init__(
        self,
        config: OPTConfig,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        **kwargs,
    ):
        super().__init__(config, is_decoder=is_decoder, **kwargs)      

        if is_cross_attention:
            reduced_embed_dim = int(self.embed_dim // config.crossatt_project_reduction)
            DIMtoNUMHEADS = {768: 12, 512: 8, 384: 6, 256: 4, 192: 3, 128: 2}
            assert reduced_embed_dim in DIMtoNUMHEADS, \
                f"embed_dim({self.embed_dim})/reduction({config.crossatt_project_reduction})={reduced_embed_dim} must be in {DIMtoNUMHEADS.keys()}"
            self.num_heads = DIMtoNUMHEADS[reduced_embed_dim]
            logger.warning_once(f"Warning: Num_heads is set to {self.num_heads} for cross-attention.")
            
            self.head_dim = reduced_embed_dim // self.num_heads

            self.scaling = self.head_dim**-0.5
            # redefine the projectors
            self.k_proj = nn.Linear(self.embed_dim, reduced_embed_dim, bias=self.enable_bias)
            self.v_proj = nn.Linear(self.embed_dim, reduced_embed_dim, bias=self.enable_bias)
            self.q_proj = nn.Linear(self.embed_dim, reduced_embed_dim, bias=self.enable_bias)
            self.out_proj = nn.Linear(reduced_embed_dim, self.embed_dim, bias=self.enable_bias)
            self.embed_dim = reduced_embed_dim


class OptFlashAttentionBottleNect2(OptFlashAttention2, OPTAttentionBottleNect):
    def __init__(self, config: OPTConfig, is_decoder: bool = False, is_cross_attention: bool = False, **kwargs):
        OPTAttentionBottleNect.__init__(self, config, is_decoder=is_decoder, is_cross_attention=is_cross_attention, **kwargs)
        
        # copied from OptFlashAttention2
        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(self, *args, **kwargs):
        return OptFlashAttention2.forward(self, *args, **kwargs)


OPT_ATTENTION_CLASSES = {
    "eager": OPTAttentionBottleNect,
    "flash_attention_2": OptFlashAttentionBottleNect2,
}

class OPTcrossattnDecoderLayer(OPTDecoderLayer):
    def __init__(self, config):
        super().__init__(config)

        if config.add_cross_attention:
            self.crossattention = OPT_ATTENTION_CLASSES[config._attn_implementation](config=config, is_decoder=True, is_cross_attention=True)
            self.ln_crossattention = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)
            ### CHECK LIST ###
            # embed_dim=self.embed_dim,
            # num_heads=config.num_attention_heads,
            # dropout=config.attention_dropout,
            # is_decoder=True,
            # bias=config.enable_bias,
            # config=config,

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        ## for cross-attention
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        cross_attn_head_mask = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        #region: Cross-Attention, only this part is different from OPTDecoderLayer
        if encoder_hidden_states is not None:
            # cross_attn_present_key_value = None
            # cross_attn_weights = None
            residual = hidden_states
            if self.do_layer_norm_before: # 125m, 1.7B, ..., 175B
                hidden_states = self.ln_crossattention(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # hidden_states, cross_attn_weights, cross_attn_present_key_value
            cross_attn_outputs = self.crossattention(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            attn_output = nn.functional.dropout(cross_attn_outputs[0], p=self.dropout, training=self.training)
            hidden_states = residual + attn_output

            if not self.do_layer_norm_before: # 350m
                hidden_states = self.ln_crossattention(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple 
            present_key_value = present_key_value + cross_attn_outputs[-1]
        else:
            raise ValueError("Cross-Attention is enabled but no encoder_hidden_states provided")
        #endregion.

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class OPTcrossattnDecoder(OPTDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([OPTcrossattnDecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ## for cross-attention
        encoder_hidden_states=None,
        encoder_attention_mask = None,
        cross_attn_head_mask = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            causal_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            attention_mask = (
                torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
                if attention_mask is None
                else attention_mask
            )
        else:
            # 4d mask is passed through the layers
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
            elif attention_mask.shape[1] != mask_seq_length:
                raise ValueError(
                    f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                    f"{mask_seq_length} (sum of the lengths of current and past inputs)"
                )
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                    # for cross-attention
                    encoder_hidden_states,
                    encoder_attention_mask,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    # for cross-attention
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    cross_attn_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            # for cross-attention
            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs[2],)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class OPTcrossattnModel(OPTModel):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = OPTcrossattnDecoder(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ## for cross-attention
        encoder_hidden_states=None,
        encoder_attention_mask = None,
        cross_attn_head_mask = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # FIXME: Always, encoder_attention_mask = None
            pass

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # for cross-attention
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            cross_attn_head_mask=cross_attn_head_mask,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )
    
class OPTcrossattnForCausalLM(OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = OPTcrossattnModel(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ## for cross-attention
        encoder_hidden_states=None,
        encoder_attention_mask = None,
        cross_attn_head_mask = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # for cross-attention
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            cross_attn_head_mask=cross_attn_head_mask,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )