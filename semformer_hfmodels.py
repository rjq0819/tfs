from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers import (
    AutoModel,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoConfig,
)

from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Model

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

# from modeling_gpt2 import GPT2Attention, GPT2Block, GPT2LMHeadModel, GPT2Model
from peft import get_peft_model


from transformers.activations import ACT2FN
from transformers.utils import ModelOutput
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.modeling_outputs import (
    CausalLMOutput,
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from sae import SAE


def get_ln(config):
    if "Qwen" in config.architectures[0]:
        return Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    elif "Gemma" in config.architectures[0]:
        return GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    elif "GPT2" in config.architectures[0]:
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
    else:
        return Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


@dataclass
class CausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reppred_loss: Optional[torch.FloatTensor] = None
    aedec_loss: Optional[torch.FloatTensor] = None
    sae_spar_loss: Optional[torch.FloatTensor] = None
    sae_rec_loss: Optional[torch.FloatTensor] = None

@dataclass
class AEOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    aedec_loss: Optional[torch.FloatTensor] = None
    sae_spar_loss: Optional[torch.FloatTensor] = None
    sae_rec_loss: Optional[torch.FloatTensor] = None
    sae_act: Optional[torch.FloatTensor] = None
    hidden_z: torch.FloatTensor = None


logger = logging.get_logger("__main__")

def l2norm(t, dim = -1,  eps = 1e-6):
    return F.normalize(t, p = 2, dim = dim, eps = eps)

def orthogonal_loss_fn(t):
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = torch.einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)


def init_weight(m, config):
    for module in m.modules():
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings

        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = True
        self.is_cross_attention = is_cross_attention
        self.reorder_and_upcast_attn = False

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)


        self.attn_dropout = nn.Identity()
        self.resid_dropout = nn.Identity()
        
        if hasattr(config, "attn_pdrop"):
            self.attn_dropout = nn.Dropout(config.attn_pdrop)
        
        if hasattr(config, "resid_pdrop"):
            self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        self.is_causal = True


    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with torch.amp.autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class PrefixEncoder_MHA(nn.Module):
    def __init__(self, config, model_args):
        super().__init__()

        self.input_dim = config.zdim
        self.hidden_dim = config.hidden_size

        self.prefix_seq_len = config.task_ztokens if hasattr(config, "task_ztokens") else config.ztokens

        self.match_n_layer = model_args.shallow_decoder_n_layer

        self.prefix_generator = model_args.prefix_generator

        if model_args.prefix_generator == "mlp":
            self.prefix_mlp = nn.Linear(
                self.input_dim, self.match_n_layer * 2 * config.hidden_size, False)
        else:
            self.scales = nn.Parameter(
                torch.ones([1]))
            
            num_prompts = self.prefix_seq_len
            self.prompts = nn.Parameter(
                torch.empty(
                [
                    2 * self.match_n_layer, 
                    num_prompts, config.hidden_size
                ]))

            for i in range(self.prompts.shape[0]):
                nn.init.xavier_uniform_(self.prompts[i])

        self.match_n_head = config.num_attention_heads
        self.match_n_embd = config.hidden_size // config.num_attention_heads

    def forward(
        self,
        input_embd
    ):

        batch_size = input_embd.size(0)
        
        if self.prefix_generator == "mlp":
            past_key_values = self.prefix_mlp(input_embd)

            past_key_values = past_key_values.view(
                batch_size, self.prefix_seq_len, self.match_n_layer, -1)

            # Resize
            past_key_values = past_key_values.view(
                batch_size,
                self.prefix_seq_len,
                self.match_n_layer * 2,
                self.match_n_head,
                self.match_n_embd,
            )

            # Transpose -> [match_n_layer*2, batch_size, match_n_head, prefix_seq_len, match_n_embd]
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4])
            past_key_values = torch.split(past_key_values, 2)

        else:
            past_key_values = []
            scale = torch.maximum(torch.ones([]), self.scales[0])

            for j in range(self.match_n_layer):
                k = self.prompts[2*j][None, :, :] * scale
                v = self.prompts[2*j+1][None, :, :] * scale
                # batch_size prefix_seq_len h
                k = k.repeat([batch_size, 1, 1]).view(batch_size, self.prefix_seq_len, self.match_n_head, self.match_n_embd)
                v = v.repeat([batch_size, 1, 1]).view(batch_size, self.prefix_seq_len, self.match_n_head, self.match_n_embd)
                k = k.permute([0, 2, 1, 3])
                v = v.permute([0, 2, 1, 3])

                past_key_values.append((k, v))

        all_kvs = ()
        for i in range(len(past_key_values)):
            # if i < 6:
            #     kvpair = None
            # else:
            kvpair = (past_key_values[i][0], past_key_values[i][1])
            
            all_kvs += (kvpair,)

        return all_kvs


class PrefixEncoder_MGA(nn.Module):
    def __init__(self, config, model_args):
        super().__init__()

        self.input_dim = config.zdim
        self.hidden_dim = config.hidden_size

        self.prefix_seq_len = config.task_ztokens if hasattr(config, "task_ztokens") else config.ztokens

        self.match_n_layer = model_args.shallow_decoder_n_layer

        # MGA
        self.match_n_head = config.num_key_value_heads
        self.match_n_embd = config.hidden_size // config.num_attention_heads

        h = self.match_n_head * self.match_n_embd
        self.prefix_mlp = nn.Linear(self.input_dim, self.match_n_layer * 2 * h)
 
    def forward(
        self,
        input_embd
    ):
        batch_size = input_embd.size(0)
        past_key_values = self.prefix_mlp(input_embd)
        past_key_values = past_key_values.view(
            batch_size, self.prefix_seq_len, self.match_n_layer, -1)
        # Resize
        past_key_values = past_key_values.view(
            batch_size,
            self.prefix_seq_len,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )

        # Transpose -> [match_n_layer*2, batch_size, match_n_head, prefix_seq_len, match_n_embd]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])
        past_key_values = torch.split(past_key_values, 2)

        all_kvs = ()
        for i in range(len(past_key_values)):
            kvpair = (past_key_values[i][0], past_key_values[i][1])
            all_kvs += (kvpair,)
        return all_kvs



class AE(PreTrainedModel):
    _supports_flash_attn_2 = True
    def __init__(self, config, model_args, lora_config=None):
        super().__init__(config)
        self.sae_kl_sparse = model_args.sae_kl_sparse

        self.use_ztokens = config.task_ztokens if hasattr(config, "task_ztokens") else config.ztokens
        zdim = config.zdim
        self.z_start_id = config.z_start_id
        self.z_end_id = self.z_start_id + self.use_ztokens

        if model_args.model_name_or_path_ae is not None:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path_ae)
            model_name_or_path = model_args.model_name_or_path_ae
            config.zdim = zdim
            config.ztokens = self.use_ztokens
        else:
            model_name_or_path = model_args.model_name_or_path

        self.zwte = nn.Embedding(self.use_ztokens, config.hidden_size)

        if model_args.from_scratch:
            self.encoder = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
            )

            dec_config = deepcopy(config)
            dec_config.num_hidden_layers = model_args.shallow_decoder_n_layer

            if model_args.aedec_load_pretrained:
                self.decoder = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    config=dec_config,)
                
                if model_args.shallow_decoder_n_layer < config.num_hidden_layers:
                    print("init decoder with skipping")
                    assert config.num_hidden_layers % model_args.shallow_decoder_n_layer == 0
                    skip_l = config.num_hidden_layers // model_args.shallow_decoder_n_layer 
                    # skip init
                    for il in range(0, model_args.shallow_decoder_n_layer):
                        state_dict = self.encoder.h[il*skip_l].state_dict()
                        self.decoder.transformer.h[il].load_state_dict(state_dict)
            else:
                self.decoder = AutoModelForCausalLM.from_config(dec_config)
        else:
            self.encoder = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
            )
            dec_config = deepcopy(config)
            
            dec_config.num_hidden_layers = model_args.shallow_decoder_n_layer

            self.decoder = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=dec_config,
            )

        if lora_config is not None:
            self.encoder = get_peft_model(self.encoder, lora_config)

        # v1
        # block_config = deepcopy(config)
        # block_config.scale_attn_by_inverse_layer_idx = False
        # block_config.add_cross_attention = True
        # block_config._attn_implementation = "eager"
        # block_config.reorder_and_upcast_attn = False
        # self.cross_block = GPT2Block(block_config)
        # self.lnz = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # v2: simple
        self.cross_block = GPT2Attention(config, is_cross_attention=True)

        self.ln_1 = get_ln(config)

        self.ln_2 = get_ln(config)
        self.ln_enc = get_ln(config)
        
        if hasattr(config, "num_key_value_heads") and config.num_key_value_heads < config.num_attention_heads:
            self.prefix_encoder = PrefixEncoder_MGA(config, model_args)
        else:
            self.prefix_encoder = PrefixEncoder_MHA(config, model_args)
        
        self.proj = None
        if config.hidden_size > zdim:
            self.proj = nn.Linear(config.hidden_size, zdim, bias=False)

        print("self.proj")
        print(self.proj)

        self.zorth_reg = model_args.zorth_reg

        self.use_sae = model_args.use_sae
        if self.use_sae:
            self.sae = SAE(config, model_args)

        self.z_from_layer = model_args.z_from_layer

        init_weight(self.zwte, config)
        init_weight(self.cross_block,config)
        init_weight(self.prefix_encoder,config)
    
        if self.proj is not None:
            init_weight(self.proj,config)

    def forward(
        self,
        input_ids_enc,
        attention_mask_enc=None,
        input_ids_dec=None,
        attention_mask_dec=None,
        labels=None,
    ):
        with torch.no_grad():
            enc_outs = self.encoder(
            input_ids=input_ids_enc,
            attention_mask=attention_mask_enc, 
            output_hidden_states=True
            )

        bz = input_ids_enc.size(0)
        z_idx = torch.arange(0, self.use_ztokens, dtype=torch.long, device=input_ids_enc.device)

        input_ids_enc_z = z_idx.unsqueeze(0).repeat(bz, 1)
        hidden_states_z = self.zwte(input_ids_enc_z)

        if attention_mask_enc is not None:
            # attention_mask_enc_4d = attention_mask_enc.view(input_ids_enc.size(0), -1)
            attention_mask_enc_4d = attention_mask_enc.view(bz, -1)
            attention_mask_enc_4d = attention_mask_enc_4d[:, None, None, :]
            attention_mask_enc_4d = attention_mask_enc_4d.to(dtype=self.dtype)
            attention_mask_enc_4d = (1.0 - attention_mask_enc_4d) * torch.finfo(self.dtype).min
        else:
            attention_mask_enc_4d = None
        
        enc_lhs = enc_outs.hidden_states[self.z_from_layer]

        '''
        hidden_z = self.cross_block(
            hidden_states_z,
            encoder_attention_mask = attention_mask_enc_4d,
            encoder_hidden_states = enc_lhs
        )[0]
        hidden_down = self.lnz(hidden_z)
        '''
        
        # residual = hidden_states_z
        hidden_states_z = self.ln_1(hidden_states_z)

        if self.z_from_layer != -1:
            enc_lhs = self.ln_enc(enc_lhs)
        
        cross_outs = self.cross_block(
            hidden_states = hidden_states_z,
            encoder_attention_mask=attention_mask_enc_4d,
            encoder_hidden_states = enc_lhs
        )
        # hidden_z = cross_outs[0] + residual
        hidden_z = cross_outs[0]

        zorth_loss = torch.tensor(0).to(hidden_z)
        if self.zorth_reg > 0:
            zorth_loss = orthogonal_loss_fn(hidden_z) * self.zorth_reg
            # logger.info(f"zorth_loss: {zorth_loss.item()}")

        if self.use_sae:
            hidden_z = self.ln_2(hidden_z)
            sae_outputs = self.sae(hidden_z, output_loss=(labels is not None))

            hidden_down, sae_loss, sae_rec_loss, sae_spar_loss, sae_pre_act, sae_sparse_act = (
                sae_outputs.reconstruction,
                sae_outputs.loss,
                sae_outputs.rec_loss,
                sae_outputs.sparsity_loss,
                sae_outputs.pre_activations,
                sae_outputs.activations
            )


            if self.sae_kl_sparse:
                sae_act = sae_sparse_act
            else:
                sae_act = sae_pre_act
            
            # check = sae_act > 0
            
            # for ib in range(sae_act.size(0)):
            #     print("smaple", ib)
            #     for i in range(sae_act.size(1)-1):
            #         notsame_element = torch.logical_xor(check[ib, i], check[ib, i+1])
            #         print(check[ib, i].sum(), check[ib, i+1].sum(), notsame_element.sum())
            
            # for iz in range(sae_act.size(1)):
            #     print("z", iz)
            #     for i in range(sae_act.size(0)-1):
            #         notsame_element = torch.logical_xor(check[i, iz], check[i+1, iz])
            #         print(check[i, iz].sum(), check[i+1, iz].sum(), notsame_element.sum())
            # exit()
            # for l in sae_act[0]:
                # print(torch.nonzero(l).view(-1))
                # print(torch.nonzero(l).view(-1).shape)
        else:
            hidden_down = self.ln_2(hidden_z)
            if self.proj is not None:
                hidden_down = self.proj(hidden_down)
            
            sae_loss = torch.tensor(0).to(hidden_z)
            sae_spar_loss = torch.tensor(0).to(hidden_z)
            sae_rec_loss = torch.tensor(0).to(hidden_z)
            sae_pre_act = None
            sae_sparse_act = None
            sae_act = None

        loss = None
        if labels is not None:
            past_key_values = self.prefix_encoder(hidden_down)

            from transformers.cache_utils import DynamicCache
            past_key_values = DynamicCache().from_legacy_cache(past_key_values)

            attention_mask_z = attention_mask_dec.new_ones(bz, hidden_down.size(1))

            dec_outs = self.decoder(
                input_ids=input_ids_dec,
                past_key_values=past_key_values,
                attention_mask=torch.cat(
                    (attention_mask_z, attention_mask_dec), dim=-1),
                output_hidden_states=False,
                output_attentions=False,
                labels=labels
            )
            dec_loss = dec_outs.loss
            loss = dec_loss + sae_loss + zorth_loss
        else:
            loss = torch.tensor(0).to(hidden_z)
            dec_loss = torch.tensor(0).to(hidden_z)
            sae_loss = torch.tensor(0).to(hidden_z)
            sae_spar_loss = torch.tensor(0).to(hidden_z)
            sae_rec_loss = torch.tensor(0).to(hidden_z)
        
        return AEOutput(
            loss=loss,
            aedec_loss = dec_loss,
            sae_spar_loss = sae_spar_loss,
            sae_rec_loss = sae_rec_loss,
            sae_act = sae_act,
            hidden_z=hidden_down
        )


class MLP(nn.Module):
    def __init__(self, isize, osize, intermediate_size, config):
        super().__init__()
        self.c_fc = nn.Linear(isize, intermediate_size, False)
        self.c_proj = nn.Linear(intermediate_size, osize, False)
        
        if hasattr(config, "activation_function"):
            activation_function = config.activation_function
        elif hasattr(config, "hidden_act"):
            activation_function = config.hidden_act

        self.act = ACT2FN[activation_function]

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class SemformerHF(PreTrainedModel):
    _supports_flash_attn_2 = True
    def __init__(self, config, model_args, lora_config=None):
        super().__init__(config)

        self.use_ztokens = config.task_ztokens if hasattr(config, "task_ztokens") else config.ztokens
        self.z_start_id = config.z_start_id
        self.z_end_id = self.z_start_id + self.use_ztokens

        self.alpha = model_args.alpha
        self.beta = model_args.beta

        self.model_args = model_args

        if model_args.regloss == "l2":
            self.regloss = nn.MSELoss()
        elif model_args.regloss == "l1":
            # self.regloss = nn.SmoothL1Loss()
            self.regloss = nn.L1Loss()
        elif model_args.regloss == "kl":
            self.regloss = nn.KLDivLoss(reduction="batchmean")
        else:
            exit("regloss")

        self.znorm = model_args.znorm

        self.model_parallel = False
        self.device_map = None
        
        if model_args.from_scratch:
            # max_position_embeddings
            dec_config = deepcopy(config)
            dec_config.max_position_embeddings *= 2
            self.main_decoder = AutoModelForCausalLM.from_config(dec_config)
        else:
            self.main_decoder = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                config=config,
            )
            dec_config = config
        
        self.proj = None

        if model_args.regloss == "l2" or model_args.regloss == "l1":
            # if dec_config.hidden_size > dec_config.zdim:
            if dec_config.predictor_nn == "mlp":
                # isize, osize, intermediate_size, config
                self.proj = MLP(dec_config.hidden_size, dec_config.zdim, dec_config.hidden_size*2, config)
            else:
                self.proj = nn.Linear(dec_config.hidden_size, dec_config.zdim, bias=False)
            # self.proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        else:
            if dec_config.predictor_nn == "mlp":
                # isize, osize, intermediate_size, config
                self.proj = MLP(dec_config.hidden_size, config.sae_activation_size, 
                                dec_config.hidden_size*2, config)
            else:
                self.proj = nn.Linear(dec_config.hidden_size, config.sae_activation_size, bias=False)
        
        print("proj in lm")
        print(self.proj)

        if self.beta > 0 or self.alpha > 0:
            self.aemodel = AE(deepcopy(config), model_args=model_args, lora_config=lora_config)
        
        # will revise config.vocab_size
        self.main_decoder.resize_token_embeddings(dec_config.len_tokenizer, pad_to_multiple_of=8)

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.norm_func = nn.LayerNorm(dec_config.zdim, elementwise_affine=False)

        init_weight(self.proj, config)

    def only_tunelayers(self):
        print(f"only_tunelayers")
        self.main_decoder.lm_head.requires_grad_(False)
        if hasattr(self, "aemodel"):
            self.aemodel.decoder.lm_head.requires_grad_(False)
            try:
                self.aemodel.decoder.model.embed_tokens.requires_grad_(False)
            except:
                self.aemodel.decoder.transformer.wte.requires_grad_(False)

    def share_emb_decoders(self):
        if self.beta > 0 or self.alpha > 0:
            print(f"share emb lmhead between decoders")
            self.aemodel.decoder.lm_head = self.main_decoder.lm_head
            try:
                self.aemodel.decoder.model.embed_tokens = self.main_decoder.model.embed_tokens
            except:
                self.aemodel.decoder.transformer.wte = self.main_decoder.transformer.wte

    def freeze_ae(self):
        if hasattr(self, "aemodel"):
            print(f"fix ae")
            self.aemodel.requires_grad_(False)

    def freeze_lm(self):
        print(f"fix lm")
        self.main_decoder.requires_grad_(False)

    def load_encoder_and_fix(self, model_name_or_path, config=None):
        if hasattr(self, "aemodel"):
            print(f"load pretrained and fix encoder", model_name_or_path)
            self.aemodel.encoder = AutoModel.from_pretrained(
                model_name_or_path,
                # config=config,
                )
            self.aemodel.encoder.requires_grad_(False)
    
    def tie_aeencoder_with_decoder(self):
        if hasattr(self, "aemodel"):
            print("/tie encoder/ with main decoder")
            self.aemodel.encoder = self.main_decoder

    def resize_token_embeddings(self, len_t):
        self.main_decoder.resize_token_embeddings(len_t, pad_to_multiple_of=8)

    def generate(self, *args, **kwargs):
        return self.main_decoder.generate(*args, **kwargs)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        input_ids_enc=None,
        input_ids_ae_dec=None,
        attention_mask_enc=None,
        attention_mask_ae_dec=None,
        labels_ae=None,
        **kwargs
    ):
        if self.beta > 0 and self.alpha == 0 and input_ids_enc is not None:
            # pretrain ae
            ae_outs = self.aemodel(
                input_ids_enc=input_ids_enc,
                attention_mask_enc=attention_mask_enc,
                input_ids_dec=input_ids_ae_dec,
                attention_mask_dec=attention_mask_ae_dec,
                labels=labels_ae
            )
            ae_totalloss = ae_outs.loss
            aedec_loss = ae_outs.aedec_loss
            sae_spar_loss = ae_outs.sae_spar_loss
            sae_rec_loss = ae_outs.sae_rec_loss

            return CausalLMOutput(
                loss=ae_totalloss,
                reppred_loss=torch.zeros_like(ae_totalloss),
                aedec_loss=aedec_loss,
                sae_spar_loss=sae_spar_loss,
                sae_rec_loss=sae_rec_loss
                )
            # return CausalLMOutput(
            #             loss=recloss,
            #             reppred_loss=torch.zeros_like(recloss),
            #             rec_loss=recloss,
            #             )
        
        bs = input_ids.size(0)
        
        main_dec_outs = self.main_decoder(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False
        )
        nllloss = main_dec_outs.loss

        if self.alpha > 0 and input_ids_enc is not None:
            main_dec_lhs = main_dec_outs.hidden_states[-1]  # bs seqlen h
            # 'CausalLMOutputWithCrossAttentions' object has no attribute 'last_hidden_state'

            is_ztokens = self.z_start_id <= input_ids

            main_hidden_z = main_dec_lhs[is_ztokens]

            if self.proj is not None:
                main_hidden_z = self.proj(main_hidden_z)

            main_hidden_z = main_hidden_z.view(bs, -1, main_hidden_z.size(-1))
            # print(main_hidden_z.shape)

            if self.beta == 0:
                ae_outs = self.aemodel(
                input_ids_enc=input_ids_enc,
                attention_mask_enc=attention_mask_enc,
                input_ids_dec=input_ids_ae_dec,
                attention_mask_dec=attention_mask_ae_dec,
                labels=None
                )
                ae_totalloss = torch.zeros_like(nllloss)
                aedec_loss = torch.zeros_like(nllloss)
                sae_spar_loss = torch.zeros_like(nllloss)
                sae_rec_loss = torch.zeros_like(nllloss)
            else:
                ae_outs = self.aemodel(
                input_ids_enc=input_ids_enc,
                attention_mask_enc=attention_mask_enc,
                input_ids_dec=input_ids_ae_dec,
                attention_mask_dec=attention_mask_ae_dec,
                labels=labels_ae
                )
                ae_totalloss = ae_outs.loss

                aedec_loss = ae_outs.aedec_loss
                sae_spar_loss = ae_outs.sae_spar_loss
                sae_rec_loss = ae_outs.sae_rec_loss
                
            # check
            # ae_outs = self.aemodel(
            #     input_ids_enc=input_ids_enc,
            #     input_ids_dec=input_ids_ae_dec,
            #     labels=labels_ae
            #     )
            # recloss = ae_outs.loss
            # print(recloss)

            if self.model_args.regloss != "kl":
                target_z = ae_outs.hidden_z.detach()
                target_z = target_z.reshape(bs, -1, target_z.size(-1))

                main_hidden_z = main_hidden_z[:, :target_z.size(1)]

                if self.znorm:
                    target_z = self.norm_func(target_z)

                reppred_loss = self.regloss(main_hidden_z, target_z)
            else:
                # torch.set_printoptions(profile="full")
                # B Z HZ
                sae_act = ae_outs.sae_act.detach()
                target_z_feature_distri = F.softmax(sae_act / self.model_args.sae_kl_temperature, dim=-1)
                main_z_feature_distri = F.log_softmax(main_hidden_z, dim=-1)
                # CrossEntropyLoss
                reppred_loss = self.regloss(main_z_feature_distri, target_z_feature_distri)

            tloss = self.alpha * reppred_loss + self.beta * ae_totalloss + nllloss
        else:
            tloss = nllloss
            reppred_loss = torch.zeros_like(tloss)
            ae_totalloss = torch.zeros_like(tloss)

            aedec_loss = torch.zeros_like(tloss)
            sae_spar_loss = torch.zeros_like(tloss)
            sae_rec_loss = torch.zeros_like(tloss)

        return CausalLMOutput(
            loss=tloss if self.training else nllloss,
            logits=main_dec_outs.logits,
            reppred_loss=reppred_loss * self.alpha,
            aedec_loss=ae_totalloss * self.beta,
            sae_spar_loss=sae_spar_loss * self.beta,
            sae_rec_loss=sae_rec_loss * self.beta
        )
