from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.cross_attention import CrossAttention
from diffusers.utils import BaseOutput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            Hidden states conditioned on `encoder_hidden_states` input. If discrete, returns probability distributions
            for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


class Transformer2DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        cross_frame_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        
        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    cross_frame_attention_dim=cross_frame_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
                for d in range(num_layers)
            ]
        )
        # Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(inner_dim, in_channels)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(
        self,
        hidden_states,
        cross_frame_attn=None,
        image_hidden_states=None,
        encoder_hidden_states=None,
        timestep=None,
        cross_attention_kwargs=None,
        return_dict: bool = True,
    ):
        
        # 1. Input
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
    
        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                cross_frame_attn=cross_frame_attn,
                image_hidden_states=image_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 3. Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.
    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        cross_frame_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # Transformer Decoder 1: self-attn, text-cross-attn, feed-forward;
        # 1. Self-Attn
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        
        # 2. Text-Cross-Attn
        if cross_attention_dim is not None:
            # self.attn2 = CrossAttention(
            #     query_dim=dim,
            #     cross_attention_dim=cross_attention_dim, # 768
            #     heads=num_attention_heads,
            #     dim_head=attention_head_dim,
            #     dropout=dropout,
            #     bias=attention_bias,
            #     upcast_attention=upcast_attention,
            # )  # is self-attn if encoder_hidden_states is none
            self.attn2 = StyleTransferAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                added_kv_proj_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None
            
        if cross_attention_dim is not None:
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
        else:
            self.norm2 = None

        # 3. Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        
        # Transformer Decoder 2: self-attn, cross-frame-attn, feed-forward;
        # Context Module in our StoryGen paper
        # 1. Self-Attn
        self.attn1_cross = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_frame_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        
        if self.use_ada_layer_norm:
            self.norm1_cross = AdaLayerNorm(dim, num_embeds_ada_norm)
        else:
            self.norm1_cross = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        
        # 2. Cross-Frame-Attn
        if cross_frame_attention_dim is not None:
            self.attn2_cross = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_frame_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2_cross = None
            
        if cross_frame_attention_dim is not None:
            self.norm2_cross = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
        else:
            self.norm2_cross = None
            
        # 3. Feed-forward
        self.ff_cross = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)
        self.norm3_cross = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        
    def forward(
        self,
        hidden_states,
        cross_frame_attn=False, # True for activating Cross Frame Attention
        image_hidden_states=None,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ):
        
        # Transformer Decoder 1: self-attn, text-cross-attn, feed-forward;
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
                
        hidden_states = attn_output + hidden_states
        
        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        
        # Transformer Decoder 2: self-attn, cross-frame-attn, feed-forward;
        if cross_frame_attn:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1_cross(hidden_states, timestep)
            else:
                norm_hidden_states = self.norm1_cross(hidden_states)

            # 1. Self-Attention
            attn_output = self.attn1_cross(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            
            hidden_states = attn_output + hidden_states
        
            norm_hidden_states = (
                self.norm2_cross(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2_cross(hidden_states)
            )

            # 2. Cross-Frame-Attention
            attn_output = self.attn2_cross(
                hidden_states = norm_hidden_states,
                encoder_hidden_states=image_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            
            hidden_states = attn_output + hidden_states
            # 3. Feed-forward
            norm_hidden_states = self.norm3_cross(hidden_states)
            ff_output = self.ff_cross(norm_hidden_states)

            hidden_states = ff_output + hidden_states
            
        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.
    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.
    """
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.
    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class AdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.
    """
    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x


class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=128):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)
         
      
class StyleTransferAttention(nn.Module):
    def __init__(self, query_dim: int, 
                 cross_attention_dim: int = 768, 
                 heads: int = 8, 
                 dim_head: int = 64, 
                 dropout: float = 0, 
                 bias=False, 
                 upcast_attention: bool = False, 
                 added_kv_proj_dim: int = 768):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.scale = dim_head**-0.5
        self.heads = heads
        self.sliceable_head_dim = heads
        self.added_kv_proj_dim = added_kv_proj_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_q_proj = LoRALinearLayer(query_dim, inner_dim)
            self.add_k_proj = LoRALinearLayer(cross_attention_dim, inner_dim)
            self.add_v_proj = LoRALinearLayer(cross_attention_dim, inner_dim)
            self.add_out_proj = LoRALinearLayer(inner_dim, query_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))
    
    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    
    def head_to_batch_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(self, attention_mask, target_length, batch_size=None):
        if batch_size is None:
            batch_size = 1

        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        if attention_mask.shape[-1] != target_length:
            if attention_mask.device.type == "mps":
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if attention_mask.shape[0] < batch_size * head_size:
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        return attention_mask

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        query = self.to_q(hidden_states)
        lora_query = self.add_q_proj(hidden_states)
        
        query = query + lora_query
        query = self.head_to_batch_dim(query)

        key = self.to_k(encoder_hidden_states)        
        value = self.to_v(encoder_hidden_states)
        lora_key = self.add_k_proj(encoder_hidden_states)
        lora_value = self.add_v_proj(encoder_hidden_states)
        
        key = key + lora_key
        value = value + lora_value
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        attention_probs = self.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)

        hidden_states = self.to_out[0](hidden_states) + self.add_out_proj(hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states