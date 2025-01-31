from typing import Callable
import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, Bool, Key
import math


from dataclasses import dataclass
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import jax.experimental


@dataclass
class GPTConfig:
    vocab_size=50258
    max_position_embeddings=200
    hidden_size=1024
    num_hidden_layers=12
    num_attention_heads=16
    n_inner=4096
    activation_function="gelu_new"
    resid_pdrop=0.1
    embd_pdrop=0.1
    attn_pdrop=0.1
    layer_norm_epsilon=1e-5
    initializer_range=0.02
    summary_type="cls_index"
    summary_use_proj=True
    summary_activation=None
    summary_proj_to_labels=True
    summary_first_dropout=0.1
    scale_attn_weights=True
    use_cache=True
    bos_token_id=50256
    eos_token_id=50256
    scale_attn_by_inverse_layer_idx=False
    reorder_and_upcast_attn=False

def glu_new(x: Float[Array, "_"]) -> Float[Array, "_"]:
    return jax.numpy.array(
        0.5
        * x
        * (
            1
            + jax.numpy.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * jax.numpy.pow(x, 3.0))
            )
        )
    )


class our_Conv1D(eqx.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    nf: int
    nx: int
    weight: jax.Array
    bias: jax.Array

    def __init__(self, nf, nx, key=None):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = jax.nn.initializers.normal(stddev=0.02)(key, (nx, nf))
        self.bias = jax.numpy.zeros((nf))

    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def __call__(self, x):
        size_out = x.shape[:-1] + (self.nf,)
        x = self.bias + jax.numpy.dot(
            jax.numpy.reshape(x, shape=(-1, x.shape[-1])), self.weight
        )
        return jax.numpy.reshape(x, size_out)

class MLP(eqx.Module):
    c_fc: our_Conv1D
    c_proj: our_Conv1D
    dropout: nn.Dropout

    def __init__(self, intermediate_size, config, key):
        key1, key2 = jr.split(key, 2)

        # The weights are transposed compraed to the feed forward.
        embed_dim = config.hidden_size
        self.c_fc = our_Conv1D(intermediate_size, embed_dim, key=key1)
        self.c_proj = our_Conv1D(embed_dim, intermediate_size, key=key2)
        self.dropout = nn.Dropout(config.resid_pdrop, deterministic=True)

    # TODO: Interesting take on the fact that vmap should be applied here ?
    def __call__(self, x: Float[Array, "data"], dropout_key: Key = None):
        y = self.c_fc(x)
        y = glu_new(y)
        y = self.c_proj(y)
        y = self.dropout(y, key=dropout_key)
        return y


class CausalSelfAttention(eqx.Module):
    c_attn: our_Conv1D
    c_proj: our_Conv1D

    resid_dropout: nn.Dropout
    attn_dropout: nn.Dropout

    bias: jax.Array = eqx.field(static=True)
    scale_attn_weights: bool
    split_size: int

    num_heads: int
    head_size: int

    def __init__(self, config, key):
        k1, k2 = jr.split(key)

        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = hidden_size // config.num_attention_heads
        self.split_size = hidden_size

        self.c_attn = our_Conv1D(3 * hidden_size, hidden_size, key=k1)
        self.c_proj = our_Conv1D(hidden_size, hidden_size, key=k2)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.bias = jnp.tril(
            jnp.ones(
                (1, config.max_position_embeddings,  config.max_position_embeddings)
            )
        )

        self.scale_attn_weights = config.scale_attn_weights

    # Could play arround with the different attention score calculations (Baidhu ?)
    # X is an embedding, it should self attend.

    def _attn(self, q, k, v, attention_mask, head_mask, dropout_key):

        att = jnp.matmul(q, jnp.transpose(k, axes=(0, 2, 1)), precision="highest")
        att = att / math.sqrt(jnp.shape(k)[-2])  # Scale weights is set to true in XTTS.

        query_length, key_length = q.shape[-2], k.shape[-2]
        mask = self.bias[:, key_length - query_length : key_length, :key_length]
        att = jnp.where(
            jax.numpy.equal(jax.lax.stop_gradient(mask), 0),
            jnp.finfo(att.dtype).min,
            att,
        )

        if attention_mask is not None:
            att = att + attention_mask

        att = jax.nn.softmax(att, axis=-1)

        att = self.attn_dropout(att, key=dropout_key)

        if head_mask is not None:
            att = att * head_mask

        return jnp.matmul(att, v, precision="highest"), att

    # Stange that they do it this way and not by simply defining the dims without permutation
    def _split_heads(self, x):
        new_shape = x.shape[:-1] + (self.num_heads, self.head_size)
        x = jax.numpy.reshape(x, new_shape)
        return jax.numpy.permute_dims(x, (1, 0, 2))

    def _merge_heads(self, x):
        x = jax.numpy.permute_dims(x, (1, 0, 2))
        new_shape = x.shape[:-2] + (self.num_heads * self.head_size,)
        return jax.numpy.reshape(x, new_shape)

    def __call__(
        self,
        hidden_states: Float[Array, "layer values"],
        layer_past: Float[Array, "layer values"] = None,
        attention_mask: Float[Array, "_"] = None,
        head_mask: Float[Array, "_"] = None,
        encoder_hidden_states: Float[Array, "_"] = None,
        encoder_attention_mask: Float[Array, "_"] = None,
        use_cache: Bool = False,
        output_attentions: Bool = False,
        dropout_key: Key = None
    ):
        dropout_key1, dropout_key2 = jr.split(dropout_key)
        # print(f"Hidden states shape: {hidden_states.shape}")
        qkv = self.c_attn(hidden_states)
        # print(f"Hidden att shape: {qkv.shape}")

        q, k, v = jax.numpy.split(qkv, 3, axis=1)

        query = self._split_heads(q)
        key = self._split_heads(k)
        value = self._split_heads(v)

        # print(f"Shape of split heads: {query.shape}")

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = jax.numpy.concat((past_key, key), axis=-1)
            value = jax.numpy.concat((past_value, value), axis=-1)

        present = None
        if use_cache is True:
            present = (key, value)

        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask, dropout_key1
        )
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output, key=dropout_key2)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    
class Block(eqx.Module):
    ln_1: nn.LayerNorm
    ln_2: nn.LayerNorm
    attn: CausalSelfAttention
    mlp: MLP

    def __init__(self, config, key):
        key1, key2 = jr.split(key, 2)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(
            (hidden_size),
            eps=config.layer_norm_epsilon,
            elementwise_affine=True,
        )
        self.attn = CausalSelfAttention(config, key=key1)
        self.ln_2 = nn.LayerNorm(
            (hidden_size),
            eps=config.layer_norm_epsilon,
            elementwise_affine=True,
        )

        self.mlp = MLP(inner_dim, config, key=key2)

    def __call__(
        self,
        hidden_states: Float[Array, "layer values"],
        layer_past: Float[Array, "layer values"] = None,
        attention_mask: Float[Array, "_"] = None,
        head_mask: Float[Array, "_"] = None,
        encoder_hidden_states: Float[Array, "_"] = None,
        encoder_attention_mask: Float[Array, "_"] = None,
        use_cache: Bool = False,
        output_attentions: Bool = False,
        dropout_key: Key = None
    ):
        dropout_key1, dropout_key2 = jr.split(dropout_key)
        residual = hidden_states
        hidden_states = jax.vmap(self.ln_1)(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            dropout_key=dropout_key1
        )  # Can't vmap as the whole point is exchange info between tokens.
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = jax.vmap(self.ln_2)(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states, dropout_key2)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            return (hidden_states,) + outputs
        else:
            return (hidden_states,) + outputs[1:]


class TransformerLayer(eqx.Module):
    wte: nn.Embedding  # Token embeddings
    wpe: nn.Embedding  # Positional embeddings

    drop: nn.Dropout

    embed_dim: int

    h: list
    norm: nn.LayerNorm

    def __init__(self, config, key):
        key1, key2 = jr.split(key, 2)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim, key=key1)
        self.wpe = nn.Embedding(
            config.max_position_embeddings, self.embed_dim, key=key2
        )
        self.drop = nn.Dropout(config.embd_pdrop)

        self.h = [
            Block(config, y) for y in jr.split(key, config.num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def __call__(
        self,
        input_ids: Int[Array, "bla"] = None,  # One ID inputted ?
        past_key_values: Int[Array, "bla"] = None,  # Used !
        attention_mask: Int[Array, "bla"] = None,  # Used !
        token_type_ids: Int[Array, "bla"] = None,  # Not used
        position_ids: Int[Array, "bla"] = None,  # Used !
        head_mask: Int[Array, "bla"] = None,  # Isn't used
        inputs_embeds: Float[Array, "bla"] = None,  # Isn't used
        output_attentions: Float[Array, "bla"] = None,  # Isn't used
        output_hidden_states: Bool = None,  # Isn't used
        use_cache: Bool = False,  # Set to true.
        return_dict: Bool = False,  # Set to true.
        dropout_key: Key = None
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Should use better positional embeddings with cos and sin.
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0].shape[-1]
        if position_ids is None:
            position_ids = jax.numpy.arange(past_length, input_shape[-1] + past_length)

        if inputs_embeds is None:
            inputs_embeds = jax.vmap(self.wte)(input_ids)

        # pos = jnp.arange(0, t, dtype=jnp.int64)

        position_embeds = jax.vmap(self.wpe)(position_ids)

        # Dropout at the first layer ? Seems a bit aggressive...
        hidden_states = inputs_embeds + position_embeds
        # No need for fancy stuff for the attention mask, simply since it's applied before the softmax change the values of 0 to -inf
        if attention_mask is not None:
            attention_mask = jax.numpy.where(
                jax.numpy.equal(attention_mask, 1), 1, -jax.numpy.inf
            )
        # No cross attention so we're all good.
        # No head mask.
        # Token type ids is none
        dropout_key1, dropout_key2 = jr.split(dropout_key)
        hidden_states = self.drop(hidden_states, key=dropout_key1)
        # Not training.
        presents = () if use_cache else None
        # Output attentions not used
        # no cross attention_mask
        # No output hidden states

        for block, layer_past in zip(self.h, past_key_values):
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                dropout_key=dropout_key2,
            )
            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)

        hidden_states = jax.vmap(self.norm)(hidden_states)

        if return_dict:
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                attentions=None,
                cross_attentions=None,
                hidden_states=None,
            )

        return hidden_states

class GPT(eqx.Module):
    transformer: TransformerLayer
    lm_head: nn.Linear

    def __init__(self, config, key):
        k1, k2 = jr.split(key, 2)

        self.transformer = TransformerLayer(config, k1)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, use_bias=False, key=k2
        )

    def __call__(self, token_ids: Int[Array, "_"], attention_mask: Int[Array, "_"], dropout_key: Key = None, past_key_values: Int[Array, "_"]=None,return_dict: Bool = False):
        y = self.transformer(token_ids, past_key_values=past_key_values, attention_mask=attention_mask, dropout_key=dropout_key, return_dict= return_dict)
        if return_dict:
            return jax.vmap(self.lm_head)(y.last_hidden_state), y
        else:
            return jax.vmap(self.lm_head)(y)

    @staticmethod
    def _init_weights(model: eqx.Module, config: GPTConfig, key=None):
        def init_layer(model, is_layer: Callable, mean: float, std: float):
            def get_weights(m):
                return [x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_layer) if is_layer(x)]
            weights = get_weights(model)

            new_weights = [
                (
                    jr.normal(k, weight.shape) * std + mean
                    if not isinstance(
                        weight, nn._shared.SharedNode
                    )  # SharedNode is a place holder value as we only have one matrix not two.
                    else weight
                )
                for weight, k in zip(weights, jr.split(key, len(weights)))
            ]

            return eqx.tree_at(get_weights, model, new_weights)

        def init_linear(model):
            def is_linear(x):
                return isinstance(x, eqx.nn.Linear)

            model = init_layer(model, is_linear, mean=0.0, std=0.2)

            def get_biases(m):
                return [x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x) and x.bias is not None]

            biases = get_biases(model)

            new_biases = [jnp.zeros_like(bias) for bias in biases]

            return eqx.tree_at(get_biases, model, new_biases)

        def init_embedding(model):
            def is_embedding(x):
                return isinstance(x, eqx.nn.Embedding)

            return init_layer(model, is_embedding, mean=0.0, std=0.2)

        def init_c_proj_weights_with_normal(model, key):

            def hop(path, x):
                nonlocal key
                if "c_proj.weight" in jax.tree_util.keystr(path):
                    key, k = jr.split(key)
                    return jr.normal(k, x.shape) * 0.02
                return x

            return jax.tree_util.tree_map_with_path(hop, model)

        model = init_linear(model)
        model = init_embedding(model)
        # apply special scaled init to the residual projections, per GPT-2 paper
        model = init_c_proj_weights_with_normal(model, key)

        return model

    @staticmethod
    def create_instance(config, key):
        key1, key2 = jr.split(key, 2)

        inst = GPT(config, key1)
        new_inst = GPT._init_weights(inst, config, key2)

        return new_inst

    def generate(self, idx: Int[Array, "_"], max_new_tokens, temperature=1.0, top_k=None, key=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.shape[1] <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits = jax.vmap(self, in_axes=(0, None))(idx_cond, False)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = jax.lax.top_k(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            key, k = jr.split(key)
            idx_next = jr.categorical(k, logits)
            # idx_next = jax.numpy.argmax(logits, axis=-1)

            # append sampled index to the running sequence and continue
            idx = jnp.concat((idx, idx_next), axis=-1)

        return idx