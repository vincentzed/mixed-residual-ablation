# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Mixed-residual ablation: default 8-layer transformer vs. a variant whose
# inter-block residual is 0.5*(previous layer) + 0.5*(two layers before).
#
# Geometry is the CS336 assignment-1 default (d_model 512, d_ff 1344 = 8/3*d_model
# rounded to a multiple of 64, 16 heads, RMSNorm pre-norm, SwiGLU, RoPE, no biases,
# untied lm_head) with num_layers raised to 8.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import ComplexRoPE, Embedding, Linear, RMSNorm
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.protocols.model_spec import ModelSpec

from .model import MixedResidualModel, UpdateBlock

__all__ = ["MixedResidualModel", "model_registry"]

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}


def _output_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _make_config(
    *,
    dim: int = 512,
    n_layers: int = 8,
    n_heads: int = 16,
    d_ff: int = 1344,
    vocab_size: int = 50257,
    seq_len: int = 256,
    rope_theta: float = 10000.0,
    mixed_residual: bool = False,
    attn_backend: str = "flex",
) -> MixedResidualModel.Config:
    head_dim = dim // n_heads
    inner_attention = get_attention_config(attn_backend)
    layers = [
        UpdateBlock.Config(
            attention_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
            attention=make_gqa_config(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=None,
                wqkv_param_init=_LINEAR_INIT,
                wo_param_init=_depth_init(layer_id),
                inner_attention=inner_attention,
                fuse_qkv=True,
                rope=ComplexRoPE.Config(
                    dim=head_dim,
                    max_context_length=seq_len,
                    theta=rope_theta,
                    scaling="none",
                ),
            ),
            feed_forward=make_ffn_config(
                dim=dim,
                hidden_dim=d_ff,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            ),
        )
        for layer_id in range(n_layers)
    ]
    return MixedResidualModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        mixed_residual=mixed_residual,
        enable_weight_tying=False,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim, out_features=vocab_size, param_init=_output_init(dim)
        ),
        layers=layers,
    )


mixed_residual_configs = {
    "default": (partial(_make_config, mixed_residual=False), 256),
    "mixed": (partial(_make_config, mixed_residual=True), 256),
}


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    vocab_size: int = 50257,
    **kwargs,
) -> ModelSpec:
    get_config, max_context_len = mixed_residual_configs[flavor]
    context_len = seq_len or max_context_len
    return ModelSpec(
        name="mixed_residual",
        flavor=flavor,
        model=get_config(
            seq_len=context_len, attn_backend=attn_backend, vocab_size=vocab_size
        ),
        max_context_length=context_len,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
