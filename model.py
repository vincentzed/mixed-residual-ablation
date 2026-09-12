# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Mixed-residual ablation.
#
# Let s_i be the residual stream entering block i (s_0 is the embedding), so s_i is
# the previous layer's output and s_{i-1} is the output two layers back:
#
#   default : s_{i+1} = s_i                     + block_i(s_i)
#   mixed   : s_{i+1} = 0.5*s_i + 0.5*s_{i-1}   + block_i(s_i)   (i > 0)
#
# The two arms share this file entirely; `mixed_residual` is the only switch, so
# nothing else can differ between them.
#
# The block returns its *update* rather than x + update, which keeps the module
# signature identical to the stock TransformerBlock -- torch.compile, activation
# checkpointing and the sharding declarations all still wrap it unchanged.

from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock


class UpdateBlock(Llama3TransformerBlock):
    """A pre-norm block that returns its contribution instead of applying it.

    The residual *inside* the block is untouched -- the FFN still reads
    ``x + attn(...)``, exactly as the default block does. Only the connection
    between blocks is what the ablation changes.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3TransformerBlock.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a = self.attention(self.attention_norm(x), attention_masks, positions)
        h = x + a
        f = self.feed_forward(self.ffn_norm(h))
        return a + f


class MixedResidualModel(Llama3Model):
    """Llama3 with a switchable inter-block residual.

    ``mixed_residual=False`` reproduces the stock model exactly (x + update).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        mixed_residual: bool = False

        def update_from_config(self, *, config, **kwargs) -> None:
            from torchtitan.models.common.decoder import Decoder

            Decoder.Config.update_from_config(self, config=config, **kwargs)
            from torchtitan.models.llama3.sharding import set_llama3_sharding_config

            set_llama3_sharding_config(
                self, enable_sp=config.parallelism.enable_sequence_parallel
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.mixed_residual = config.mixed_residual

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ):
        x = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        x_two_back = None
        for i, layer in enumerate(self.layers.values()):
            update = layer(x, attention_masks, positions)
            if self.mixed_residual and i > 0:
                assert x_two_back is not None
                residual = 0.5 * x + 0.5 * x_two_back
            else:
                residual = x
            x_two_back, x = x, residual + update

        x = self.norm(x) if self.norm is not None else x
        if self._skip_lm_head:
            return x
        return self.lm_head(x) if self.lm_head is not None else x


__all__ = ["UpdateBlock", "MixedResidualModel"]
