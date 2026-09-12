# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer configs for the mixed-residual ablation.

Both arms are byte-identical except for ``mixed_residual``: same data order (same
seed), same token budget, same schedule. Each arm is swept over the six learning
rates {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2} and scored by its best result.
"""

import glob

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import CompileConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import TextProcessor
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry

_TOKENIZER = "./assets_gpt2/gpt2"
_VOCAB = 50257
_C4_DIR = "/home/brayden/c4_local/en"
_SEQ = 256
_TOKENS_PER_STEP = 256 * _SEQ  # 256 sequences
_STEPS = 1200  # 78.6M tokens per run


def _local_c4(pattern: str) -> SingleDatasetConfig:
    files = sorted(glob.glob(f"{_C4_DIR}/{pattern}"))
    if not files:
        raise FileNotFoundError(f"no C4 shards matching {_C4_DIR}/{pattern}")
    return SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={"data_files": files},
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    )


def _arm(flavor: str, lr: float) -> Trainer.Config:
    model_spec = model_registry(flavor, seq_len=_SEQ, vocab_size=_VOCAB)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path=_TOKENIZER,
        model_spec=model_spec,
        compile=CompileConfig(enable=True),
        optimizer=default_adamw(lr=lr),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=100, decay_ratio=0.9, decay_type="cosine", min_lr_factor=0.0
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=_TOKENS_PER_STEP,
            num_tokens_per_train_step=_TOKENS_PER_STEP,
            max_context_length=_SEQ,
            steps=_STEPS,
            max_norm=1.0,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=_local_c4("c4-train.*.json.gz")
            ),
        ),
        metrics=MetricsProcessor.Config(log_freq=100),
        checkpoint=CheckpointManager.Config(
            interval=1000000, last_save_model_only=True
        ),
        activation_checkpoint=None,
        validator=Validator.Config(
            enable=True,  # defaults to False -- without this validation silently never runs
            freq=_STEPS,  # once, at the end
            steps=50,
            dataloader=GrainDataLoader.Config(
                dataset=ConcatThenSplitPackingConfig(
                    dataset=_local_c4("c4-validation.*.json.gz")
                ),
                shuffle=False,
            ),
        ),
    )


# lr is overridden on the CLI; these exist so --config has something to name.
def mixed_residual_default(seq_len: int | None = _SEQ) -> Trainer.Config:
    return _arm("default", 1e-3)


def mixed_residual_mixed(seq_len: int | None = _SEQ) -> Trainer.Config:
    return _arm("mixed", 1e-3)
