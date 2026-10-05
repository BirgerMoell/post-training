"""Direct preference optimisation (DPO) method."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from accelerate import PartialState
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from post_training.data.loader import load_and_mix_datasets
from post_training.methods.common import (
    align_generation_eos,
    build_callbacks,
    build_common_training_kwargs,
    build_model_init_kwargs,
    build_tokenizer,
    prioritize_metric_callbacks,
    sanitize_generation_config,
)

if TYPE_CHECKING:
    from post_training.config import PostTrainingConfig

logger = logging.getLogger(__name__)


def _has_preference_content(value: Any) -> bool:
    """Return whether a completion contains text or another meaningful payload."""
    if isinstance(value, str):
        return bool(value.strip())
    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict):
            if item:
                return True
            continue
        if any(item.get(key) for key in ("content", "tool_calls", "function_call")):
            return True
    return False


def _render_pair_side(row: dict[str, Any], side: str, tokenizer: Any) -> list[int]:
    """Render one preference branch exactly as a conversational DPO sample."""
    prompt = row.get("prompt")
    completion = row[side]
    if isinstance(prompt, list) and isinstance(completion, list):
        rendered = tokenizer.apply_chat_template(
            [*prompt, *completion], tokenize=True, add_generation_prompt=False
        )
    elif prompt is None and isinstance(completion, list):
        rendered = tokenizer.apply_chat_template(
            completion, tokenize=True, add_generation_prompt=False
        )
    else:
        text = f"{prompt or ''}{completion}"
        rendered = tokenizer(text, add_special_tokens=False)

    # Transformers 4.57 returns a BatchEncoding from apply_chat_template,
    # while newer releases may return the input-id list directly.
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return rendered


def _filter_dpo_rows(
    ds: Dataset,
    num_proc: int,
    *,
    tokenizer: Any | None = None,
    max_length: int | None = None,
    truncated_pair_action: str = "warn",
) -> Dataset:
    """Drop empty pairs and, when requested, pairs that would be truncated."""
    filtered = ds.filter(
        lambda row: _has_preference_content(row["chosen"])
        and _has_preference_content(row["rejected"]),
        num_proc=num_proc,
        desc="filtering empty preference pairs",
    )
    if truncated_pair_action != "drop":
        return filtered
    if tokenizer is None or max_length is None:
        raise ValueError("Dropping truncated DPO pairs requires a tokenizer and max_length.")

    return filtered.filter(
        lambda row: len(_render_pair_side(row, "chosen", tokenizer)) <= max_length
        and len(_render_pair_side(row, "rejected", tokenizer)) <= max_length,
        num_proc=num_proc,
        desc=f"dropping preference pairs longer than {max_length} tokens",
    )


def build_dpo_trainer(config: PostTrainingConfig, run_dir: Path) -> DPOTrainer:
    """Build a TRL :class:`DPOTrainer` from *config*.

    Parameters
    ----------
    config:
        Fully resolved post-training configuration.
    run_dir:
        Run output directory (already created).

    Returns
    -------
    DPOTrainer
        Ready to call ``.train()``.
    """
    mc = config.dpo  # method-specific config

    tokenizer = build_tokenizer(config)
    row_filter = partial(
        _filter_dpo_rows,
        tokenizer=tokenizer,
        max_length=mc.max_seq_length,
        truncated_pair_action=mc.truncated_pair_action,
    )
    with PartialState().main_process_first():
        dataset = load_and_mix_datasets(config.data, dataset_filter_fn=row_filter)
        eval_dataset = None
        if config.data.eval_datasets:
            eval_data_config = dataclasses.replace(
                config.data,
                datasets=config.data.eval_datasets,
                eval_datasets=[],
                mix_manifest=None,
                mix_id=None,
                mix_digest=None,
            )
            eval_dataset = load_and_mix_datasets(
                eval_data_config,
                dataset_filter_fn=row_filter,
            )

    dpo_config = DPOConfig(
        **build_common_training_kwargs(config, run_dir),
        beta=mc.beta,
        loss_type=mc.loss_type,
        max_length=mc.max_seq_length,
        dataset_num_proc=mc.dataset_num_proc,
        precompute_ref_log_probs=mc.precompute_ref_log_probs,
        precompute_ref_batch_size=mc.precompute_ref_batch_size,
        model_init_kwargs=build_model_init_kwargs(config),
    )

    trainer = DPOTrainer(
        model=config.model.name_or_path,
        ref_model=mc.ref_model_name_or_path,  # None → TRL creates implicit copy
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=dpo_config,
        callbacks=build_callbacks(config, run_dir),
    )
    sanitize_generation_config(trainer)
    align_generation_eos(trainer)
    prioritize_metric_callbacks(trainer)
    return trainer
