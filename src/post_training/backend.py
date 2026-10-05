"""Training backend dispatch."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from post_training.config import PostTrainingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class Backend(ABC):
    @abstractmethod
    def validate(self, config: PostTrainingConfig) -> None:
        """Backend-specific config validation."""

    @abstractmethod
    def generate_run_name(self, config: PostTrainingConfig, timestamp: str) -> str:
        """Return an auto-generated run name."""

    @abstractmethod
    def run_dir_subdirs(self) -> list[str]:
        """Extra subdirectories to create inside the run directory."""

    @abstractmethod
    def render_slurm_script(
        self,
        config: PostTrainingConfig,
        run_dir: Path,
        frozen_config_path: str,
        *,
        tokenize_only: bool = False,
    ) -> Path:
        """Render the SLURM batch script and return its path."""

    @abstractmethod
    def post_freeze(self, config: PostTrainingConfig, run_dir: Path) -> None:
        """Copy any extra artifacts into the run directory after config freeze."""


# ---------------------------------------------------------------------------
# TRL
# ---------------------------------------------------------------------------

_SUPPORTED_METHODS = ("sft", "dpo")


def _shorten_model_name(name_or_path: str) -> str:
    short = name_or_path.rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9\-]", "", short.lower())


def _shorten_dataset_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_\-]", "_", name.lower())


def _dataset_mix_hash(datasets: list) -> str:
    canonical = sorted([asdict(dataset) for dataset in datasets], key=lambda item: item["name"])
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:8]


class TRLBackend(Backend):
    def validate(self, config: PostTrainingConfig) -> None:
        if config.method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{config.method}'. "
                f"Supported methods: {', '.join(_SUPPORTED_METHODS)}"
            )

        if isinstance(config.deepspeed, dict) and "config_path" in config.deepspeed:
            raise ValueError(
                "deepspeed.config_path is no longer supported. Inline the DeepSpeed "
                "JSON/YAML directly under the `deepspeed:` key (see configs/trl/sft.yaml "
                "for an example), or set `deepspeed: null` to disable DeepSpeed."
            )

        # Container validation (only when container.image is set)
        if config.container is not None and config.container.image:
            if not config.container.bind_mounts:
                raise ValueError(
                    "container.bind_mounts must be non-empty when container.image is set."
                )
            if not config.container.env_file:
                raise ValueError("container.env_file must be set when container.image is set.")

        for field_name, entries in (
            ("data.datasets", config.data.datasets),
            ("data.eval_datasets", config.data.eval_datasets),
        ):
            total_weight = 0.0
            for entry in entries:
                if entry.weight < 0:
                    raise ValueError(
                        f"{field_name}[].weight must be non-negative, got {entry.weight} "
                        f"for dataset '{entry.name}'."
                    )
                total_weight += entry.weight
            if entries and total_weight <= 0:
                raise ValueError(f"Sum of {field_name}[].weight must be > 0.")

        t = config.training

        if t.effective_batch_size <= 0:
            raise ValueError("training.effective_batch_size must be positive.")
        if t.per_device_train_batch_size <= 0:
            raise ValueError("training.per_device_train_batch_size must be positive.")
        if config.evaluation.per_device_batch_size <= 0:
            raise ValueError("evaluation.per_device_batch_size must be positive.")

        if config.method == "dpo":
            if config.dpo.max_seq_length <= 0:
                raise ValueError("dpo.max_seq_length must be positive.")
            if config.dpo.truncated_pair_action not in {"warn", "drop"}:
                raise ValueError(
                    "dpo.truncated_pair_action must be either 'warn' or 'drop'."
                )
            if (
                config.dpo.precompute_ref_batch_size is not None
                and config.dpo.precompute_ref_batch_size <= 0
            ):
                raise ValueError("dpo.precompute_ref_batch_size must be positive when set.")

        # Training length: exactly one of max_steps / num_train_epochs /
        # num_training_samples / num_training_tokens must be specified.
        length_fields = {
            "training.max_steps": t.max_steps,
            "training.num_train_epochs": t.num_train_epochs,
            "training.num_training_samples": t.num_training_samples,
            "training.num_training_tokens": t.num_training_tokens,
        }
        specified = [name for name, value in length_fields.items() if value is not None]
        if len(specified) == 0:
            raise ValueError(
                "Exactly one of training.max_steps, training.num_train_epochs, "
                "training.num_training_samples, or training.num_training_tokens "
                "must be specified."
            )
        if len(specified) > 1:
            raise ValueError(
                "Training length is over-specified. Choose exactly one of "
                "training.max_steps, training.num_train_epochs, "
                "training.num_training_samples, or training.num_training_tokens. "
                f"You set: {', '.join(specified)}."
            )

        if t.max_steps is not None and t.max_steps <= 0:
            raise ValueError("training.max_steps must be a positive integer.")

        if t.num_train_epochs is not None and t.num_train_epochs <= 0:
            raise ValueError("training.num_train_epochs must be a positive number.")

        if t.num_training_tokens is not None:
            if config.method != "sft" or not config.sft.packing:
                raise ValueError(
                    "training.num_training_tokens is only valid for method='sft' "
                    "when sft.packing=true."
                )
            if t.num_training_tokens <= 0:
                raise ValueError("training.num_training_tokens must be a positive integer.")
            if config.sft.max_seq_length <= 0:
                raise ValueError("sft.max_seq_length must be positive.")

        if t.num_training_samples is not None:
            if t.num_training_samples <= 0:
                raise ValueError("training.num_training_samples must be a positive integer.")

    def generate_run_name(self, config: PostTrainingConfig, timestamp: str) -> str:
        model_short = _shorten_model_name(config.model.name_or_path)
        datasets = config.data.datasets
        if len(datasets) == 1:
            ds_part = _shorten_dataset_name(datasets[0].name)
        else:
            ds_part = f"mix_{_dataset_mix_hash(datasets)}"
        return f"{config.method}-{model_short}-{ds_part}-{timestamp}"

    def run_dir_subdirs(self) -> list[str]:
        return ["checkpoints", "inference_checkpoints"]

    def render_slurm_script(self, config, run_dir, frozen_config_path, *, tokenize_only=False):
        if config.container is not None and config.container.image:
            from post_training.slurm.launcher import render_trl_container_slurm_script

            return render_trl_container_slurm_script(
                config, run_dir, frozen_config_path, tokenize_only=tokenize_only
            )

        from post_training.slurm.launcher import render_trl_slurm_script

        return render_trl_slurm_script(
            config, run_dir, frozen_config_path, tokenize_only=tokenize_only
        )

    def post_freeze(self, config, run_dir):
        """Freeze src/post_training and scripts/ into run_dir for containerized runs.

        Done here (at submission time) rather than inside the SLURM job script,
        so a job that sits queued isn't affected by later repo changes. Each
        directory is copied into a temp path and atomically swapped into place,
        so an interrupted previous attempt can never leave a partial freeze that
        gets mistaken for a completed one.
        """
        from post_training.data.mix import write_mix_lock

        write_mix_lock(config, run_dir)

        if config.container is None or not config.container.image:
            return

        repo_dir = Path.cwd()
        self._freeze_dir(repo_dir / "src" / "post_training", run_dir / "src" / "post_training")
        self._freeze_dir(repo_dir / "scripts", run_dir / "scripts")

    @staticmethod
    def _freeze_dir(src: Path, dst: Path) -> None:
        if dst.exists():
            logger.warning("%s already exists — leaving frozen source in place.", dst)
            return

        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, tmp)
        os.replace(tmp, dst)


# ---------------------------------------------------------------------------
# LlamaFactory
# ---------------------------------------------------------------------------


class LlamaFactoryBackend(Backend):
    def validate(self, config: PostTrainingConfig) -> None:
        if not config.llamafactory:
            raise ValueError("llamafactory must be set when backend='llamafactory'.")
        if config.container is None or not config.container.image:
            raise ValueError("container.image must be set when backend='llamafactory'.")

    def generate_run_name(self, config: PostTrainingConfig, timestamp: str) -> str:
        return f"llamafactory-{config.method}-{timestamp}"

    def run_dir_subdirs(self) -> list[str]:
        return []

    def render_slurm_script(self, config, run_dir, frozen_config_path, *, tokenize_only=False):
        # LlamaFactory has no --tokenize-only equivalent; submit.py rejects it before this point.
        del tokenize_only
        from post_training.slurm.launcher import render_llamafactory_slurm_script

        return render_llamafactory_slurm_script(config, run_dir)

    def post_freeze(self, config, run_dir):
        from post_training.data.mix import write_mix_lock

        write_mix_lock(config, run_dir)

        if config.data.mix_id:
            registry: dict[str, dict] = {}
            names: list[str] = []
            dataset_dir = run_dir / "llamafactory_data"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            for entry in config.data.datasets:
                if entry.weight != 1.0:
                    raise ValueError(
                        "LlamaFactory mix manifests currently require weight=1.0. "
                        "Use LlamaFactory-native interleave settings for weighted mixes."
                    )
                metadata = dict(entry.llamafactory or {})
                resolved_path = entry.resolve_path(config.data.location)
                if entry.prefetched_files:
                    release_dir = dataset_dir / "releases" / entry.name
                    release_dir.mkdir(parents=True, exist_ok=True)
                    for index, source in enumerate(entry.prefetched_files):
                        source_path = Path(source).resolve()
                        if not source_path.is_file():
                            raise ValueError(
                                f"Prefetched file for dataset '{entry.name}' no longer exists: "
                                f"{source_path}"
                            )
                        link = release_dir / f"part-{index:05d}{source_path.suffix}"
                        if not link.exists():
                            link.symlink_to(source_path)
                    # A file entry takes precedence over a Hub entry in the
                    # generated registry and makes the selected revision exact.
                    metadata.pop("hf_hub_url", None)
                    metadata["file_name"] = str(release_dir.relative_to(dataset_dir))
                elif "file_name" not in metadata and "hf_hub_url" not in metadata:
                    if Path(resolved_path).exists() or resolved_path.startswith("/"):
                        metadata["file_name"] = resolved_path
                    else:
                        if entry.revision is not None:
                            raise ValueError(
                                f"LlamaFactory cannot pass revision={entry.revision!r} for Hub "
                                f"dataset '{entry.name}' through dataset_info.json. Keep "
                                "prefetch_assets=true or select a cluster-local release so the "
                                "pinned artifact is materialized before submission."
                            )
                        metadata["hf_hub_url"] = resolved_path
                elif entry.revision is not None and "hf_hub_url" in metadata:
                    raise ValueError(
                        f"LlamaFactory dataset_info.json has no revision field, so explicit "
                        f"hf_hub_url for pinned dataset '{entry.name}' is unsafe. Keep "
                        "prefetch_assets=true or use file_name for an immutable local release."
                    )
                if "hf_hub_url" in metadata:
                    if entry.subset is not None:
                        metadata.setdefault("subset", entry.subset)
                    if entry.data_dir is not None:
                        metadata.setdefault("folder", entry.data_dir)
                    metadata.setdefault("split", entry.split)
                registry[entry.name] = metadata
                names.append(entry.name)

            (dataset_dir / "dataset_info.json").write_text(
                json.dumps(registry, indent=2) + "\n",
                encoding="utf-8",
            )
            config.llamafactory = dict(config.llamafactory or {})
            config.llamafactory["dataset_dir"] = str(dataset_dir.resolve())
            config.llamafactory["dataset"] = ",".join(names)

        lf_config_path = run_dir / "llamafactory_config.yaml"
        with open(lf_config_path, "w") as f:
            yaml.dump(config.llamafactory, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, Backend] = {
    "trl": TRLBackend(),
    "llamafactory": LlamaFactoryBackend(),
}


def get_backend(name: str) -> Backend:
    """Return the backend implementation for *name*."""
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Supported: {', '.join(_BACKENDS)}")
    return _BACKENDS[name]
