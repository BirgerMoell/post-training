"""Tests for nullable config fields."""

import pytest
import yaml

from post_training.config import ModelConfig, PostTrainingConfig
from post_training.methods.common import build_common_training_kwargs


def test_nullable_container_and_training_kwargs_load(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "backend": "trl",
                "container": None,
                "training": {
                    "max_steps": 1,
                    "effective_batch_size": 1,
                    "per_device_train_batch_size": 1,
                    "lr_scheduler_kwargs": None,
                    "gradient_checkpointing_kwargs": None,
                },
                "deepspeed": None,
                "data": {
                    "datasets": [
                        {
                            "name": "dummy",
                            "path": "dummy/path",
                            "weight": 1.0,
                        }
                    ]
                },
            }
        )
    )

    config = PostTrainingConfig.load(config_path)
    kwargs = build_common_training_kwargs(config, tmp_path)

    assert config.container is None
    assert kwargs["lr_scheduler_kwargs"] is None
    assert kwargs["gradient_checkpointing_kwargs"] is None
    assert kwargs["deepspeed"] is None


def test_deepspeed_empty_dict_normalized_to_none(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "backend": "trl",
                "training": {
                    "max_steps": 1,
                    "effective_batch_size": 1,
                    "per_device_train_batch_size": 1,
                },
                "deepspeed": {},
                "data": {
                    "datasets": [
                        {
                            "name": "dummy",
                            "path": "dummy/path",
                            "weight": 1.0,
                        }
                    ]
                },
            }
        )
    )

    config = PostTrainingConfig.load(config_path)
    kwargs = build_common_training_kwargs(config, tmp_path)

    assert kwargs["deepspeed"] is None


def test_fractional_warmup_steps_are_forwarded_as_ratio(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    config = PostTrainingConfig()
    config.training.max_steps = 100
    config.training.effective_batch_size = 1
    config.training.per_device_train_batch_size = 1
    config.training.warmup_steps = 0.03

    kwargs = build_common_training_kwargs(config, tmp_path)

    assert kwargs["warmup_ratio"] == 0.03
    assert kwargs["warmup_steps"] == 0


def test_absolute_warmup_steps_remain_steps(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    config = PostTrainingConfig()
    config.training.max_steps = 100
    config.training.effective_batch_size = 1
    config.training.per_device_train_batch_size = 1
    config.training.warmup_steps = 3

    kwargs = build_common_training_kwargs(config, tmp_path)

    assert "warmup_ratio" not in kwargs
    assert kwargs["warmup_steps"] == 3


@pytest.mark.parametrize(
    ("budget_field", "budget_value"),
    [
        ("num_training_samples", 33),
        ("num_training_tokens", 2_097_152),
    ],
)
def test_derived_step_budget_round_trips_without_mutation(
    tmp_path, monkeypatch, budget_field, budget_value
):
    """A frozen sample/token budget remains valid when train.py reloads it."""
    monkeypatch.setenv("WORLD_SIZE", "1")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "backend": "trl",
                "training": {
                    budget_field: budget_value,
                    "effective_batch_size": 32,
                    "per_device_train_batch_size": 1,
                },
                "sft": {"max_seq_length": 32_768, "packing": True},
                "data": {
                    "datasets": [
                        {
                            "name": "dummy",
                            "path": "dummy/path",
                            "weight": 1.0,
                        }
                    ]
                },
            }
        )
    )

    config = PostTrainingConfig.load(config_path)
    assert config.training.max_steps is None
    assert config.resolve_max_steps() == 2
    assert build_common_training_kwargs(config, tmp_path)["max_steps"] == 2

    frozen_path = tmp_path / "frozen.yaml"
    config.save(frozen_path)
    reloaded = PostTrainingConfig.load(frozen_path)

    assert reloaded.training.max_steps is None
    assert reloaded.resolve_max_steps() == 2


def test_explicit_and_derived_step_budgets_conflict(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "backend": "trl",
                "training": {
                    "max_steps": 2,
                    "num_training_tokens": 2_097_152,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="Training length is over-specified"):
        PostTrainingConfig.load(config_path)


def test_deepspeed_old_style_config_path_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "backend": "trl",
                "training": {
                    "max_steps": 1,
                    "effective_batch_size": 1,
                    "per_device_train_batch_size": 1,
                },
                "deepspeed": {"config_path": "configs/deepspeed/zero2.yaml"},
                "data": {
                    "datasets": [
                        {
                            "name": "dummy",
                            "path": "dummy/path",
                            "weight": 1.0,
                        }
                    ]
                },
            }
        )
    )

    with pytest.raises(ValueError, match="deepspeed.config_path is no longer supported"):
        PostTrainingConfig.load(config_path)


@pytest.mark.parametrize(
    ("tokenizer_name_or_path", "tokenizer_revision", "expected"),
    [
        # No tokenizer override: the tokenizer follows the model.
        (None, None, ("org/model", "r1")),
        # A tokenizer revision alone pins the tokenizer inside the model repo.
        (None, "r2", ("org/model", "r2")),
        # A separate repo ignores model.revision, which pins a different repo.
        ("org/tokenizer", None, ("org/tokenizer", None)),
        ("org/tokenizer", "r2", ("org/tokenizer", "r2")),
    ],
)
def test_resolve_tokenizer(tokenizer_name_or_path, tokenizer_revision, expected):
    model = ModelConfig(
        name_or_path="org/model",
        revision="r1",
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_revision=tokenizer_revision,
    )

    assert model.resolve_tokenizer() == expected


def test_resolve_tokenizer_without_any_revision():
    assert ModelConfig(name_or_path="org/model").resolve_tokenizer() == ("org/model", None)


def test_relative_mix_manifest_expands_and_frozen_config_is_self_contained(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    mix_path = tmp_path / "mixes" / "instruct.yaml"
    mix_path.parent.mkdir()
    mix_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "flag-sft-instruct-v1",
                "method": "sft",
                "backend": "trl",
                "sampling": {"mode": "multiplier"},
                "data": {
                    "chat_template": "olmo3-instruct-sft",
                    "seed": 7,
                    "datasets": [
                        {
                            "name": "dolci",
                            "path": "org/dolci",
                            "revision": "abc123",
                            "split": "train",
                            "weight": 0.5,
                            "catalogue_entry": "dolci/1.0.0/README.md",
                            "schema": "chat-messages-v1",
                            "expected_rows": 10,
                            "locations": {"lumi": "/scratch/dolci"},
                        }
                    ],
                },
            }
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "backend": "trl",
                "training": {
                    "max_steps": 1,
                    "effective_batch_size": 1,
                    "per_device_train_batch_size": 1,
                },
                "data": {
                    "mix_manifest": "mixes/instruct.yaml",
                    "location": "lumi",
                },
            }
        )
    )

    config = PostTrainingConfig.load(config_path)
    assert config.data.mix_id == "flag-sft-instruct-v1"
    assert len(config.data.mix_digest) == 64
    assert config.data.chat_template == "olmo3-instruct-sft"
    assert config.data.seed == 7
    assert config.data.datasets[0].revision == "abc123"
    assert config.data.datasets[0].resolve_path(config.data.location) == "/scratch/dolci"

    frozen_path = tmp_path / "frozen.yaml"
    config.save(frozen_path)
    mix_path.unlink()
    reloaded = PostTrainingConfig.load(frozen_path)
    assert reloaded.data.mix_id == config.data.mix_id
    assert reloaded.data.mix_digest == config.data.mix_digest
    assert reloaded.data.datasets[0].path == "org/dolci"


def test_mix_manifest_rejects_method_mismatch(tmp_path):
    mix_path = tmp_path / "dpo.yaml"
    mix_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "dpo-v1",
                "method": "dpo",
                "data": {"datasets": [{"name": "prefs", "path": "org/prefs"}]},
            }
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "training": {"max_steps": 1},
                "data": {"mix_manifest": str(mix_path)},
            }
        )
    )

    with pytest.raises(ValueError, match="does not match run method"):
        PostTrainingConfig.load(config_path)


def test_collection_release_expands_into_current_dataset_config(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    collection_path = tmp_path / "collection" / "pilot" / "dolci" / "metadata.yaml"
    collection_path.parent.mkdir(parents=True)
    collection_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "dolci",
                "task": "sft",
                "training_eligible": True,
                "catalogue_entry": "dolci/1.0.0/README.md",
                "checks": {
                    name: {"status": "pending"}
                    for name in (
                        "license",
                        "pii",
                        "safety",
                        "integrity",
                        "language_quality",
                        "task_quality",
                        "contamination",
                    )
                },
                "source": {"default": {"revision": "source123"}},
                "release": {
                    "candidate": {
                        "status": "candidate",
                        "input": "source.default",
                        "dataset": {
                            "path": "org/dolci-release",
                            "revision": "release123",
                            "split": "train",
                            "schema": "chat-messages-v1",
                            "configs": {"sv": {"expected_rows": 12}},
                            "locations": {"lumi": "/scratch/release/dolci"},
                        },
                    }
                },
            }
        )
    )
    mix_path = tmp_path / "mixes" / "pilot" / "sft.yaml"
    mix_path.parent.mkdir(parents=True)
    mix_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "collection-sft-v1",
                "status": "pilot",
                "method": "sft",
                "sampling": {"mode": "multiplier"},
                "data": {
                    "datasets": [
                        {
                            "name": "dolci_sv",
                            "collection": "../../collection/pilot/dolci/metadata.yaml",
                            "release": "candidate",
                            "subset": "sv",
                            "weight": 0.1,
                        }
                    ]
                },
            }
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "backend": "trl",
                "training": {"max_steps": 1},
                "data": {"mix_manifest": str(mix_path), "location": "lumi"},
            }
        )
    )

    config = PostTrainingConfig.load(config_path)
    entry = config.data.datasets[0]
    assert entry.path == "org/dolci-release"
    assert entry.revision == "release123"
    assert entry.release == "candidate"
    assert entry.release_status == "candidate"
    assert entry.expected_rows == 12
    assert entry.resolve_path(config.data.location) == "/scratch/release/dolci"
    assert len(entry.collection_digest) == 64
    assert entry.checks["pii"]["status"] == "pending"


def test_approved_mix_rejects_pending_collection_check(tmp_path):
    collection_path = tmp_path / "metadata.yaml"
    checks = {
        name: {"status": "complete"}
        for name in (
            "license",
            "pii",
            "safety",
            "integrity",
            "language_quality",
            "task_quality",
            "contamination",
        )
    }
    checks["pii"] = {"status": "pending"}
    collection_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "unsafe",
                "task": "sft",
                "training_eligible": True,
                "checks": checks,
                "source": {"default": {}},
                "release": {
                    "default": {
                        "status": "approved",
                        "input": "source.default",
                        "dataset": {
                            "path": "org/release",
                            "revision": "abc123",
                            "split": "train",
                            "schema": "chat-messages-v1",
                        },
                    }
                },
            }
        )
    )
    mix_path = tmp_path / "mix.yaml"
    mix_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "approved-sft",
                "status": "approved",
                "method": "sft",
                "data": {
                    "datasets": [
                        {
                            "name": "unsafe",
                            "collection": "metadata.yaml",
                            "release": "default",
                        }
                    ]
                },
            }
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "sft",
                "training": {"max_steps": 1},
                "data": {"mix_manifest": str(mix_path)},
            }
        )
    )

    with pytest.raises(ValueError, match="checks.pii.status"):
        PostTrainingConfig.load(config_path)
