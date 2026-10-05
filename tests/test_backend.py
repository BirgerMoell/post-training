"""Tests for TRLBackend.post_freeze() — the submission-time source freeze."""

import json
import logging

import pytest
import yaml

from post_training.backend import LlamaFactoryBackend, TRLBackend
from post_training.config import ContainerConfig, DatasetEntry, PostTrainingConfig


def _make_fake_repo(repo_dir, marker: str = "live") -> None:
    (repo_dir / "src" / "post_training").mkdir(parents=True)
    (repo_dir / "src" / "post_training" / "marker.txt").write_text(marker)
    (repo_dir / "scripts").mkdir(parents=True)
    (repo_dir / "scripts" / "marker.txt").write_text(marker)


def _config_with_container() -> PostTrainingConfig:
    cfg = PostTrainingConfig()
    cfg.container = ContainerConfig(
        image="/shared/containers/trl.sif",
        bind_mounts=["/scratch:/scratch"],
        env_file="/shared/env/cluster.env",
    )
    return cfg


def test_post_freeze_noop_without_container(tmp_path, monkeypatch):
    """No container configured — nothing is copied, run_dir stays untouched."""
    repo_dir = tmp_path / "repo"
    _make_fake_repo(repo_dir)
    monkeypatch.chdir(repo_dir)

    cfg = PostTrainingConfig()
    cfg.container = None
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    TRLBackend().post_freeze(cfg, run_dir)

    assert not (run_dir / "src").exists()
    assert not (run_dir / "scripts").exists()


def test_post_freeze_copies_when_fresh(tmp_path, monkeypatch):
    """Neither directory exists yet — both get copied from the live repo."""
    repo_dir = tmp_path / "repo"
    _make_fake_repo(repo_dir, marker="live")
    monkeypatch.chdir(repo_dir)

    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    TRLBackend().post_freeze(_config_with_container(), run_dir)

    assert (run_dir / "src" / "post_training" / "marker.txt").read_text() == "live"
    assert (run_dir / "scripts" / "marker.txt").read_text() == "live"


def test_post_freeze_skips_when_fully_frozen(tmp_path, monkeypatch, caplog):
    """Both already frozen — existing copies are preserved, not overwritten."""
    repo_dir = tmp_path / "repo"
    _make_fake_repo(repo_dir, marker="live-later")
    monkeypatch.chdir(repo_dir)

    run_dir = tmp_path / "outputs" / "my-run"
    (run_dir / "src" / "post_training").mkdir(parents=True)
    (run_dir / "src" / "post_training" / "marker.txt").write_text("frozen")
    (run_dir / "scripts").mkdir(parents=True)
    (run_dir / "scripts" / "marker.txt").write_text("frozen")

    with caplog.at_level(logging.WARNING):
        TRLBackend().post_freeze(_config_with_container(), run_dir)

    assert (run_dir / "src" / "post_training" / "marker.txt").read_text() == "frozen"
    assert (run_dir / "scripts" / "marker.txt").read_text() == "frozen"
    assert "already exists" in caplog.text


def test_post_freeze_completes_partial_state(tmp_path, monkeypatch, caplog):
    """Only one directory exists (interrupted prior attempt) — the missing one
    is copied to complete the freeze; the existing one is left untouched."""
    repo_dir = tmp_path / "repo"
    _make_fake_repo(repo_dir, marker="live")
    monkeypatch.chdir(repo_dir)

    run_dir = tmp_path / "outputs" / "my-run"
    (run_dir / "scripts").mkdir(parents=True)
    (run_dir / "scripts" / "marker.txt").write_text("frozen")
    # src/post_training deliberately left missing.

    with caplog.at_level(logging.WARNING):
        TRLBackend().post_freeze(_config_with_container(), run_dir)

    assert (run_dir / "src" / "post_training" / "marker.txt").read_text() == "live"
    assert (run_dir / "scripts" / "marker.txt").read_text() == "frozen"
    assert "already exists" in caplog.text


def test_post_freeze_recovers_from_interrupted_copy(tmp_path, monkeypatch):
    """A killed process mid-copytree leaves only a .tmp dir, not a corrupted
    frozen dir — the next post_freeze must redo the copy in full rather than
    silently adopting the half-written content.

    Regression test for the bug reported on PR #46: since post_freeze's
    existence check was the only signal used to decide "already frozen,
    leave it", a directory that existed but was only partially copied got
    permanently treated as a completed freeze, and the job silently ran
    with incomplete/stale code (`cd`'d into run_dir and imported from there).
    """
    repo_dir = tmp_path / "repo"
    _make_fake_repo(repo_dir, marker="live")
    monkeypatch.chdir(repo_dir)

    run_dir = tmp_path / "outputs" / "my-run"
    frozen_src = run_dir / "src" / "post_training"
    tmp_src = frozen_src.with_name(frozen_src.name + ".tmp")
    tmp_src.mkdir(parents=True)
    (tmp_src / "half-written.py").write_text("partial")
    # frozen_src itself deliberately does not exist yet — simulates the
    # process dying after shutil.copytree started writing but before it finished.

    TRLBackend().post_freeze(_config_with_container(), run_dir)

    assert (frozen_src / "marker.txt").read_text() == "live"
    assert not (frozen_src / "half-written.py").exists()
    assert not tmp_src.exists()


def test_trl_post_freeze_writes_resolved_mix_lock(tmp_path):
    cfg = PostTrainingConfig()
    cfg.container = None
    cfg.data.mix_id = "flag-sft-v1"
    cfg.data.mix_digest = "a" * 64
    cfg.data.mix_manifest = "/catalogue/mix.yaml"
    cfg.data.location = "lumi"
    cfg.data.datasets = [
        DatasetEntry(
            name="dolci",
            path="org/dolci",
            revision="abc123",
            locations={"lumi": "/scratch/dolci"},
        )
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    TRLBackend().post_freeze(cfg, run_dir)

    lock = yaml.safe_load((run_dir / "data-mix.lock.yaml").read_text())
    assert lock["mix_id"] == "flag-sft-v1"
    assert lock["datasets"][0]["resolved_path"] == "/scratch/dolci"
    assert lock["datasets"][0]["revision"] == "abc123"


def test_llamafactory_post_freeze_generates_registry_from_mix(tmp_path):
    cfg = PostTrainingConfig()
    cfg.backend = "llamafactory"
    cfg.llamafactory = {"stage": "sft", "dataset": "placeholder"}
    cfg.data.mix_id = "long-context-sft-v1"
    cfg.data.mix_digest = "b" * 64
    cfg.data.location = "lumi"
    cfg.data.datasets = [
        DatasetEntry(
            name="long_sft",
            path="org/long-sft",
            revision="def456",
            locations={"lumi": "/scratch/long-sft.jsonl"},
            llamafactory={
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
            },
        )
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    LlamaFactoryBackend().post_freeze(cfg, run_dir)

    registry = json.loads((run_dir / "llamafactory_data" / "dataset_info.json").read_text())
    assert registry["long_sft"]["file_name"] == "/scratch/long-sft.jsonl"
    lf_config = yaml.safe_load((run_dir / "llamafactory_config.yaml").read_text())
    assert lf_config["dataset"] == "long_sft"
    assert lf_config["dataset_dir"] == str((run_dir / "llamafactory_data").resolve())


def test_llamafactory_uses_prefetched_revision_files(tmp_path):
    cached = tmp_path / "cache" / "pinned.arrow"
    cached.parent.mkdir()
    cached.write_bytes(b"arrow")
    cfg = PostTrainingConfig()
    cfg.backend = "llamafactory"
    cfg.llamafactory = {"stage": "sft"}
    cfg.data.mix_id = "pinned-lf-v1"
    cfg.data.datasets = [
        DatasetEntry(
            name="long_sft",
            path="org/long-sft",
            revision="commit123",
            prefetched_files=[str(cached)],
            llamafactory={"formatting": "sharegpt"},
        )
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    LlamaFactoryBackend().post_freeze(cfg, run_dir)

    registry = json.loads((run_dir / "llamafactory_data" / "dataset_info.json").read_text())
    assert registry["long_sft"]["file_name"] == "releases/long_sft"
    link = run_dir / "llamafactory_data" / "releases" / "long_sft" / "part-00000.arrow"
    assert link.resolve() == cached.resolve()


def test_llamafactory_rejects_unmaterialized_pinned_hub_dataset(tmp_path):
    cfg = PostTrainingConfig()
    cfg.backend = "llamafactory"
    cfg.llamafactory = {"stage": "sft"}
    cfg.data.mix_id = "pinned-lf-v1"
    cfg.data.datasets = [DatasetEntry(name="long_sft", path="org/long-sft", revision="commit123")]
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="prefetch_assets=true"):
        LlamaFactoryBackend().post_freeze(cfg, run_dir)
