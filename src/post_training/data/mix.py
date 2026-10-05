"""Load and freeze reusable post-training data-mix manifests.

Mix manifests intentionally contain data selection only. Model, optimiser,
infrastructure, and backend-specific training settings stay in the ordinary
post-training run config. At load time the manifest is expanded into the
existing ``data.datasets`` representation, so the rest of the framework does
not need a second data path.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from post_training.config import PostTrainingConfig


_REQUIRED_APPROVAL_CHECKS = (
    "license",
    "pii",
    "safety",
    "integrity",
    "language_quality",
    "task_quality",
    "contamination",
)


def _require_mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _read_yaml(path: Path, label: str) -> tuple[bytes, dict]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read {label} '{path}': {exc}") from exc
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {label} '{path}': {exc}") from exc
    return raw, _require_mapping(value, f"{label} '{path}'")


def _resolve_relative(reference: str, parent: Path) -> Path:
    path = Path(reference)
    if not path.is_absolute():
        path = parent / path
    return path.resolve()


def _validate_approved_collection(
    metadata: dict,
    release: dict,
    *,
    metadata_path: Path,
    release_name: str,
) -> None:
    if release.get("status") != "approved":
        raise ValueError(
            f"{metadata_path}: approved mixes require release.{release_name}.status='approved'."
        )

    checks = _require_mapping(metadata.get("checks"), f"{metadata_path}: checks")
    for name in _REQUIRED_APPROVAL_CHECKS:
        check = _require_mapping(checks.get(name), f"{metadata_path}: checks.{name}")
        status = check.get("status")
        if status not in {"complete", "inapplicable"}:
            raise ValueError(
                f"{metadata_path}: approved mixes require checks.{name}.status to be "
                f"'complete' or 'inapplicable'; got {status!r}."
            )
        if status == "inapplicable" and not check.get("reason"):
            raise ValueError(
                f"{metadata_path}: checks.{name}.reason is required when status='inapplicable'."
            )


def _expand_collection_entry(
    value: Any,
    *,
    manifest_path: Path,
    method: str | None,
    approved_mix: bool,
) -> tuple[dict, tuple[str, bytes] | None]:
    item = dict(_require_mapping(value, f"{manifest_path}: data.datasets[]"))
    collection_ref = item.pop("collection", None)
    if not collection_ref:
        # Direct dataset entries remain supported for backwards compatibility.
        return item, None
    if not isinstance(collection_ref, str):
        raise ValueError(f"{manifest_path}: data.datasets[].collection must be a string.")

    metadata_path = _resolve_relative(collection_ref, manifest_path.parent)
    raw, metadata = _read_yaml(metadata_path, "collection metadata")
    if str(metadata.get("schema_version")) != "1":
        raise ValueError(
            f"{metadata_path}: schema_version must be 1; got "
            f"{metadata.get('schema_version')!r}."
        )
    metadata_task = metadata.get("task")
    if metadata_task and method and metadata_task != method:
        raise ValueError(
            f"{metadata_path}: task={metadata_task!r} does not match mix method={method!r}."
        )
    if metadata.get("training_eligible") is not True:
        raise ValueError(
            f"{metadata_path}: training_eligible must be true before a release can be mixed."
        )

    release_name = item.pop("release", "default")
    releases = _require_mapping(metadata.get("release"), f"{metadata_path}: release")
    release = _require_mapping(
        releases.get(release_name), f"{metadata_path}: release.{release_name}"
    )
    if approved_mix:
        _validate_approved_collection(
            metadata,
            release,
            metadata_path=metadata_path,
            release_name=release_name,
        )

    dataset = dict(
        _require_mapping(release.get("dataset"), f"{metadata_path}: release.{release_name}.dataset")
    )
    configs = dataset.pop("configs", {})
    subset = item.get("subset", dataset.get("subset"))
    if configs:
        configs = _require_mapping(
            configs, f"{metadata_path}: release.{release_name}.dataset.configs"
        )
        if subset is None:
            raise ValueError(
                f"{metadata_path}: release.{release_name}.dataset defines configs; "
                "the mix entry must select subset."
            )
        selected = _require_mapping(
            configs.get(subset),
            f"{metadata_path}: release.{release_name}.dataset.configs.{subset}",
        )
        dataset.update(selected)
    elif "subset" in item:
        raise ValueError(
            f"{metadata_path}: subset selection must be declared under "
            f"release.{release_name}.dataset.configs."
        )

    protected = {
        "path",
        "revision",
        "data_dir",
        "split",
        "transform",
        "schema",
        "expected_rows",
        "sha256",
        "locations",
        "llamafactory",
    }
    overridden = protected.intersection(item)
    if overridden:
        raise ValueError(
            f"{manifest_path}: collection-backed entry '{item.get('name', metadata.get('name'))}' "
            f"must not override release identity fields {sorted(overridden)}. Update "
            f"{metadata_path} instead."
        )

    dataset.update(item)
    dataset.setdefault("name", metadata.get("name"))
    dataset.setdefault("catalogue_entry", metadata.get("catalogue_entry"))
    dataset["collection_metadata"] = str(metadata_path)
    dataset["collection_digest"] = hashlib.sha256(raw).hexdigest()
    dataset["release"] = release_name
    dataset["release_status"] = release.get("status")
    dataset["checks"] = metadata.get("checks", {})
    if not dataset.get("name"):
        raise ValueError(f"{metadata_path}: name must be set in metadata or the mix entry.")
    if not dataset.get("path"):
        raise ValueError(f"{metadata_path}: release.{release_name}.dataset.path is required.")
    return dataset, (collection_ref, raw)


def expand_mix_manifest(run_config: dict, config_path: Path) -> dict:
    """Expand ``data.mix_manifest`` into an ordinary structured run config.

    Explicit run-config data settings such as ``location`` and ``num_proc``
    override the corresponding manifest values. Dataset lists cannot be
    supplied in both places because merging them would be ambiguous.

    A frozen config already contains ``mix_digest`` and the expanded dataset
    entries. It therefore remains self-contained and is not expanded again.
    """

    data = run_config.get("data")
    if data is None:
        return run_config
    data = _require_mapping(data, "data")
    manifest_ref = data.get("mix_manifest")
    if not manifest_ref:
        return run_config
    if data.get("mix_digest") and data.get("datasets"):
        return run_config
    if data.get("datasets"):
        raise ValueError(
            "Set either data.mix_manifest or data.datasets, not both. "
            "The manifest expands into data.datasets automatically."
        )

    manifest_path = _resolve_relative(str(manifest_ref), config_path.parent)
    raw, manifest = _read_yaml(manifest_path, "data mix manifest")

    if str(manifest.get("schema_version")) != "1":
        raise ValueError(
            f"{manifest_path}: schema_version must be 1; got {manifest.get('schema_version')!r}."
        )
    mix_id = manifest.get("id")
    if not isinstance(mix_id, str) or not mix_id.strip():
        raise ValueError(f"{manifest_path}: id must be a non-empty string.")

    manifest_method = manifest.get("method")
    run_method = run_config.get("method")
    if manifest_method and run_method and manifest_method != run_method:
        raise ValueError(
            f"{manifest_path}: method={manifest_method!r} does not match "
            f"run method={run_method!r}."
        )

    manifest_backend = manifest.get("backend")
    run_backend = run_config.get("backend")
    if manifest_backend and run_backend and manifest_backend != run_backend:
        raise ValueError(
            f"{manifest_path}: backend={manifest_backend!r} does not match "
            f"run backend={run_backend!r}."
        )

    mix_data = _require_mapping(manifest.get("data"), f"{manifest_path}: data")
    dataset_specs = mix_data.get("datasets")
    if not isinstance(dataset_specs, list) or not dataset_specs:
        raise ValueError(f"{manifest_path}: data.datasets must be a non-empty list.")

    sampling = manifest.get("sampling", {"mode": "multiplier"})
    sampling = _require_mapping(sampling, f"{manifest_path}: sampling")
    if sampling.get("mode", "multiplier") != "multiplier":
        raise ValueError(
            f"{manifest_path}: only sampling.mode='multiplier' is supported by the "
            "current post-training loader."
        )

    expanded_datasets = []
    collection_blobs: list[tuple[str, bytes]] = []
    for item in dataset_specs:
        expanded_item, collection_blob = _expand_collection_entry(
            item,
            manifest_path=manifest_path,
            method=manifest_method or run_method,
            approved_mix=manifest.get("status") == "approved",
        )
        expanded_datasets.append(expanded_item)
        if collection_blob is not None:
            collection_blobs.append(collection_blob)

    digest = hashlib.sha256()
    digest.update(b"mix-manifest\0")
    digest.update(raw)
    for reference, metadata_raw in sorted(collection_blobs, key=lambda value: value[0]):
        digest.update(b"\0collection\0")
        digest.update(reference.encode())
        digest.update(b"\0")
        digest.update(metadata_raw)

    expanded_data = dict(mix_data)
    # Run-local choices override portable manifest defaults.
    expanded_data.update(data)
    expanded_data["datasets"] = expanded_datasets
    expanded_data["mix_manifest"] = str(manifest_path)
    expanded_data["mix_id"] = mix_id
    expanded_data["mix_digest"] = digest.hexdigest()

    expanded = dict(run_config)
    expanded["data"] = expanded_data
    return expanded


def write_mix_lock(config: PostTrainingConfig, run_dir: Path) -> Path | None:
    """Write the selected and resolved data identities into a run directory."""

    if not config.data.mix_id:
        return None

    datasets = []
    for entry in config.data.datasets:
        item = dataclasses.asdict(entry)
        item["resolved_path"] = entry.resolve_path(config.data.location)
        datasets.append(item)

    lock = {
        "schema_version": 1,
        "mix_id": config.data.mix_id,
        "mix_digest": config.data.mix_digest,
        "source_manifest": config.data.mix_manifest,
        "method": config.method,
        "backend": config.backend,
        "location": config.data.location,
        "chat_template": config.data.chat_template,
        "seed": config.data.seed,
        "datasets": datasets,
    }
    path = run_dir / "data-mix.lock.yaml"
    path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    return path
