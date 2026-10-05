"""Dataset loading, transformation, and mixing.

The main entry point is :func:`load_and_mix_datasets` which reads the
``data`` section of the config, loads each dataset, applies per-dataset
transforms, rescales each dataset according to its weight, then concatenates
and shuffles the result.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset, Features, List, Value, concatenate_datasets, load_dataset

from post_training.data.transforms import get_transform

if TYPE_CHECKING:
    from post_training.config import DataConfig

logger = logging.getLogger(__name__)

_MAX_NUM_PROC = min(
    32,
    len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1),
)

_EXTENSION_TO_BUILDER = {
    ".parquet": "parquet",
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".txt": "text",
}


def _builder_for_local_file(path: Path) -> str | None:
    """Return the datasets builder for plain or compressed local files."""
    name = path.name.lower()
    for suffix, builder in _EXTENSION_TO_BUILDER.items():
        if name.endswith(suffix) or any(
            name.endswith(suffix + compression) for compression in (".zst", ".zstd", ".gz")
        ):
            return builder
    return None


def _load_dataset_entry(
    path: str,
    split: str,
    *,
    revision: str | None = None,
    **load_kwargs,
) -> Dataset:
    """Load *path* as a local data file, a local dataset directory, or a Hub repo id.

    ``load_dataset(path, ...)`` auto-detects a local file from the bare
    ``path`` string, but that inference is unreliable for a single literal
    file (as opposed to a directory it can glob into) — it can report "no
    data file found" for a file that demonstrably exists. Passing the
    builder name explicitly via ``data_files=`` for local files sidesteps
    that ambiguity; local directories and Hub repo ids keep working through
    the normal bare-path form.
    """
    local_path = Path(path)
    if local_path.is_file():
        builder = _builder_for_local_file(local_path)
        if builder is None:
            raise ValueError(
                f"Unsupported local dataset file extension '{local_path.suffix}' for '{path}'. "
                f"Supported base extensions: {sorted(_EXTENSION_TO_BUILDER)} "
                "(optionally compressed with .zst, .zstd, or .gz)."
            )
        # Name the local builder split after the configured source split.  A
        # standalone validation file would otherwise be registered as
        # ``train`` by datasets and ``split='validation'`` would fail.
        return load_dataset(
            builder,
            data_files={split: str(local_path)},
            split=split,
            **load_kwargs,
        )
    if revision is not None and not local_path.exists():
        load_kwargs["revision"] = revision
    return load_dataset(path, split=split, **load_kwargs)


def _is_null_feature(feature: object) -> bool:
    return isinstance(feature, Value) and feature.dtype == "null"


def _merge_list_struct_fields(
    column: str,
    features: list[List],
) -> dict:
    """Build a nullable union for a list-of-struct column across datasets."""
    merged: dict = {}
    for feature in features:
        if not isinstance(feature.feature, dict):
            raise ValueError(f"Column '{column}' is not consistently a list of message records.")
        for name, field_feature in feature.feature.items():
            if name not in merged or _is_null_feature(merged[name]):
                merged[name] = field_feature
                continue
            if _is_null_feature(field_feature) or merged[name] == field_feature:
                continue
            raise ValueError(
                f"Cannot align nested field '{column}.{name}': conflicting feature types "
                f"{merged[name]!r} and {field_feature!r}. Normalize this field upstream."
            )
    return merged


def _align_nested_list_structs(datasets: list[Dataset], num_proc: int) -> list[Dataset]:
    """Align optional fields inside ``messages``/preference conversation lists.

    ``concatenate_datasets`` already fills missing top-level columns with null,
    but it cannot align nested records such as ``messages[{role, content}]``
    with function-calling messages that additionally carry ``functions`` and
    ``function_calls``. Add the missing nested keys as nullable fields before
    concatenation while preserving every field present in any input.
    """
    if len(datasets) < 2:
        return datasets

    columns = sorted(set().union(*(ds.column_names for ds in datasets)))
    unions: dict[str, dict] = {}
    for column in columns:
        list_features = [
            ds.features[column]
            for ds in datasets
            if column in ds.features
            and isinstance(ds.features[column], List)
            and isinstance(ds.features[column].feature, dict)
        ]
        if len(list_features) < 2:
            continue
        merged = _merge_list_struct_fields(column, list_features)
        if any(feature.feature != merged for feature in list_features):
            unions[column] = merged

    if not unions:
        return datasets

    aligned: list[Dataset] = []
    for ds in datasets:
        applicable = {column: fields for column, fields in unions.items() if column in ds.features}
        if not applicable:
            aligned.append(ds)
            continue

        target_features = Features(dict(ds.features))
        for column, fields in applicable.items():
            target_features[column] = List(fields)

        def normalize_nested(row: dict) -> dict:
            updates = {}
            for column, fields in applicable.items():
                values = row.get(column)
                if values is None:
                    updates[column] = None
                    continue
                updates[column] = [{name: item.get(name) for name in fields} for item in values]
            return updates

        aligned.append(
            ds.map(
                normalize_nested,
                features=target_features,
                num_proc=num_proc,
                desc="aligning nested conversation schemas",
            )
        )
    return aligned


def _resolve_num_proc(configured: int | None) -> int:
    """Return the number of worker processes for ``.map()`` / ``.filter()``.

    When *configured* is ``None`` (the default), use the process CPU affinity
    mask so SLURM jobs respect their allocated cores instead of all CPUs on an
    HPC node. An explicit value is still clamped to that affinity count so we
    never request more workers than allocated cores.
    """
    if configured is not None:
        return min(configured, _MAX_NUM_PROC)
    return _MAX_NUM_PROC


def _resample_to_size(ds: Dataset, target_n: int, seed: int) -> Dataset:
    """Return exactly ``target_n`` rows from ``ds``.

    Undersampling is done without replacement. Oversampling repeats full
    shuffled copies of the dataset and takes a final remainder slice.
    """
    n = len(ds)
    if target_n <= 0:
        raise ValueError("target_n must be positive.")
    if n == 0:
        raise ValueError("Cannot resample an empty dataset.")

    ds_shuffled = ds.shuffle(seed=seed)

    if target_n <= n:
        return ds_shuffled.select(range(target_n))

    full_copies = target_n // n
    remainder = target_n % n

    copies: list[Dataset] = [ds_shuffled] * full_copies
    if remainder > 0:
        copies.append(ds_shuffled.select(range(remainder)))

    if len(copies) == 1:
        return copies[0]
    return concatenate_datasets(copies)


def load_and_mix_datasets(
    config: DataConfig,
    dataset_filter_fn: Callable[[Dataset, int], Dataset] | None = None,
    columns_to_keep: list[str] | None = None,
    features: Features | None = None,
) -> Dataset:
    """Load, transform, and optionally filter/mix datasets.

    Parameters
    ----------
    config:
        The ``data`` section of :class:`PostTrainingConfig`.
    dataset_filter_fn:
        Optional callable ``(dataset, num_proc) -> dataset`` applied to each
        dataset after transforms and column selection. Each training method
        passes its own filter (e.g. SFT drops rows whose assistant tokens
        never enter the loss, DPO drops rows with an empty ``chosen`` or
        ``rejected``). Bind extra arguments with :func:`functools.partial`.
        When ``None``, no filtering is applied.
    columns_to_keep:
        Optional list of columns to retain after loading/mapping. When a
        transform is applied, original input columns are removed before the
        transform output is materialized.
    features:
        Optional dataset schema to enforce while mapping transformed rows.
        Already-structured datasets (transform: null) are preserved after
        top-level column selection so nested message fields remain available
        to chat templates.

    Returns
    -------
    datasets.Dataset
        A single concatenated and shuffled dataset ready for the trainer.
    """
    entries = config.datasets
    if not entries:
        raise ValueError("No datasets specified in data.datasets.")

    num_proc = _resolve_num_proc(config.num_proc)
    logger.info("Dataset processing will use num_proc=%d", num_proc)
    seed = config.seed

    weights: list[float] = []
    for entry in entries:
        if entry.weight < 0:
            raise ValueError(
                f"data.datasets[].weight must be non-negative, got {entry.weight} "
                f"for dataset '{entry.name}'."
            )
        weights.append(entry.weight)

    loaded_datasets: list[Dataset] = []
    for entry in entries:
        resolved_path = entry.resolve_path(config.location)
        logger.info(
            "Loading dataset '%s' from '%s' (revision=%s, data_dir=%s, subset=%s, "
            "split=%s, weight=%s, transform=%s)",
            entry.name,
            resolved_path,
            entry.revision,
            entry.data_dir,
            entry.subset,
            entry.split,
            entry.weight,
            entry.transform,
        )

        # Build kwargs for load_dataset, only passing optional params when set.
        load_kwargs: dict = {}
        if entry.data_dir is not None:
            load_kwargs["data_dir"] = entry.data_dir
        if entry.subset is not None:
            load_kwargs["name"] = entry.subset

        ds = _load_dataset_entry(
            resolved_path,
            entry.split,
            revision=entry.revision,
            **load_kwargs,
        )

        if entry.expected_rows is not None and len(ds) != entry.expected_rows:
            raise ValueError(
                f"Dataset '{entry.name}' has {len(ds)} rows at the selected source, "
                f"expected {entry.expected_rows}. Check its revision, subset, and split."
            )

        # Apply optional per-dataset transform.
        if entry.transform is not None:
            try:
                transform_fn = get_transform(entry.transform)
            except KeyError as exc:
                raise KeyError(
                    f"Unknown transform '{entry.transform}' for dataset '{entry.name}'. "
                    "Define it in 'post_training.data.transforms' using "
                    "@register_transform, or update your 'data.datasets[].transform' "
                    "value. Original error: "
                    f"{exc}"
                ) from exc

            logger.info(
                "Applying transform '%s' to dataset '%s'.",
                entry.transform,
                entry.name,
            )
            map_kwargs: dict = {"num_proc": num_proc}
            if columns_to_keep is not None or features is not None:
                map_kwargs["remove_columns"] = ds.column_names
            if features is not None:
                map_kwargs["features"] = features
            ds = ds.map(transform_fn, **map_kwargs)

        if columns_to_keep is not None:
            present = [column for column in columns_to_keep if column in ds.column_names]
            if not present:
                raise KeyError(
                    f"None of the expected columns {columns_to_keep} were found in "
                    f"dataset '{entry.name}'. Available columns: {ds.column_names}"
                )
            ds = ds.select_columns(present)

        # Apply the method-specific dataset filter (e.g. SFT drops rows with
        # an all-zero assistant loss mask, DPO drops empty preference pairs).
        if dataset_filter_fn is not None:
            logger.info("Filtering dataset '%s' (%d rows).", entry.name, len(ds))
            ds = dataset_filter_fn(ds, num_proc=num_proc)
            logger.info("Dataset '%s' has %d rows after filtering.", entry.name, len(ds))

        loaded_datasets.append(ds)

    loaded_datasets = _align_nested_list_structs(loaded_datasets, num_proc)

    resampled_datasets: list[Dataset] = []
    for idx, (ds, weight) in enumerate(zip(loaded_datasets, weights, strict=True)):
        target_n = round(weight * len(ds))
        logger.info(
            "Resampling dataset '%s' from %d rows to %d rows (weight=%s).",
            entries[idx].name,
            len(ds),
            target_n,
            weight,
        )
        if target_n <= 0:
            continue

        resampled_datasets.append(_resample_to_size(ds, target_n, seed + idx))

    if not resampled_datasets:
        raise ValueError(
            "No rows left after applying data.datasets[].weight. Check your weights and filters."
        )

    if len(resampled_datasets) == 1:
        mixed = resampled_datasets[0]
    else:
        mixed = concatenate_datasets(resampled_datasets)

    return mixed.shuffle(seed=seed)
