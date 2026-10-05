#!/usr/bin/env python3
"""Stage the full HelpSteer3, Dolci translated, and UltraFeedback DPO mix."""
from __future__ import annotations

import gzip
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_url

ROOT = Path("/flash/project_465002530/users/bmoell/helpsteer3-dpo/artifacts/data/threeway-full")
HELPSTEER_ROOT = Path("/scratch/project_465002530/users/bmoell/helpsteer3-dpo/artifacts/data")
DOLCI_LANGUAGES = ("cs", "de", "el", "en", "es", "fi", "fr", "it", "pl", "ro", "sv", "uk")
DOLCI_REPO = "openeurollm/Dolci-Instruct-DPO-translated"
DOLCI_REVISION = "a57841a6bbabfed8ca2597438995c2991e5b73b7"


def dolci_files() -> dict[str, tuple[str, list[str]]]:
    """Resolve explicit data files because mixed formats defeat repo inference."""
    paths = HfApi().list_repo_files(DOLCI_REPO, repo_type="dataset", revision=DOLCI_REVISION)
    result: dict[str, tuple[str, list[str]]] = {}
    for language in DOLCI_LANGUAGES:
        parquet = sorted(path for path in paths if path.startswith(f"{language}/shard") and path.endswith(".parquet"))
        jsonl = sorted(path for path in paths if path == f"{language}/translated_dpo.jsonl")
        if parquet:
            builder, selected = "parquet", parquet
        elif jsonl:
            builder, selected = "json", jsonl
        else:
            raise FileNotFoundError(f"No Dolci DPO data files for {language} at {DOLCI_REVISION}")
        urls = [hf_hub_url(DOLCI_REPO, path, repo_type="dataset", revision=DOLCI_REVISION) for path in selected]
        result[language] = (builder, urls)
    return result


def messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TypeError(f"expected message list, got {type(value).__name__}")
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in value]


def response_messages(value: Any) -> list[dict[str, str]]:
    """Dolci JSONL has response strings; its Parquet shards have messages."""
    if isinstance(value, str):
        return [{"role": "assistant", "content": value}]
    return messages(value)


def write_row(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_helpsteer(rows: Iterable[dict[str, Any]], source_split: str, train, validation) -> tuple[int, int, int]:
    train_count = validation_count = dropped = 0
    destination = validation if source_split == "validation" else train
    for row in rows:
        preference = int(row["overall_preference"])
        if preference == 0:
            dropped += 1
            continue
        chosen_text, rejected_text = (
            (row["response2"], row["response1"])
            if preference > 0
            else (row["response1"], row["response2"])
        )
        write_row(
            destination,
            {
                "prompt": messages(row["context"]),
                "chosen": [{"role": "assistant", "content": chosen_text}],
                "rejected": [{"role": "assistant", "content": rejected_text}],
                "source": "nvidia/HelpSteer3",
                "language": row.get("language", "unknown"),
            },
        )
        if source_split == "validation":
            validation_count += 1
        else:
            train_count += 1
    return train_count, validation_count, dropped


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}

    with ExitStack() as stack:
        hs_train = stack.enter_context(gzip.open(ROOT / "helpsteer3-train.jsonl.gz", "wt", encoding="utf-8"))
        hs_val = stack.enter_context(gzip.open(ROOT / "helpsteer3-validation.jsonl.gz", "wt", encoding="utf-8"))
        dolci_train = stack.enter_context(gzip.open(ROOT / "dolci-train.jsonl.gz", "wt", encoding="utf-8"))
        dolci_val = stack.enter_context(gzip.open(ROOT / "dolci-validation.jsonl.gz", "wt", encoding="utf-8"))
        ultra_train = stack.enter_context(gzip.open(ROOT / "ultrafeedback-train.jsonl.gz", "wt", encoding="utf-8"))
        ultra_val = stack.enter_context(gzip.open(ROOT / "ultrafeedback-validation.jsonl.gz", "wt", encoding="utf-8"))

        hs_total = {"train": 0, "validation": 0, "dropped_ties": 0}
        for split in ("train", "validation"):
            path = HELPSTEER_ROOT / f"preference-{split}.jsonl.gz"
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                source_rows = (json.loads(line) for line in handle)
                train_n, val_n, dropped_n = normalize_helpsteer(source_rows, split, hs_train, hs_val)
            hs_total["train"] += train_n
            hs_total["validation"] += val_n
            hs_total["dropped_ties"] += dropped_n
        counts["nvidia/HelpSteer3"] = hs_total

        dolci_total = {"train": 0, "validation": 0}
        dolci_by_language: dict[str, dict[str, int]] = {}
        explicit_files = dolci_files()
        for language in DOLCI_LANGUAGES:
            language_counts = {"train": 0, "validation": 0}
            builder, urls = explicit_files[language]
            print("dolci files", language, builder, len(urls), flush=True)
            dataset = load_dataset(
                builder,
                data_files={"train": urls},
                split="train",
                streaming=True,
            )
            for index, row in enumerate(dataset):
                record = {
                    "prompt": messages(row["prompt"]),
                    "chosen": response_messages(row["chosen"]),
                    "rejected": response_messages(row["rejected"]),
                    "source": "openeurollm/Dolci-Instruct-DPO-translated",
                    "language": language,
                }
                split = "validation" if index % 100 == 0 else "train"
                write_row(dolci_val if split == "validation" else dolci_train, record)
                language_counts[split] += 1
                dolci_total[split] += 1
            dolci_by_language[language] = language_counts
            print("dolci", language, language_counts, flush=True)
        counts["openeurollm/Dolci-Instruct-DPO-translated"] = {
            **dolci_total,
            "languages": len(DOLCI_LANGUAGES),
        }
        counts["dolci_by_language"] = dolci_by_language  # type: ignore[assignment]

        ultra_counts = {"train": 0, "validation": 0}
        ultra = load_dataset(
            "argilla/ultrafeedback-binarized-preferences-cleaned",
            split="train",
            streaming=True,
        )
        for index, row in enumerate(ultra):
            chosen = messages(row["chosen"])
            rejected = messages(row["rejected"])
            record = {
                "prompt": [{"role": "user", "content": str(row["prompt"])}],
                "chosen": chosen[1:] if chosen and chosen[0]["role"] == "user" else chosen,
                "rejected": rejected[1:] if rejected and rejected[0]["role"] == "user" else rejected,
                "source": "argilla/ultrafeedback-binarized-preferences-cleaned",
                "language": "en",
            }
            split = "validation" if index % 100 == 0 else "train"
            write_row(ultra_val if split == "validation" else ultra_train, record)
            ultra_counts[split] += 1
        counts["argilla/ultrafeedback-binarized-preferences-cleaned"] = ultra_counts

    manifest = {
        "seed_policy": "native HelpSteer3 validation; every 100th source row held out for Dolci and UltraFeedback",
        "sources": counts,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
