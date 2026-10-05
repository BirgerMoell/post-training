#!/usr/bin/env python3
"""Stage deterministic 1% DPO slices for a small mixed-dataset run."""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

from urllib.request import urlopen

SEED = 42


def write_jsonl_gz(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_message_list(value):
    if isinstance(value, list):
        return [{"role": x["role"], "content": x["content"]} for x in value]
    raise TypeError(f"expected message list, got {type(value)}")


def main() -> None:
    out = Path("/scratch/project_465002530/users/bmoell/helpsteer3-dpo/artifacts/data/threeway-1pct")
    rng = random.Random(SEED)

    # HelpSteer3: exact 1% of the 38,459-row staged train split.
    with gzip.open("/scratch/project_465002530/users/bmoell/helpsteer3-dpo/artifacts/data/preference-train.jsonl.gz", "rt", encoding="utf-8") as f:
        hs = [json.loads(line) for line in f]
    hs = rng.sample(hs, max(1, round(len(hs) * 0.01)))
    hs_out = []
    for x in hs:
        p = int(x["overall_preference"])
        if p == 0:
            continue
        chosen, rejected = (x["response2"], x["response1"]) if p > 0 else (x["response1"], x["response2"])
        hs_out.append({"prompt": x["context"], "chosen": [{"role": "assistant", "content": chosen}], "rejected": [{"role": "assistant", "content": rejected}], "source": "nvidia/HelpSteer3"})

    # Dolci: the checked-in English smoke slice is already a deterministic 1% slice.
    dolci_path = "/pfs/lustrep3/scratch/project_465002530/users/bmoell/helpsteer3-dpo/artifacts/data/dolci-en-1pct.jsonl"
    dolci = []
    with open(dolci_path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            dolci.append({"prompt": normalize_message_list(x["prompt"]), "chosen": normalize_message_list(x["chosen"]), "rejected": normalize_message_list(x["rejected"]), "source": "openeurollm/Dolci-Instruct-DPO-translated"})

    # UltraFeedback has 60,917 rows; the dataset-server API provides a stable
    # row slice, so take exactly 609 rows (the requested 1%) without downloading
    # the full 143 MB parquet dataset on the login node.
    ultra = []
    for offset in range(0, 609, 100):
        length = min(100, 609 - offset)
        url = ("https://datasets-server.huggingface.co/rows?dataset="
               "argilla%2Fultrafeedback-binarized-preferences-cleaned&config=default"
               f"&split=train&offset={offset}&length={length}")
        with urlopen(url, timeout=120) as response:
            ultra.extend(x["row"] for x in json.load(response)["rows"])
    ultra_rows = []
    for x in ultra:
        chosen = normalize_message_list(x["chosen"])
        rejected = normalize_message_list(x["rejected"])
        ultra_rows.append({"prompt": [{"role": "user", "content": x["prompt"]}], "chosen": chosen[1:] if chosen and chosen[0]["role"] == "user" else chosen, "rejected": rejected[1:] if rejected and rejected[0]["role"] == "user" else rejected, "source": "argilla/ultrafeedback-binarized-preferences-cleaned"})

    write_jsonl_gz(hs_out, out / "helpsteer3-1pct.jsonl.gz")
    write_jsonl_gz(dolci, out / "dolci-en-1pct.jsonl.gz")
    write_jsonl_gz(ultra_rows, out / "ultrafeedback-1pct.jsonl.gz")
    print({"helpsteer3": len(hs_out), "dolci": len(dolci), "ultrafeedback": len(ultra_rows), "total": len(hs_out) + len(dolci) + len(ultra_rows)})


if __name__ == "__main__":
    main()
