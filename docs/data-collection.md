# Data collection and mix manifests

The framework supports the same layered data model used by OpenEuroLLM
pre-training:

```text
source + overlays (PII, contamination, quality)
                    │
                    ▼
             canonical release
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
      TRL loading      LlamaFactory registry
```

The collection lives in `OpenEuroLLM/post-training-data`; this repository only
needs a path to a mix manifest. Model and optimizer settings remain in the
ordinary run YAML.

## Run configuration

```yaml
method: sft
backend: trl

data:
  mix_manifest: /shared/post-training-data/mixes/pilots/sft-instruct-smoke-v1.yaml
  location: lumi
```

Relative manifest paths are resolved from the run config. Relative collection
paths inside a mix are resolved from that mix. `location` selects a release's
cluster mirror and falls back to its portable `path`, normally a Hugging Face
dataset ID.

At configuration load time, collection entries are compiled into the existing
`DatasetEntry` fields. Direct `data.datasets` and direct entries inside a mix
remain valid for backward compatibility. A config cannot set both
`data.mix_manifest` and `data.datasets`.

## Reproducibility and approvals

Every Hub release should use a commit revision. Every local release should
have a checksum or immutable versioned directory. Submission writes a
`data-mix.lock.yaml` containing:

- the expanded dataset identities and resolved paths;
- the mix-manifest digest;
- every referenced collection-metadata digest;
- selected release and check states; and
- seed, method, backend, and chat template.

Changing collection metadata therefore changes the effective mix digest even
when the mix file itself is unchanged. A frozen config already contains the
expanded entries and reloads without the external repository.

An `approved` mix is rejected unless every selected release is `approved` and
its license, PII, safety, integrity, language-quality, task-quality, and
contamination checks are `complete` or explicitly `inapplicable` with a
reason. Draft and pilot mixes can select candidate releases so that processing
and schema integration can be tested without representing them as production
ready.

## Backend behavior

TRL loads the selected Hub revision or local release directly. It can read a
single `.jsonl.zst`, `.jsonl.gz`, Parquet, CSV, JSON, or text file, and aligns
optional nested conversation fields before concatenating compatible datasets.

For a LlamaFactory mix, submission generates a run-local
`llamafactory_data/dataset_info.json` from each release's `llamafactory`
mapping and points the native config to it. Because LlamaFactory's current
registry has no Hub revision field, login-node prefetching materializes the
selected revision and the generated registry points at its revision-specific
Arrow files. The adapter refuses a pinned but unmaterialized Hub source rather
than silently training from `main`. The current adapter requires `weight: 1.0`;
weighted LlamaFactory interleaving remains governed by LlamaFactory-native
settings.

RLVR is intentionally not claimed here because the current repository exposes
SFT and DPO trainers only. The collection schema can catalogue RLVR releases,
but runtime support should be added with the RL backend rather than disguised
as SFT.
