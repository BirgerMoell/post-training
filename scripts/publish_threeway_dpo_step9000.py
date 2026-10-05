#!/usr/bin/env python3
"""Publish a verified, immutable step-9000 DPO snapshot to Hugging Face."""

from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "birgermoell/oellm-9b-threeway-dpo-step9000"
SOURCE = Path(
    "/flash/project_465002530/users/bmoell/helpsteer3-dpo/releases/"
    "oellm-9b-threeway-dpo-step9000"
)


def main() -> None:
    required = {
        "README.md",
        "training_config.yaml",
        "data_manifest.json",
        "stage_threeway_dpo_full.py",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "trainer_state.json",
    }
    required.update(f"model-{index:05d}-of-00008.safetensors" for index in range(1, 9))
    missing_local = sorted(name for name in required if not (SOURCE / name).is_file())
    if missing_local:
        raise FileNotFoundError(f"Incomplete local checkpoint: {missing_local}")

    api = HfApi()
    repo = api.repo_info(REPO_ID, repo_type="model")
    if not repo.private:
        raise RuntimeError("Repository must be private until upload verification completes")

    print(f"Uploading {SOURCE} to private {REPO_ID}", flush=True)
    commit = api.upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=str(SOURCE),
        commit_message="Publish intermediate DPO checkpoint at step 9000 with provenance",
    )
    print(f"Upload commit: {commit}", flush=True)

    remote = set(api.list_repo_files(REPO_ID, repo_type="model"))
    missing_remote = sorted(required - remote)
    if missing_remote:
        raise RuntimeError(f"Upload incomplete; keeping repository private: {missing_remote}")

    api.update_repo_visibility(repo_id=REPO_ID, repo_type="model", private=False)
    print(f"Published and verified: https://huggingface.co/{REPO_ID}", flush=True)


if __name__ == "__main__":
    main()
