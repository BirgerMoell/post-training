---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft
datasets:
  - nvidia/HelpSteer3
  - openeurollm/Dolci-Instruct-DPO-translated
  - argilla/ultrafeedback-binarized-preferences-cleaned
tags:
  - dpo
  - trl
  - qwen3
  - multilingual
  - intermediate-checkpoint
  - lumi
---

# OELLM 9B three-source DPO — step 9,000

This is an **intermediate, full-weight DPO checkpoint**, not the final result of the run. It was saved at optimizer step 9,000 of 11,263 planned steps (0.799 epoch) while the one-epoch training job was still running. No held-out evaluation or downstream benchmark result was available when this checkpoint was published. Do not interpret the training preference accuracy below as a validation score.

The starting model is [Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft](https://huggingface.co/Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft), an OELLM 9B SFT checkpoint. This DPO stage optimizes response preferences; it does **not** add reasoning traces or establish reasoning performance. The model retains the base model's Qwen3-style chat format. The saved weights are full model weights in float32 safetensors, not a LoRA adapter; training used bf16 mixed precision. The eight weight shards total approximately 36.4 GB (decimal).

## Intended use and status

This is a research checkpoint for comparison, ablation, and evaluation. It is not a production-validated release. The final checkpoint may differ materially. The inherited empty `<think>` block convention from the SFT model should not be confused with a trained reasoning capability. Review the base model card for its tokenizer, chat template, stop-token, and broader use guidance.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "birgermoell/oellm-9b-threeway-dpo-step9000"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(
    repo_id, torch_dtype=torch.bfloat16, device_map="auto"
)
messages = [{"role": "user", "content": "What is the capital of Finland?"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False,
    return_dict=True, return_tensors="pt",
).to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Training data

The three datasets were transformed into conversational `prompt`, `chosen`, and `rejected` pairs and staged on LUMI. The counts below are **staged rows before the trainer's empty/overlength-pair filtering**. Each source has weight 1.0, so this is a natural-size mixture, **not** equal sampling by source. The exact staging counts and per-language Dolci counts are also included in [`data_manifest.json`](./data_manifest.json).

| Source | Training pairs | Held-out pairs | Preparation | Source license |
| --- | ---: | ---: | --- | --- |
| [NVIDIA HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3), `preference` | 36,299 | 1,920 | `overall_preference` sign selects the preferred response; 2,257 ties removed; native validation split retained | CC BY 4.0 |
| [OpenEuroLLM Dolci-Instruct-DPO-translated](https://huggingface.co/datasets/openeurollm/Dolci-Instruct-DPO-translated) | 1,388,132 | 14,029 | All 12 staged languages; every 100th source row held out; string and message-list responses normalized | Apache 2.0 |
| [Argilla UltraFeedback cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) | 60,307 | 610 | Every 100th source row held out; duplicated leading user turn removed from responses | MIT |
| **Total** | **1,484,738** | **16,559** | | |

Dolci languages and staged training/validation pair counts: Czech 116,249/1,175; German 115,264/1,165; Greek 101,516/1,026; English 243,752/2,463; Spanish 126,629/1,280; Finnish 109,131/1,103; French 14,931/151; Italian 87,439/884; Polish 119,624/1,209; Romanian 116,510/1,177; Swedish 121,715/1,230; Ukrainian 115,372/1,166.

Dolci data files were resolved explicitly at dataset revision `a57841a6bbabfed8ca2597438995c2991e5b73b7`. The staged HelpSteer3 and UltraFeedback inputs were not revision-pinned in the staging script. This matters for exact reproduction. The normalization code is included as [`stage_threeway_dpo_full.py`](./stage_threeway_dpo_full.py); the run configuration is included as [`training_config.yaml`](./training_config.yaml). LUMI filesystem paths in these files describe the original run environment and are not required for inference.

## Training procedure

- Direct Preference Optimization with TRL 0.28.0, sigmoid loss, beta 0.1, and the SFT starting model as the implicit reference.
- One epoch planned; this artifact is step 9,000/11,263, approximately 79.9% through that epoch.
- Learning rate 5e-7, cosine scheduler, 3% warmup, weight decay 0, effective batch size 128, per-device batch size 1, seed 42.
- Maximum sequence length 2,048 tokens; truncated pairs dropped; Qwen3 chat template; gradient checkpointing enabled.
- Full-parameter FSDP training on 4 LUMI nodes × 8 AMD MI250 GPUs (32 GPUs total). Training used bf16 mixed precision and saved model-only checkpoints every 1,000 steps, retaining the latest two. This checkpoint does **not** include optimizer/scheduler state for exact training resume.
- Run configuration: `oellm-9b-threeway-dpo-full-v1`; Slurm job `22441061`. Training code started from `post-training` Git commit `db9d560011cb4502adaffe1ba3a94b87f9f404f4` with local modifications, so the commit alone is not a complete reproduction of the executed code.
- Framework versions recorded by the trainer: Transformers 4.57.3, PyTorch 2.9.1+rocm6.4, Datasets 4.6.0, Tokenizers 0.22.2.

At step 9,000, the trainer recorded loss **0.2622**, training-batch preference accuracy **0.8977**, and reward margin **3.1365**. These are online training metrics, not held-out evaluation. Full metric history is in [`trainer_state.json`](./trainer_state.json). The saved checkpoint is self-contained for inference but has not been independently benchmarked.

## Limitations and attribution

The preference mixture is dominated numerically by Dolci, and its 12 languages do not imply coverage of all OpenEuroLLM target languages. HelpSteer3 and UltraFeedback add other preference styles but are much smaller. Dataset provenance and licenses are listed above; please attribute the original dataset creators. The base model is Apache-2.0 licensed. No safety or multilingual-quality claim is made from the training metrics alone.
