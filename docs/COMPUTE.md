# Compute

This note summarises the compute behind the released runs. Every run records its own settings in
`runs/<run>/env.json` (serving image digest, model revision, context length, GPU memory share) and
`runs/<run>/manifest.json` (code commit, worlds, parameters, output hashes).

## Hardware

- GPUs: NVIDIA A100 40GB. Two GPUs served the experiments registered in `prereg/v0.md` to
  `prereg/v3_controls.md`. One GPU served the experiments registered in `prereg/v4_round3.md` and
  `prereg/v4b_round3_claude.md`.
- Experiment drivers, scoring and all analyses run on CPU. They make no GPU calls of their own.
  The drivers ran in CPU containers built from `docker/Dockerfile` (`scripts/run.sh`), and the native
  memory systems in the images of `docker/Dockerfile.native` (`scripts/build_native.sh`).

## GPU time

About 140 A100 GPU-hours in total, counted as the time a model server held a GPU. This is an upper
bound on use, since servers also idled between runs.

| Registrations | GPUs | GPU-hours (approx.) | Experiments |
|---|---|---|---|
| v0 to v3 | 2 | 117 | development and test worlds of the main study, writer switches, horizons, sync, native systems, export, failure-mode probes, replication, other model families, model-written derivations, rewritten worlds, controls |
| v4 and v4b | 1 | 23 | refresh and upsert, Claude writers, long thinking, held-out definitions, program sets, Mem0 interventions, unnamed probes, natural branch, Claude reader, definitions in the user's own words |

Scoring uses no LLM judge, so evaluating a released output costs no GPU time.

## Models served locally

All open-weight models ran under vLLM 0.30.0 (`vllm/vllm-openai:v0.30.0`, image
`sha256:d076d77d8dade699041dbcb6bb989a058e51b532e6367f13f80b5a4e63ee5c53`) in bfloat16 with prefix
caching, one model per GPU unless stated. gpt-oss-20b ran in its native MXFP4 weights.

| Model | Revision | Role |
|---|---|---|
| Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | main writer and reader, 32,768-token context |
| Qwen/Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | embeddings for plain RAG and the native systems |
| BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` | embedding swap |
| XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B | `2367e865d009c13ac81713a2878291d33ab28177` | writer switch, same tier |
| Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | writer switch, up and down, tensor parallel over two GPUs |
| meta-llama/Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` | other model family |
| google/gemma-4-12B-it | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | other model family, world rewriter |
| openai/gpt-oss-20b | `6cee5e81ee83917806bbde320786a8fb61efebee` | other model family |
| mistralai/Ministral-3-14B-Instruct-2512-BF16 | `3cea74c1ebaf5ce5f5a2553de470e2ceab825142` | reader swap, new writer |
| Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | new writer |

The first reader swap to Ministral-3-14B ran on vLLM 0.20.0. It was rerun on vLLM 0.30.0, and the
first run is kept only to measure the engine difference (`runs/stage3_theta_r_engine`).

## Anthropic API

Claude Sonnet 5.5 (`claude-sonnet-5-5`) and Claude Opus 5.5 (`claude-opus-5-5`) ran through
Anthropic's API. They wrote definition items and programs, and Sonnet 5.5 also served as a reader.
The reader requests went through the Message Batches API. Total API spend was about USD 340.
