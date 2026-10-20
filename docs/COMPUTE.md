# Compute

This note summarises the compute behind the released runs. Every run records its own settings in
`runs/<run>/env.json` (serving image digest, model revision, context length, GPU memory share) and
`runs/<run>/manifest.json` (code commit, worlds, parameters, output hashes).
`memshift.analysis.paper_extras` computes the GPU-hours, the Claude call dates, tokens and cost
below from these files and from the Claude rows of each run's `calls.jsonl`.

## Hardware

- GPUs: two NVIDIA A100 40GB.
- Experiment drivers, scoring and all analyses run on CPU. They make no GPU calls of their own.
  The drivers ran in CPU containers built from `docker/Dockerfile` (`scripts/run.sh`), and the native
  memory systems in the images of `docker/Dockerfile.native` (`scripts/build_native.sh`).

## GPU time

The model servers held the GPUs for about 140 A100 GPU-hours in all. The native-system runs used 34
of them, all other runs 79, and the servers idled for the rest.

A model server holds a GPU from its launch (recorded in `env.json`) to the end of the last run that
used it (`created_utc` in `manifest.json`), and each GPU counts the union of these intervals. The
time in which runs were active is split between the native-system runs and all other runs, sharing
each GPU equally between the two groups when both had a run on it.

Scoring uses no LLM judge, so evaluating a released output costs no GPU time.

## Models served locally

All open-weight models ran under vLLM 0.30.0 (`vllm/vllm-openai:v0.30.0`, image
`sha256:d076d77d8dade699041dbcb6bb989a058e51b532e6367f13f80b5a4e63ee5c53`) in bfloat16 with prefix
caching, one model per GPU unless stated. gpt-oss-20b ran in its native MXFP4 weights.

| Model | Revision | Role |
|---|---|---|
| Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | main writer and reader, program writer, record rewriter, field extractor, LLM of the native systems, 32,768-token context |
| Qwen/Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | embeddings for plain RAG and the native systems |
| BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` | embedding swap (failure-mode probes) |
| XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B | `2367e865d009c13ac81713a2878291d33ab28177` | writer switch, same tier |
| Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | writer switch, up and down, tensor parallel over two GPUs |
| meta-llama/Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` | other model families, program sets from six writers |
| google/gemma-4-12B-it | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | other model families, program sets from six writers, world rewriter (rewritten worlds) |
| openai/gpt-oss-20b | `6cee5e81ee83917806bbde320786a8fb61efebee` | other model families, program sets from six writers |
| mistralai/Ministral-3-14B-Instruct-2512-BF16 | `3cea74c1ebaf5ce5f5a2553de470e2ceab825142` | reader swap (failure-mode probes), new writer (other model families) |
| Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | new writer (other model families) |

## Anthropic API

Claude Sonnet 5.5 (`claude-sonnet-5-5`) and Claude Opus 5.5 (`claude-opus-5-5`) were called
through Anthropic's API from 2026-10-16 to 2026-10-19, with 278 million input and output tokens in
all at a cost of USD 336. Both wrote definition items and programs, and Sonnet 5.5 also served as a
reader, whose requests went through the Message Batches API.
