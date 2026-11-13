# Local LLM backend (§7.1, ADR-043, ADR-055)

Grok-driven agents (M03, M04, M22 advisor, M24 vague filter, M25, M41, M42, M44) call a **local** model through `/agents/llm_cache.py` → `/agents/local_llm.py`. Default mode is offline (rule path). Real inference is GPU-only.

## Selection

| Variable | Values | Default |
| :---- | :---- | :---- |
| `STAMPEDE_LLM` | `offline` \| `grok` \| `local` | `offline` |
| `STAMPEDE_LOCAL_LLM_MODEL` | Any server model id | `Meta-Llama-3.1-8B-Instruct-Q4_K_M` |
| `STAMPEDE_LOCAL_LLM_URL` | OpenAI-compatible base URL | `http://127.0.0.1:8080/v1` |
| `STAMPEDE_LOCAL_LLM_WEIGHTS` | GGUF directory | registry `Local LLM weights directory` (repo-relative `models/gguf`) |
| `STAMPEDE_LOCAL_LLM_GGUF` | Full GGUF path | `{weights}/{model}.gguf` |
| `STAMPEDE_LOCAL_LLM_FAKE` | `1` to force the in-process fake | unset |

Swap models by changing `STAMPEDE_LOCAL_LLM_MODEL` (and the GGUF name). Any OpenAI-compatible server works. A reply that fails pydantic / schema checks is retried once, then the agent uses its deterministic rule path.

## Switch on (RTX 3060, ~5 GB at Q4)

1. Put `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` under the weights directory (default `models/gguf`).
2. `export STAMPEDE_LOCAL_LLM_WEIGHTS=/path/to/gguf-dir` if the default path is wrong for this machine.
3. `./scripts/serve_local_llm.sh` — refuses to start without an NVIDIA GPU.
4. In the sim shell: `export STAMPEDE_LLM=local`.

Unit tests never load weights or CUDA. They use `FakeLocalBackend` (CPU).
