---
description: Quick-eval the infra_synth environment with the fast static verifier backend.
argument-hint: "[-n N] [-m model] [-b vllm_url]"
allowed-tools: Bash
---

Install + eval the environment (static backend = no Docker/Sentinel needed):

```
uv run vf-install infra-synth
uv run vf-eval infra-synth -a '{"verifier_backend":"static"}' $ARGUMENTS
```

Genuine build+smoke: use `'{"verifier_backend":"local-docker"}'` (needs Docker). Hardened: `'{"verifier_backend":"sentinel","sentinel_base_url":"http://localhost:8080"}'` (needs Sentinel up).
