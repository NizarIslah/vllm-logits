# TODO

## Adaptive per-token alpha for logit steering

The logit-steering operators (`sparse steer` / `dense steer`) currently mix at a
fixed scalar `alpha` (default 0.7) for every fired position
(`processors/logit_repair.py`). Make alpha **per-token**: compute `alpha_t` from the
local specialist/ancestor logits so each position gets the mixing strength its own
geometry warrants (e.g. sharper-specialist tokens get more ancestor weight).

The signal functions already exist and are pure: `entropy_gap`, `chi2_divergence`,
`adaptive_alpha` in `alpha.py`. The work is to evaluate one of them at the fired
position inside `LogitRepairProcessor` and use the resulting `alpha_t` in the mix,
plus a knob to fall back to the scalar. Validate that the dual-load tests stay green
and that a fixed-alpha run is reproduced when the adaptive path is disabled.

## Offloading — both checkpoints live in GPU memory at once

The dual backbone loads the specialist *and* the ancestor into one vLLM model, so
both sets of weights sit in GPU memory simultaneously (roughly the footprint of two
models). This is the main resource drawback and caps the model size that fits on a
single GPU. Integrate offloading so the second stream's weights can live on CPU and
be streamed in (or otherwise paged), trading some decode latency for a far smaller
GPU footprint. Keep the dual-load tests green and confirm outputs are unchanged with
offloading on.
