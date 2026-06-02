"""src/vllm_logits/processors/logit_repair.py — per-request scan-statistic processor.

vLLM-internal imports routed through `_compat`.

Reads ancestor logits from `backbones._REPAIR_BATCH_LOGITS_A` (set by
RepairDualQwen2ForCausalLM.compute_logits) via the module-level
_CURRENT_REQ_IDX sentinel that apply() updates before each closure call.

Conditions (via SamplingParams.extra_args["repair_mode"]):
    geo_pred  — online scan detection, predicted tau (path/cov)
    geo_wrong — online scan detection, opposite tau
    rand      — fire at pre-selected random position and tau
    dense     — mix at every position (post-hoc pass)
    local_temp — scan detection, but at fire samples specialist at T_local
retry is handled outside (SamplingParams(temperature=T) with no repair_mode).

**continuation_mode** (extra_args["continuation_mode"], default "temperature"):
    temperature — post-fire suffix decodes at `logit_S / T_cont` (T_cont default
                  = T, == retry baseline): PRODUCTION recoverability.
    greedy      — post-fire suffix decodes at `logit_S × 1000` (argmax): the
                  attribution lower bound (rescue ⇒ the steering act).
The knob swaps ONLY the post-fire suffix decode; it is operator-independent
(the at-fire behavior per operator is unchanged). It also governs the dense
condition's per-step decode (which is itself a continuous intervention).

State IPC: the closure runs in the vLLM EngineCore subprocess. After fire,
state is written to VLLM_LOGITS_STATE_DIR/{req_key}.json so the main process can read it.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from .. import _compat
from ..scoring import compute_scores_at_t

AdapterLogitsProcessor = _compat.AdapterLogitsProcessor
SamplingParams = _compat.SamplingParams
RequestLogitsProcessor = _compat.RequestLogitsProcessor

# ── Module-level sentinel: set by apply() before each per-request closure call ─
_CURRENT_REQ_IDX: int = 0

# Multiplier used to force greedy (argmax) selection from a target distribution.
_GREEDY_SCALE = 1000.0


def _get_ancestor_logits(req_idx: int) -> Optional[torch.Tensor]:
    """Read ancestor logits from the RepairDualQwen module-level buffer at req_idx."""
    try:
        from .. import backbones as _bb
        if _bb._REPAIR_BATCH_LOGITS_A is None:
            return None
        return _bb._REPAIR_BATCH_LOGITS_A[req_idx].float()
    except (ImportError, IndexError, AttributeError):
        return None


def _mix_logit(logit_S: torch.Tensor, logit_A: torch.Tensor,
               tau: str, alpha: float, T: float = 1.0) -> torch.Tensor:
    """e-geodesic (tau='path') or m-geodesic (tau='cov') logit mixing."""
    if tau == "path":
        return alpha * logit_S + (1.0 - alpha) * logit_A
    else:
        pS = F.softmax(logit_S / T, dim=-1)
        pA = F.softmax(logit_A / T, dim=-1)
        return torch.log((alpha * pS + (1.0 - alpha) * pA).clamp_min(1e-12))


class LogitRepairProcessor(AdapterLogitsProcessor):
    """Per-request scan-statistic logit processor.

    Overrides apply() to inject req_idx into _CURRENT_REQ_IDX before each
    closure call so closures can look up the correct ancestor logit row.
    """

    def is_argmax_invariant(self) -> bool:
        return False

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        global _CURRENT_REQ_IDX
        if self.req_info:
            for req_idx, req_lp in self.req_info.items():
                _CURRENT_REQ_IDX = req_idx
                req_logits = logits[req_idx]
                new_logits = req_lp(req_logits)
                logits[req_idx] = new_logits
        return logits

    def new_req_logits_processor(self, params: SamplingParams) -> Optional[RequestLogitsProcessor]:
        if not params.extra_args or "repair_mode" not in params.extra_args:
            return None

        ea = params.extra_args
        mode       = ea["repair_mode"]
        background = ea["background"]
        alpha      = float(ea.get("alpha", 0.7))
        # TODO(adaptive-alpha): make the logit-steering alpha per-token instead of a
        # fixed scalar. Compute alpha_t at the fired position from the local spec/anc
        # logits via the signals already in alpha.py (entropy_gap / chi2_divergence ->
        # adaptive_alpha), so sharper-specialist tokens get more ancestor weight. See
        # TODO.md.
        T          = float(ea.get("T", 0.6))
        top_k_kl   = int(ea.get("top_k_kl", 100))
        top_k_cov  = int(ea.get("top_k_cov", 20))
        warmup     = int(ea.get("warmup", 20))
        w          = int(ea.get("w", 1))
        req_key    = ea["req_key"]
        max_new_tokens = int(ea.get("max_new_tokens", 1024))
        lambda_J   = background["lambda_J"]

        # ── continuation_mode: post-fire suffix decode policy ──────────────────
        continuation_mode = ea.get("continuation_mode", "temperature")
        # T_cont defaults to T (== retry baseline) for fair apples-to-apples.
        T_cont = float(ea.get("T_cont", T))

        def _continue(logit_S_curr: torch.Tensor, dtype) -> torch.Tensor:
            """Post-fire / dense continuation decode per continuation_mode."""
            if continuation_mode == "greedy":
                return (logit_S_curr * _GREEDY_SCALE).to(dtype)
            # temperature (production): sample specialist at T_cont
            return (logit_S_curr / T_cont).to(dtype)

        rand_t:    Optional[int] = ea.get("rand_t")
        rand_tau:  Optional[str] = ea.get("rand_tau")
        dense_tau: Optional[str] = ea.get("dense_tau")
        t_force:   Optional[int] = ea.get("t_force")
        tau_force: Optional[str] = ea.get("tau_force")
        T_local:   float         = float(ea.get("T_local", 1.0))

        state: dict = {
            "t": 0,
            "prev_logit_S": None,
            "prev_logit_A": None,
            "J_window": [],
            "fired": False,
            "t_hat": None,
            "tau_hat": None,
            "fire_score": None,
            "entropy_sensitive": None,
        }
        state_dir  = os.environ.get("VLLM_LOGITS_STATE_DIR", "/tmp")
        state_path = os.path.join(state_dir, f"{req_key}.json")

        def processor(prompt_ids: List[int], output_ids: List[int],
                      logits_V: torch.Tensor) -> torch.Tensor:
            t = state["t"]
            logit_S_curr = logits_V.float()
            logit_A_curr = _get_ancestor_logits(_CURRENT_REQ_IDX)
            if logit_A_curr is None:
                logit_A_curr = logit_S_curr  # graceful fallback

            # Dense: mix at every step. At-fire mix is greedy-of-mix; the
            # continuation policy governs the SUFFIX after each mixed token —
            # but dense mixes at every position, so the mix itself is the decode.
            # We keep the at-mix greedy (the operator definition) and let
            # continuation_mode only differentiate the (non-existent) post-fire
            # region; for dense the per-step mix decode is unchanged.
            if mode == "dense":
                state["t"] += 1
                state["prev_logit_S"] = logit_S_curr
                state["prev_logit_A"] = logit_A_curr
                mixed = _mix_logit(logit_S_curr, logit_A_curr, dense_tau, alpha, T)
                return (mixed * _GREEDY_SCALE).to(logits_V.dtype)

            # Post-intervention: continuation per continuation_mode.
            if state["fired"]:
                state["t"] += 1
                state["prev_logit_S"] = logit_S_curr
                state["prev_logit_A"] = logit_A_curr
                return _continue(logit_S_curr, logits_V.dtype)

            # Score previous step (one-step lag): x_{t-1} in output_ids[-1]
            sc = None
            S_t = 0.0
            if t >= 1 and state["prev_logit_S"] is not None and output_ids:
                x_prev = int(output_ids[-1])
                sc = compute_scores_at_t(
                    state["prev_logit_S"], state["prev_logit_A"], x_prev,
                    background, T=T, top_k_kl=top_k_kl, top_k_cov=top_k_cov,
                )
                state["J_window"].append(sc["J_t"])
                if len(state["J_window"]) > w:
                    state["J_window"].pop(0)
                S_t = sum(state["J_window"])

            # Fire condition
            fire = False
            if mode == "rand":
                fire = (t == rand_t)
            elif mode in ("geo_pred", "geo_wrong", "local_temp"):
                if t_force is not None:
                    fire = (t == t_force)
                elif sc is not None:
                    fire = (len(state["J_window"]) == w
                            and S_t > lambda_J
                            and (t - 1) >= warmup)

            if fire:
                state["t_hat"] = t if (t_force is not None or sc is None) else (t - 1)

                if mode == "rand":
                    state["tau_hat"] = rand_tau
                elif mode == "geo_wrong" and sc is not None:
                    state["tau_hat"] = "cov" if sc["tau_t"] == "path" else "path"
                elif tau_force is not None:
                    state["tau_hat"] = tau_force
                elif sc is not None:
                    state["tau_hat"] = sc["tau_t"]
                else:
                    state["tau_hat"] = "path"

                if sc is not None:
                    state["fire_score"] = {
                        "Z_path": sc["Z_path"], "Z_cov": sc["Z_cov"],
                        "J_t": sc["J_t"], "S_t": float(S_t),
                        "t": state["t_hat"],
                        "t_frac": state["t_hat"] / max(1, max_new_tokens),
                        "pS_on_set": sc["pS_on_set"], "pA_on_set": sc["pA_on_set"],
                    }
                    state["entropy_sensitive"] = sc["entropy_sensitive"]

                state["fired"] = True
                try:
                    with open(state_path, "w") as _f:
                        json.dump({
                            "fired": True,
                            "t_hat": state["t_hat"],
                            "tau_hat": state["tau_hat"],
                            "fire_score": state["fire_score"],
                            "entropy_sensitive": state["entropy_sensitive"],
                        }, _f)
                except Exception:
                    pass
                state["t"] += 1
                state["prev_logit_S"] = logit_S_curr
                state["prev_logit_A"] = logit_A_curr

                if mode == "local_temp":
                    # Temperature-only intervention: specialist at T_local, no ancestor.
                    return (logit_S_curr / T_local).to(logits_V.dtype)
                mixed = _mix_logit(logit_S_curr, logit_A_curr, state["tau_hat"], alpha, T)
                return (mixed * _GREEDY_SCALE).to(logits_V.dtype)

            # Pre-fire: sample at temperature T (carrier=1.0 in SamplingParams)
            state["t"] += 1
            state["prev_logit_S"] = logit_S_curr
            state["prev_logit_A"] = logit_A_curr
            return (logit_S_curr / T).to(logits_V.dtype)

        return processor


# Backwards-compatible alias (the repo class name).
