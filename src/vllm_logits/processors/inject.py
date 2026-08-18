"""Generic logit-injection processor (vLLM V1).

vLLM-internal imports
routed through `_compat`.

Injects ancestor (shunt) logits at chosen positions [t_start, t_start+w),
supporting LogitMix, ProbMix, adaptive mixing, temperature scaling, top-k/p,
min-p, and repetition-penalty modes. Per-request config flows through
`SamplingParams.extra_args["inject_*"]`.
"""
import os
import traceback
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from .. import _compat
from ..alpha import entropy_gap, chi2_divergence, adaptive_alpha
from ..storage import LogitStore

AdapterLogitsProcessor = _compat.AdapterLogitsProcessor
SamplingParams = _compat.SamplingParams
RequestLogitsProcessor = _compat.RequestLogitsProcessor


class InjectionLogitsProcessor(AdapterLogitsProcessor):
    """vLLM V1 adapter for robust per-request logit injection.

    Supports: LogitMix, ProbMix, Temperature Scaling, Top-K/P filtering.
    Guarantees: float32 precision for math, fault tolerance.
    """

    def __init__(self, vllm_config: Any, device: torch.device, is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory)
        store_path = os.environ.get("LOGIT_INJECT_STORE_PATH")
        if store_path:
            self.store = LogitStore(store_path)
        else:
            self.store = None

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams) -> Optional[RequestLogitsProcessor]:
        if not params.extra_args or "inject_task" not in params.extra_args:
            return None

        task = params.extra_args["inject_task"]
        detector_id = params.extra_args.get("detector_id", "default")
        mode = params.extra_args.get("inject_mode", "logit_mix")
        alpha = params.extra_args.get("inject_alpha", 0.9)
        adaptive_T = params.extra_args.get("adaptive_T", 1.0)
        temp = params.extra_args.get("inject_temp", 1.0)
        min_p_val = params.extra_args.get("inject_min_p", 0.0)
        top_p_val = params.extra_args.get("inject_top_p", 1.0)
        reppen_val = params.extra_args.get("inject_reppen", 1.0)
        # `inject_top_k` was read here but never applied, so setting it silently did nothing.
        # Fail loudly instead: min_p / top_p / repetition penalty are supported at the injected
        # position, top-k is not. Implementing it belongs with a GPU to verify against.
        if params.extra_args.get("inject_top_k") is not None:
            raise ValueError(
                "inject_top_k is not supported at the injected position (it was previously "
                "accepted and ignored). Use inject_top_p or inject_min_p, or open an issue if "
                "you need top-k filtering here."
            )

        # Junction timing: [t_start, t_start + w)
        t_start = task.get("step", 0)
        w = params.extra_args.get("window_size", 1)

        # Pre-load shunt logits for the intervention window to avoid per-token disk I/O
        z_a_win = None
        if mode in ["logit_mix", "prob_mix", "adaptive_logit_mix", "adaptive_prob_mix"] and self.store:
            key = f"{task['problem_id']}_{task['rollout_id']}_{task['step_abs']}"
            z_a_win, _, _ = self.store.load_logits_with_prefix(detector_id, key)

        is_sampling_mode = params.extra_args.get("inject_sampling_mode", False)

        def processor(prompt_ids: List[int], output_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
            try:
                current_step = len(output_ids)

                if t_start <= current_step < t_start + w:
                    z_s = logits.to(torch.float32)

                    if not is_sampling_mode and temp > 0 and temp != 1.0:
                        z_s = z_s / temp

                    z_a = None
                    if z_a_win is not None:
                        w_idx = current_step - t_start
                        if z_a_win.dim() > 1:
                            if w_idx < z_a_win.shape[0]:
                                z_a = z_a_win[w_idx]
                        else:
                            z_a = z_a_win

                        if z_a is not None:
                            z_a = z_a.to(logits.device).to(torch.float32)

                            if z_a.shape[-1] != z_s.shape[-1]:
                                if z_a.shape[-1] > z_s.shape[-1]:
                                    z_a = z_a[..., :z_s.shape[-1]]
                                else:
                                    new_z_a = torch.full_like(z_s, -float("inf"))
                                    new_z_a[..., :z_a.shape[-1]] = z_a
                                    z_a = new_z_a

                            if not is_sampling_mode and temp > 0 and temp != 1.0:
                                z_a = z_a / temp

                    if mode == "logit_mix" and z_a is not None:
                        z_mix = alpha * z_s + (1.0 - alpha) * z_a
                        return z_mix.to(logits.dtype)

                    elif mode == "prob_mix" and z_a is not None:
                        p_s = F.softmax(z_s, dim=-1)
                        p_a = F.softmax(z_a, dim=-1)
                        p_mix = alpha * p_s + (1.0 - alpha) * p_a
                        return torch.log(p_mix + 1e-12).to(logits.dtype)

                    elif mode == "adaptive_logit_mix" and z_a is not None:
                        gap = entropy_gap(z_s.unsqueeze(0), z_a.unsqueeze(0)).squeeze(0)
                        a_t = adaptive_alpha(-gap, T=adaptive_T).squeeze(-1)
                        z_mix = a_t * z_s + (1.0 - a_t) * z_a
                        return z_mix.to(logits.dtype)

                    elif mode == "adaptive_prob_mix" and z_a is not None:
                        chi2 = chi2_divergence(z_s.unsqueeze(0), z_a.unsqueeze(0)).squeeze(0)
                        a_t = adaptive_alpha(-chi2, T=adaptive_T).squeeze(-1)
                        p_s = F.softmax(z_s, dim=-1)
                        p_a = F.softmax(z_a, dim=-1)
                        p_mix = a_t * p_s + (1.0 - a_t) * p_a
                        return torch.log(p_mix + 1e-10).to(logits.dtype)

                    elif mode == "min_p":
                        p_s = F.softmax(z_s, dim=-1)
                        max_p = torch.max(p_s)
                        threshold = max_p * min_p_val
                        z_s[p_s < threshold] = -float("inf")
                        return z_s.to(logits.dtype)

                    elif mode == "top_p":
                        p_s = F.softmax(z_s, dim=-1)
                        sorted_probs, sorted_indices = torch.sort(p_s, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p_val
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        z_s[indices_to_remove] = -float("inf")
                        return z_s.to(logits.dtype)

                    elif mode == "reppen":
                        for token_id in set(prompt_ids + output_ids):
                            if z_s[token_id] > 0:
                                z_s[token_id] /= reppen_val
                            else:
                                z_s[token_id] *= reppen_val
                        return z_s.to(logits.dtype)

                    return z_s.to(logits.dtype)

                else:
                    # OUTSIDE the junction window
                    if is_sampling_mode:
                        return logits
                    else:
                        # Force greedy: keep intervention strictly local.
                        return (logits * 100.0).to(logits.dtype)

            except Exception as e:
                print(f"[InjectionProcessor Error] {str(e)}")
                traceback.print_exc()

            return logits

        return processor
