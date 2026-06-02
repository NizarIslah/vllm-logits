"""Capturing logits processor for vLLM V1 proxy tuning.

`src/vllm_plugins/proxy_tuning_logits_processor.py`. vLLM-internal imports
routed through `_compat`.

Captures pre-sample logits into a module-level dict keyed by (model_name, tag).
The driver loop reads from this dict after each generate(max_tokens=1) call on
each engine and blends externally (expert − base logit arithmetic).
"""
from __future__ import annotations

from typing import Optional

import torch

from .. import _compat

AdapterLogitsProcessor = _compat.AdapterLogitsProcessor
SamplingParams = _compat.SamplingParams
VllmConfig = _compat.VllmConfig


# Module-global capture store. Keyed by (model_name, request_tag).
CAPTURE_STORE: dict[tuple[str, str], torch.Tensor] = {}


def _make_capture_callable(model_name: str, request_tag: str):
    """V0-style RequestLogitsProcessor: `(token_ids, logits) -> logits`.
    Stores a CPU copy into CAPTURE_STORE; returns logits unchanged.
    """
    def _cb(token_ids, logits):  # noqa: ARG001
        CAPTURE_STORE[(model_name, request_tag)] = logits.detach().to(
            "cpu", copy=True
        )
        return logits
    return _cb


class CapturingAdapter(AdapterLogitsProcessor):
    """vLLM V1 logits-processor adapter.

    Reads the per-request `pt_request_tag` from `SamplingParams.extra_args` and
    stores the engine's pre-sample logits into CAPTURE_STORE keyed by
    `(model_name, request_tag)`.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory)
        self.model_name = vllm_config.model_config.model

    def is_argmax_invariant(self) -> bool:
        return True  # we do not modify logits

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> Optional[object]:
        extra = getattr(params, "extra_args", None) or {}
        tag = extra.get("pt_request_tag")
        if tag is None:
            return None  # not a proxy-tuning request; skip
        return _make_capture_callable(self.model_name, tag)
