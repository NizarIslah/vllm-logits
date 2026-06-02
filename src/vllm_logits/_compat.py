"""vLLM-internals firewall.

Every import that reaches into vLLM's private model_executor / v1 sampling
internals lives HERE so that a vLLM version bump is a single-file diff. The
backbones and processors import their vLLM dependencies through this module
(or via the small accessor functions below) rather than reaching into
`vllm.model_executor...` directly.

Tested vLLM range: >=0.15,<0.16 (pinned today at 0.15.1). The arch-fix
regression tests (tests/test_dual_load_*.py) are the canaries for a bump:
if vLLM moves these symbols, those tests fail loudly with the exact dual-load
greedy outputs that the backbones must reproduce.
"""
from __future__ import annotations

# ── Model registry (used by register.py) ──────────────────────────────────────
from vllm.model_executor.models.registry import ModelRegistry  # noqa: F401

# ── Logits processor base + sampling params (generic processors) ───────────────
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor  # noqa: F401
from vllm.sampling_params import SamplingParams  # noqa: F401
from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor  # noqa: F401
from vllm.config import VllmConfig  # noqa: F401


# ── Backbone class accessors (imported lazily inside backbones to keep import of
#    this module cheap, but kept here so the touchpoints are enumerable) ─────────

def get_backbone_cls(arch: str):
    """Return the vLLM backbone *Model* class for a model_type / arch string.

    arch is `hf_config.model_type` (qwen2 / qwen3 / olmo2 / olmo3 / phi3 / llama).
    Phi-3 / Phi-4-mini use LlamaModel as backbone (vLLM's phi3 module imports
    LlamaForCausalLM and only overrides packed_modules_mapping); HF state dict
    ships qkv_proj pre-stacked while gate/up are still split.
    """
    if arch == "qwen3":
        from vllm.model_executor.models.qwen3 import Qwen3Model as Backbone
    elif arch in ("olmo2", "olmo3"):
        from vllm.model_executor.models.olmo2 import Olmo2Model as Backbone
    elif arch == "phi3":
        from vllm.model_executor.models.llama import LlamaModel as Backbone
    elif arch == "llama":
        from vllm.model_executor.models.llama import LlamaModel as Backbone
    else:
        from vllm.model_executor.models.qwen2 import Qwen2Model as Backbone
    return Backbone


def get_parallel_lm_head():
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    return ParallelLMHead


def get_logits_processor_layer():
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    return LogitsProcessor


def get_weight_iterators():
    from vllm.model_executor.model_loader.weight_utils import (
        pt_weights_iterator, safetensors_weights_iterator)
    return pt_weights_iterator, safetensors_weights_iterator


def get_default_weight_loader():
    import vllm.model_executor.model_loader.weight_utils as _wu
    return _wu.default_weight_loader
