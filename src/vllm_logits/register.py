"""vLLM model-registration shims for the dual backbones.

relevant `register_*` BEFORE constructing `vllm.LLM(...)` (vLLM resolves the
architecture string from `config.architectures` against its ModelRegistry).
"""
from ._compat import ModelRegistry


def register_dual_qwen():
    ModelRegistry.register_model(
        "DualQwen2ForCausalLM",
        "vllm_logits.backbones:DualQwen2ForCausalLM",
    )


def register_dual_llama():
    ModelRegistry.register_model(
        "DualLlamaForCausalLM",
        "vllm_logits.backbones:DualLlamaForCausalLM",
    )


def register_dual_phi3():
    ModelRegistry.register_model(
        "DualPhi3ForCausalLM",
        "vllm_logits.backbones:DualPhi3ForCausalLM",
    )


def register_repair_dual_qwen():
    ModelRegistry.register_model(
        "RepairDualQwen2ForCausalLM",
        "vllm_logits.backbones:RepairDualQwen2ForCausalLM",
    )


def register_all():
    """Register every dual backbone (idempotent)."""
    for fn in (register_dual_qwen, register_dual_llama,
               register_dual_phi3, register_repair_dual_qwen):
        try:
            fn()
        except Exception:
            pass
