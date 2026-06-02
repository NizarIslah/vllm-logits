"""Dual-model backbone classes for zero-overhead logit mixing in vLLM.



Each `DualX...ForCausalLM` loads a specialist (model B) + ancestor (model A)
backbone into ONE vLLM model and mixes their logits before the lm_head. Mixing
at the final hidden state equals logit-level mixing because lm_head is a pure
linear projection.

The dot-anchored `stacked_params_mapping` (".v_proj" not "v_proj") is LOAD-
BEARING — without the leading dot, "v_proj" is a substring of "qkv_proj" and the
name.replace corrupts pre-stacked Phi-3 weights. DO NOT "clean up" that mapping.
The arch-fix tests (tests/test_dual_load_*.py) guard it.

vLLM-internal imports are routed through `_compat` (the version-bump firewall).

Configuration is via environment variables (set before vLLM forks workers):
    VLLM_LOGIT_MIX_ANCESTOR    local path to ancestor weights
    VLLM_LOGIT_MIX_SPECIALIST  local path to specialist weights
    VLLM_LOGIT_MIX_ALPHA       specialist weight (0=anc, 1=spec, else mix)
    VLLM_LOGIT_MIX_A_LAYERS    ancestor layer count (split point)
    VLLM_LOGIT_MIX_MODE        logit | prob | adaptive_logit | adaptive_prob
    VLLM_LOGIT_MIX_TEMP        temp for prob-mode log-mean-exp
    VLLM_LOGIT_MIX_PROTECT_IDS comma ids always taken from specialist (EOS)
"""
import copy
import glob
import itertools
import os
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alpha import entropy_gap, chi2_divergence, adaptive_alpha as _adaptive_alpha
from . import _compat


class _ShiftedLayerList(nn.Module):
    """Drop-in replacement for nn.ModuleList that registers children with
    shifted integer keys so vLLM's extract_layer_index() assigns unique
    KV cache slots.
    """

    def __init__(self, layers, offset):
        super().__init__()
        self._layers = list(layers)
        for i, layer in enumerate(self._layers):
            self.add_module(str(i + offset), layer)

    def __len__(self):
        return len(self._layers)

    def __iter__(self):
        return iter(self._layers)

    def __getitem__(self, idx):
        return self._layers[idx]


class DualQwen2ForCausalLM(nn.Module):
    """Two Qwen2/Qwen3 (or olmo2/3, phi3) backbones with hidden-state mixing.

    Despite the name, arch is auto-selected from `hf_config.model_type`; this
    one class covers qwen2/qwen3/olmo2/olmo3/phi3 (phi3 uses LlamaModel backbone),
    which is why the phi4 arch-fix test loads phi4 weights through this class.
    """

    _PROTECT_DEFAULT = "151645"  # Qwen3 <|im_end|>

    def __init__(self, vllm_config, prefix=""):
        super().__init__()

        self.vllm_config = vllm_config
        hf_config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        protect_str = os.environ.get("VLLM_LOGIT_MIX_PROTECT_IDS", self._PROTECT_DEFAULT)
        self._protect_ids = [int(x) for x in protect_str.split(",") if x.strip()]

        total_layers = hf_config.num_hidden_layers
        offset_a = int(os.environ.get("VLLM_LOGIT_MIX_A_LAYERS", total_layers // 2))

        layers_a = offset_a
        layers_b = total_layers - offset_a

        print(f"{type(self).__name__}: heterogeneous layers: "
              f"Model A (Ancestor)={layers_a}, Model B (Specialist)={layers_b}")

        sub_vllm_config_a = copy.deepcopy(vllm_config)
        sub_vllm_config_a.model_config.hf_config.num_hidden_layers = layers_a

        sub_vllm_config_b = copy.deepcopy(vllm_config)
        sub_vllm_config_b.model_config.hf_config.num_hidden_layers = layers_b

        arch = getattr(self, "_force_arch", None) or getattr(hf_config, "model_type", "qwen2")
        Backbone = _compat.get_backbone_cls(arch)

        self.model_a = Backbone(vllm_config=sub_vllm_config_a, prefix=prefix + ".a")
        self.model_b = Backbone(vllm_config=sub_vllm_config_b, prefix=prefix + ".b")
        self.layers_a = layers_a

        self._remap_layer_indices(self.model_a, 0)
        self._remap_layer_indices(self.model_b, layers_a)

        if getattr(hf_config, "tie_word_embeddings", False):
            self.lm_head_a = self.model_a.embed_tokens
            self.lm_head_b = self.model_b.embed_tokens
        else:
            ParallelLMHead = _compat.get_parallel_lm_head()
            self.lm_head_a = ParallelLMHead(
                hf_config.vocab_size, hf_config.hidden_size,
                quant_config=quant_config, prefix=prefix + ".lm_head_a",
            )
            self.lm_head_b = ParallelLMHead(
                hf_config.vocab_size, hf_config.hidden_size,
                quant_config=quant_config, prefix=prefix + ".lm_head_b",
            )

        LogitsProcessor = _compat.get_logits_processor_layer()
        self.logits_processor = LogitsProcessor(
            hf_config.vocab_size, hf_config.vocab_size, logits_as_input=False,
        )

    @staticmethod
    def _remap_layer_indices(model, offset):
        old_layers = model.layers
        wrapped = _ShiftedLayerList(old_layers, offset)
        model.layers = wrapped

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model_a.embed_tokens(input_ids)

    def forward(self, input_ids=None, positions=None, intermediate_tensors=None, **kwargs):
        alpha = float(os.environ.get("VLLM_LOGIT_MIX_ALPHA", "0.5"))
        hidden_a = self.model_a(
            input_ids=input_ids, positions=positions,
            intermediate_tensors=intermediate_tensors, **kwargs,
        )
        if alpha == 0.0:
            return hidden_a
        hidden_b = self.model_b(
            input_ids=input_ids, positions=positions,
            intermediate_tensors=intermediate_tensors, **kwargs,
        )
        if alpha == 1.0:
            return hidden_b
        return torch.cat([hidden_a, hidden_b], dim=-1)

    def compute_logits(self, hidden_states, sampling_metadata=None, **kwargs):
        alpha = float(os.environ.get("VLLM_LOGIT_MIX_ALPHA", "0.5"))
        mode = os.environ.get("VLLM_LOGIT_MIX_MODE", "logit")

        hidden_size = self.lm_head_a.weight.shape[1]
        is_mixed = hidden_states.shape[-1] == hidden_size * 2

        if alpha == 0.0:
            return self.logits_processor(self.lm_head_a, hidden_states)
        elif alpha == 1.0:
            return self.logits_processor(self.lm_head_b, hidden_states)
        elif not is_mixed:
            return self.logits_processor(self.lm_head_a, hidden_states)

        hidden_a = hidden_states[..., :hidden_size]
        hidden_b = hidden_states[..., hidden_size:]

        logits_a = self.logits_processor(self.lm_head_a, hidden_a)
        logits_b = self.logits_processor(self.lm_head_b, hidden_b)

        # Vocabulary mismatch: specialist (b) is reference for output size.
        v_s = logits_b.shape[-1]
        v_a = logits_a.shape[-1]
        if v_s != v_a:
            min_v = min(v_s, v_a)
            l_a_aligned = torch.full_like(logits_b, -float("inf"))
            l_a_aligned[..., :min_v] = logits_a[..., :min_v]
            logits_a = l_a_aligned

        if mode == "prob":
            temp = float(os.environ.get("VLLM_LOGIT_MIX_TEMP", "1.0"))
            if temp < 1e-5:
                mixed = (1.0 - alpha) * logits_a + alpha * logits_b
                if self._protect_ids:
                    mixed[..., self._protect_ids] = logits_b[..., self._protect_ids]
                return mixed.to(logits_a.dtype)
            p_a = torch.softmax(logits_a.float() / temp, dim=-1)
            p_b = torch.softmax(logits_b.float() / temp, dim=-1)
            p_mix = (1.0 - alpha) * p_a + alpha * p_b
            if self._protect_ids:
                p_mix[..., self._protect_ids] = p_b[..., self._protect_ids]
            return (temp * torch.log(p_mix.clamp_min(1e-12))).to(logits_a.dtype)

        elif mode == "adaptive_logit":
            adaptive_T = float(os.environ.get("VLLM_LOGIT_MIX_ADAPTIVE_T", "1.0"))
            base_alpha = float(os.environ.get("VLLM_LOGIT_MIX_BASE_ALPHA", "0.7"))
            gap = entropy_gap(logits_b, logits_a)
            a_t = _adaptive_alpha(-gap, T=adaptive_T, base_alpha=base_alpha)
            mixed = a_t * logits_b + (1.0 - a_t) * logits_a
            if self._protect_ids:
                mixed[..., self._protect_ids] = logits_b[..., self._protect_ids]
            return mixed.to(logits_a.dtype)

        elif mode == "adaptive_prob":
            temp = float(os.environ.get("VLLM_LOGIT_MIX_TEMP", "1.0"))
            adaptive_T = float(os.environ.get("VLLM_LOGIT_MIX_ADAPTIVE_T", "1.0"))
            base_alpha = float(os.environ.get("VLLM_LOGIT_MIX_BASE_ALPHA", "0.7"))
            chi2 = chi2_divergence(logits_b, logits_a)
            a_t = _adaptive_alpha(-chi2, T=adaptive_T, base_alpha=base_alpha)
            if temp < 1e-5:
                mixed = a_t * logits_b + (1.0 - a_t) * logits_a
                if self._protect_ids:
                    mixed[..., self._protect_ids] = logits_b[..., self._protect_ids]
                return mixed.to(logits_a.dtype)
            p_b = F.softmax(logits_b.float() / temp, dim=-1)
            p_a = F.softmax(logits_a.float() / temp, dim=-1)
            p_mix = a_t * p_b + (1.0 - a_t) * p_a
            if self._protect_ids:
                p_mix[..., self._protect_ids] = p_b[..., self._protect_ids]
            return (temp * torch.log(p_mix.clamp_min(1e-12))).to(logits_a.dtype)

        else:  # logit (default)
            mixed = (1.0 - alpha) * logits_a + alpha * logits_b
            if self._protect_ids:
                mixed[..., self._protect_ids] = logits_b[..., self._protect_ids]
            return mixed.to(logits_a.dtype)

    def sample(self, logits, sampling_metadata):
        return None

    # Sub-prefix for the per-model weight prefix ("a"/"b" for Qwen, "model_a"/"model_b"
    # for Llama/Phi3). lm_head_{prefix[-1]} resolves to lm_head_a / lm_head_b either way.
    _PREFIX_A = "model_a"
    _PREFIX_B = "model_b"
    _SKIP_NAMES = ("rotary_emb.inv_freq",)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        pt_iter, st_iter = _compat.get_weight_iterators()
        default_loader = _compat.get_default_weight_loader()

        ancestor_path = os.environ.get("VLLM_LOGIT_MIX_ANCESTOR")
        specialist_path = os.environ.get("VLLM_LOGIT_MIX_SPECIALIST")

        print(f"{type(self).__name__}: Loading ancestor from {ancestor_path}")
        print(f"{type(self).__name__}: Loading specialist from {specialist_path}")

        def _yield_weights(path, prefix, offset):
            safe_files = glob.glob(os.path.join(path, "*.safetensors"))
            if len(safe_files) > 0:
                iterator = st_iter(safe_files, False)
            else:
                pt_files = glob.glob(os.path.join(path, "*.bin"))
                iterator = pt_iter(pt_files, False)

            for name, tensor in iterator:
                if name == "lm_head.weight":
                    yield f"lm_head_{prefix[-1]}.weight", tensor
                elif name.startswith("model."):
                    new_name = name.replace("model.", f"{prefix}.", 1)
                    if ".layers." in new_name:
                        parts = new_name.split(".")
                        try:
                            l_idx = int(parts[2])
                            parts[2] = str(l_idx + offset)
                            new_name = ".".join(parts)
                        except ValueError:
                            pass
                    yield new_name, tensor
                else:
                    pass

        all_weights = itertools.chain(
            _yield_weights(ancestor_path, self._PREFIX_A, 0),
            _yield_weights(specialist_path, self._PREFIX_B, self.layers_a),
        )

        params_dict = dict(self.named_parameters())

        # LOAD-BEARING leading-dot anchors — see module docstring.
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        loaded_params: set = set()
        for name, loaded_weight in all_weights:
            if any(s in name for s in self._SKIP_NAMES):
                continue

            mapped = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_loader)
                if weight_loader == default_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                mapped = True
                break

            if not mapped:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return {name for name, _ in self.named_parameters()}


class DualLlamaForCausalLM(DualQwen2ForCausalLM):
    """Two Llama backbones with logit-level mixing."""

    _PROTECT_DEFAULT = "128001,128009"  # Llama-3 EOS ids
    _SKIP_NAMES = ("rotary_emb.inv_freq", "rotary_emb.cos_cached", "rotary_emb.sin_cached")

    def __init__(self, vllm_config, prefix=""):
        # Llama config reports model_type "llama"; the default _compat branch
        # picks Qwen2Model, so force the llama backbone explicitly.
        self._force_arch = "llama"
        super().__init__(vllm_config, prefix=prefix)


class DualPhi3ForCausalLM(DualQwen2ForCausalLM):
    """Two Phi-3 / Phi-4-mini backbones (LlamaModel backbone, pre-stacked qkv)."""

    _PROTECT_DEFAULT = "199999,200020"  # Phi-4-mini EOS / endoftext ids
    _SKIP_NAMES = ("rotary_emb.inv_freq", "rotary_emb.cos_cached", "rotary_emb.sin_cached")


# ── online-scan repair variant ────────────────────────────────────────────────────
# Module-level buffers written by RepairDualQwen2ForCausalLM.compute_logits and read
# by the repair logits processor via the req_idx in its overridden apply().
_REPAIR_BATCH_LOGITS_A: Optional[torch.Tensor] = None   # ancestor
_REPAIR_BATCH_LOGITS_S: Optional[torch.Tensor] = None   # specialist


class RepairDualQwen2ForCausalLM(DualQwen2ForCausalLM):
    """DualQwen variant exposing (logits_a, logits_b) via module-level buffers.

    Inherits all weight loading / construction / forward from the parent; only
    compute_logits is overridden so the repair processor can read ancestor logits
    per request without the 2V shape hack.

    Thread safety: safe for single-GPU vLLM because compute_logits and the
    processor's apply() run sequentially in the same worker process.
    """

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata=None, **kwargs) -> torch.Tensor:
        global _REPAIR_BATCH_LOGITS_A, _REPAIR_BATCH_LOGITS_S

        hidden_size = self.lm_head_a.weight.shape[1]
        is_mixed = hidden_states.shape[-1] == hidden_size * 2

        if not is_mixed:
            _REPAIR_BATCH_LOGITS_A = None
            _REPAIR_BATCH_LOGITS_S = None
            return self.logits_processor(self.lm_head_a, hidden_states)

        hidden_a = hidden_states[..., :hidden_size]
        hidden_b = hidden_states[..., hidden_size:]

        logits_a = self.logits_processor(self.lm_head_a, hidden_a)  # ancestor
        logits_b = self.logits_processor(self.lm_head_b, hidden_b)  # specialist

        _REPAIR_BATCH_LOGITS_A = logits_a.float()
        _REPAIR_BATCH_LOGITS_S = logits_b.float()

        return logits_b  # specialist; processor overrides per request
