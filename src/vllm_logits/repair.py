"""src/vllm_logits/repair.py — vLLM-native logit-repair engine (repair_logits).


Uses RepairDualQwen2ForCausalLM (Frankenstein dual forward pass) +
LogitRepairProcessor to run all repair conditions in batched llm.generate() calls.

Conditions: retry / rand / geo_pred / geo_wrong (+ optional local_temp per
temperature). Dense via run_dense_batch(). The **continuation_mode** knob
(temperature | greedy) is forwarded to every repair request uniformly.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

from .register import register_repair_dual_qwen
from .scoring import estimate_background  # noqa: F401  (re-exported convenience)
from .processors.logit_repair import LogitRepairProcessor

register_repair_dual_qwen()  # registers RepairDualQwen2ForCausalLM in this process

_ABLATION_OFFSETS = (-10, -5, 0, 5, 10)
_CONDITIONS = ("retry", "rand", "geo_pred", "geo_wrong")


class LogitRepairEngine:
    """Batched vLLM-native logit-repair engine.

    Weight loading: RepairDualQwen2ForCausalLM reads specialist + ancestor via
    VLLM_LOGIT_MIX_ANCESTOR / VLLM_LOGIT_MIX_SPECIALIST env vars. The Frankenstein
    config (num_hidden_layers×2, or ancestor+specialist) is written to a temp dir.

    continuation_mode: "temperature" (default, production) decodes the post-fire
    suffix at logit_S / T_cont (== retry baseline); "greedy" decodes it at
    logit_S × 1000 (attribution lower bound). Applied uniformly to all
    operators.
    """

    def __init__(
        self,
        specialist_path: str,
        ancestor_path: str,
        T: float = 0.6,
        alpha: float = 0.7,
        w: int = 1,
        q_J: float = 0.99,
        q_V: float = 0.75,
        top_k_kl: int = 100,
        top_k_cov: int = 20,
        warmup: int = 20,
        max_new_tokens: int = 1024,
        seed: int = 42,
        gpu_memory_utilization: float = 0.85,
        temp_interventions: Optional[List[float]] = None,
        ancestor_layers: Optional[int] = None,
        specialist_layers: Optional[int] = None,
        sampling: Optional[Dict[str, Any]] = None,
        continuation_mode: str = "temperature",
        T_cont: Optional[float] = None,
    ) -> None:
        self.T = T
        self.alpha = alpha
        self.w = w
        self.q_J = q_J
        self.q_V = q_V
        self.top_k_kl = top_k_kl
        self.top_k_cov = top_k_cov
        self.warmup = warmup
        self.max_new_tokens = max_new_tokens
        self.temp_interventions: List[float] = list(temp_interventions or [])
        self.rng = random.Random(seed)
        self.sampling = dict(sampling or {})
        if continuation_mode not in ("temperature", "greedy"):
            raise ValueError(
                f"continuation_mode must be 'temperature' or 'greedy', got {continuation_mode!r}")
        self.continuation_mode = continuation_mode
        self.T_cont = float(T_cont) if T_cont is not None else float(T)

        # Resolve HF model IDs to local cache paths (HF-native; no repo coupling).
        if not os.path.exists(ancestor_path):
            from huggingface_hub import snapshot_download
            ancestor_path = snapshot_download(ancestor_path)
        if not os.path.exists(specialist_path):
            from huggingface_hub import snapshot_download
            specialist_path = snapshot_download(specialist_path)

        # Env vars must be set before LLM() — vLLM forks worker processes at init.
        os.environ["VLLM_LOGIT_MIX_ANCESTOR"]   = ancestor_path
        os.environ["VLLM_LOGIT_MIX_SPECIALIST"]  = specialist_path
        os.environ["VLLM_LOGIT_MIX_ALPHA"]       = "0.5"

        self.state_dir = tempfile.mkdtemp(prefix="repair_states_")
        os.environ["VLLM_LOGITS_STATE_DIR"] = self.state_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            ancestor_path, trust_remote_code=True
        )

        self._tmp = tempfile.TemporaryDirectory()
        tmp_dir = self._tmp.name
        config = AutoConfig.from_pretrained(ancestor_path, trust_remote_code=True)
        config.architectures = ["RepairDualQwen2ForCausalLM"]
        if ancestor_layers is not None and specialist_layers is not None:
            if hasattr(config, "layer_types"):
                anc_types = list(config.layer_types)
                config.layer_types = anc_types + (anc_types * (
                    (specialist_layers + len(anc_types) - 1) // len(anc_types)
                ))[:specialist_layers]
            config.num_hidden_layers = ancestor_layers + specialist_layers
            os.environ["VLLM_LOGIT_MIX_A_LAYERS"] = str(ancestor_layers)
            print(f"[LogitRepairEngine] Heterogeneous dual-load: "
                  f"ancestor={ancestor_layers} + specialist={specialist_layers} "
                  f"= {config.num_hidden_layers} total layers")
        else:
            if hasattr(config, "layer_types"):
                config.layer_types = config.layer_types * 2
            config.num_hidden_layers *= 2

        # Collapse mixed layer_types → single value so vLLM's is_interleaved()
        # gate is bypassed (Olmo3 ships mixed sliding/full attention). No-op for
        # Qwen2 configs. Functionally equivalent at ≤4096-token contexts.
        if hasattr(config, "layer_types") and config.layer_types:
            if len(set(config.layer_types)) > 1:
                config.layer_types = ["full_attention"] * len(config.layer_types)
        if not hasattr(config, "max_window_layers"):
            config.max_window_layers = config.num_hidden_layers
        if not hasattr(config, "use_sliding_window"):
            config.use_sliding_window = False

        config.save_pretrained(tmp_dir)
        self.tokenizer.save_pretrained(tmp_dir)

        for fname in os.listdir(ancestor_path):
            if fname.endswith((".safetensors", ".bin")):
                src = os.path.join(ancestor_path, fname)
                dst = os.path.join(tmp_dir, fname)
                try:
                    os.symlink(src, dst)
                except FileExistsError:
                    pass

        tp = torch.cuda.device_count()
        print(f"[LogitRepairEngine] Initializing with {tp} GPU(s), "
              f"specialist={specialist_path}, ancestor={ancestor_path}, "
              f"continuation_mode={self.continuation_mode}")

        self.llm = LLM(
            model=tmp_dir,
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            tensor_parallel_size=tp,
            disable_custom_all_reduce=True,
            max_num_batched_tokens=32768,
            logits_processors=[LogitRepairProcessor],
        )
        print("[LogitRepairEngine] Engine ready.")

    # ── SamplingParams factories ───────────────────────────────────────────────

    def _sampling_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        for key in ("top_p", "top_k", "min_p",
                    "repetition_penalty", "presence_penalty"):
            if key in self.sampling and self.sampling[key] is not None:
                kwargs[key] = self.sampling[key]
        return kwargs

    def _retry_params(self) -> SamplingParams:
        return SamplingParams(temperature=self.T, max_tokens=self.max_new_tokens,
                              **self._sampling_kwargs())

    def _repair_params(self, mode: str, background: dict, req_key: str,
                    **extra) -> SamplingParams:
        return SamplingParams(
            temperature=1.0,  # carrier; processor handles all scaling
            max_tokens=self.max_new_tokens,
            **self._sampling_kwargs(),
            extra_args={
                "repair_mode": mode,
                "background": background,
                "req_key": req_key,
                "alpha": self.alpha,
                "T": self.T,
                "top_k_kl": self.top_k_kl,
                "top_k_cov": self.top_k_cov,
                "warmup": self.warmup,
                "w": self.w,
                "max_new_tokens": self.max_new_tokens,
                "continuation_mode": self.continuation_mode,
                "T_cont": self.T_cont,
                **extra,
            },
        )

    # ── Core generation helper ─────────────────────────────────────────────────

    def _generate(
        self,
        requests: List[Tuple[List[int], SamplingParams]],
    ) -> List[List[int]]:
        if not requests:
            return []
        prompt_ids_list = [r[0] for r in requests]
        params_list     = [r[1] for r in requests]
        prompts = [{"prompt_token_ids": ids} for ids in prompt_ids_list]
        outputs = self.llm.generate(prompts, sampling_params=params_list, use_tqdm=True)
        return [list(out.outputs[0].token_ids) for out in outputs]

    # ── Pass 1: all conditions ──────────────────────────────────────────────────

    def run_all_conditions(
        self,
        pid_k_caches: List[Tuple[str, int, List[dict]]],
        background_map: Dict[Tuple[str, int], dict],
        prompt_ids_map: Dict[str, List[int]],
        verifier_fn: Callable,
    ) -> List[dict]:
        """Run retry / rand / geo_pred / geo_wrong (+ local_temp) for all (pid, k)."""
        requests: List[Tuple[List[int], SamplingParams]] = []
        meta: List[dict] = []

        for pid, k, _caches in pid_k_caches:
            bg = background_map[(pid, k)]
            prompt = prompt_ids_map[pid]

            requests.append((prompt, self._retry_params()))
            meta.append({"condition": "retry", "pid": pid, "k": k})

            rand_t = self.rng.randint(self.warmup,
                                      max(self.warmup + 1, self.max_new_tokens // 2))
            rand_tau = self.rng.choice(["path", "cov"])
            rk_rand = f"{pid}_{k}_rand"
            requests.append((prompt, self._repair_params(
                "rand", bg, rk_rand, rand_t=rand_t, rand_tau=rand_tau)))
            meta.append({"condition": "rand", "pid": pid, "k": k, "req_key": rk_rand})

            rk_geo = f"{pid}_{k}_geo_pred"
            requests.append((prompt, self._repair_params("geo_pred", bg, rk_geo)))
            meta.append({"condition": "geo_pred", "pid": pid, "k": k, "req_key": rk_geo})

            rk_gw = f"{pid}_{k}_geo_wrong"
            requests.append((prompt, self._repair_params("geo_wrong", bg, rk_gw)))
            meta.append({"condition": "geo_wrong", "pid": pid, "k": k, "req_key": rk_gw})

            for T_local in self.temp_interventions:
                tag = str(T_local).replace(".", "p")
                rk_lt = f"{pid}_{k}_local_temp_{tag}"
                requests.append((prompt, self._repair_params(
                    "local_temp", bg, rk_lt, T_local=T_local)))
                meta.append({"condition": f"local_temp_{tag}", "pid": pid, "k": k,
                             "req_key": rk_lt, "T_local": T_local})

        print(f"[LogitRepairEngine] Pass 1: {len(requests)} requests "
              f"({len(pid_k_caches)} pid×k pairs)", flush=True)

        token_outputs = self._generate(requests)

        results_by_pk: Dict[Tuple[str, int], dict] = {}

        for i, (tids, m) in enumerate(zip(token_outputs, meta)):
            pid, k = m["pid"], m["k"]
            cond   = m["condition"]
            bg     = background_map[(pid, k)]

            if (pid, k) not in results_by_pk:
                results_by_pk[(pid, k)] = {
                    "pid": pid, "k": k,
                    "background": {kk: vv for kk, vv in bg.items()
                                   if isinstance(vv, (float, int))},
                }
            res = results_by_pk[(pid, k)]

            correct = verifier_fn(tids, pid)

            if cond == "retry":
                res["retry"] = {"correct": correct, "n_tokens": len(tids),
                                "t_hat": None, "tau_hat": None, "fired": False,
                                "fire_score": None, "entropy_sensitive": None,
                                "V_hat": None}
            else:
                req_key = m["req_key"]
                state_path = os.path.join(self.state_dir, f"{req_key}.json")
                try:
                    with open(state_path) as _f:
                        st = json.load(_f)
                except Exception:
                    st = {}
                entry = {
                    "correct": correct,
                    "n_tokens": len(tids),
                    "t_hat": st.get("t_hat"),
                    "tau_hat": st.get("tau_hat"),
                    "fired": st.get("fired", False),
                    "fire_score": st.get("fire_score"),
                    "entropy_sensitive": st.get("entropy_sensitive"),
                    "V_hat": None,
                }
                if cond.startswith("local_temp_"):
                    entry["T_local"] = m.get("T_local")
                res[cond] = entry

        # ── Pass 2: position ablation for fired geo_pred cases ────────────────
        abl_requests: List[Tuple[List[int], SamplingParams]] = []
        abl_meta: List[dict] = []

        for (pid, k), res in results_by_pk.items():
            gp = res.get("geo_pred", {})
            if gp.get("fired") and gp.get("t_hat") is not None:
                t_hat  = gp["t_hat"]
                tau_hat = gp["tau_hat"]
                bg     = background_map[(pid, k)]
                prompt = prompt_ids_map[pid]
                for offset in _ABLATION_OFFSETS:
                    t_abl = max(self.warmup, t_hat + offset)
                    rk_abl = f"{pid}_{k}_abl_{offset}"
                    abl_requests.append((prompt, self._repair_params(
                        "geo_pred", bg, rk_abl,
                        t_force=t_abl, tau_force=tau_hat)))
                    abl_meta.append({"pid": pid, "k": k, "offset": offset,
                                     "req_key": rk_abl})

        if abl_requests:
            print(f"[LogitRepairEngine] Pass 2 (ablation): {len(abl_requests)} requests",
                  flush=True)
            abl_outputs = self._generate(abl_requests)
            for tids, am in zip(abl_outputs, abl_meta):
                pid, k, offset = am["pid"], am["k"], am["offset"]
                correct = verifier_fn(tids, pid)
                results_by_pk[(pid, k)].setdefault("ablation", {})[offset] = correct

        for res in results_by_pk.values():
            res.setdefault("ablation", {})

        for fname in os.listdir(self.state_dir):
            try:
                os.remove(os.path.join(self.state_dir, fname))
            except Exception:
                pass

        return list(results_by_pk.values())

    # ── Dense batch ───────────────────────────────────────────────────────────

    def run_dense_batch(
        self,
        fired_cases: List[Tuple[str, int, str]],
        prompt_ids_map: Dict[str, List[int]],
        verifier_fn: Callable,
        background_map: Optional[Dict[Tuple[str, int], dict]] = None,
    ) -> List[dict]:
        """Run dense condition for fired (pid, k, tau_hat) triples."""
        requests: List[Tuple[List[int], SamplingParams]] = []
        case_meta: List[Tuple[str, int, str]] = []

        for pid, k, tau_hat in fired_cases:
            if pid not in prompt_ids_map:
                continue
            bg = (background_map or {}).get((pid, k), {"lambda_J": 0.0,
                  "mu_path": 0.0, "sig_path": 1.0, "mu_cov": 0.0,
                  "sig_cov": 1.0, "mu_logV": 0.0, "sig_logV": 1.0,
                  "lambda_V": 0.0, "w": self.w})
            rk = f"{pid}_{k}_dense"
            requests.append((
                prompt_ids_map[pid],
                self._repair_params("dense", bg, rk, dense_tau=tau_hat),
            ))
            case_meta.append((pid, k, tau_hat))

        if not requests:
            return []

        print(f"[LogitRepairEngine] Dense pass: {len(requests)} requests", flush=True)
        token_outputs = self._generate(requests)

        results = []
        for tids, (pid, k, tau_hat) in zip(token_outputs, case_meta):
            results.append({
                "pid": pid, "k": k,
                "condition": "dense",
                "tau_hat": tau_hat,
                "correct": verifier_fn(tids, pid),
                "n_tokens": len(tids),
            })

        return results

    def unload(self) -> None:
        del self.llm
        torch.cuda.empty_cache()
        self._tmp.cleanup()
        shutil.rmtree(self.state_dir, ignore_errors=True)
