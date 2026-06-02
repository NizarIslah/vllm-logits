"""src/vllm_logits/pipeline.py — the LogitPipeline public API.

`LogitPipeline(specialist, ancestor, arch="auto", ...).run(problems, rollouts,
correctness_fn, operators, k_values)` runs cache_logits → repair under the hood.
Both steps are also callable separately: `.cache_logits(...)` / `.repair(...)`.

The ONLY task-specific input is `correctness_fn(problem, text) -> bool`; the
ONLY model-specific input is `arch` (auto-detected from config.architectures).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from .storage import LogitStore
from .features import VllmCacheLogitsEngine
from .scoring import estimate_background
from . import register


# arch model_type → dual-backbone register fn (repair variant is what repair uses).
_ARCH_REGISTER = {
    "qwen2": register.register_dual_qwen,
    "qwen3": register.register_dual_qwen,
    "olmo2": register.register_dual_qwen,
    "olmo3": register.register_dual_qwen,
    "phi3":  register.register_dual_phi3,
    "llama": register.register_dual_llama,
}


def _resolve_local(path: str) -> str:
    """Resolve an HF repo id to a local snapshot dir (or return a local path)."""
    if os.path.exists(path):
        return path
    from huggingface_hub import snapshot_download
    return snapshot_download(path)


def _detect_arch(model_path: str) -> str:
    """Read config.architectures / model_type to pick the dual backbone family."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return getattr(cfg, "model_type", "qwen2")


class LogitPipeline:
    """Self-contained logit-intervention pipeline: cache_logits → repair.

    Args:
        specialist: HF id or local path to the fine-tuned (specialist) model.
        ancestor:   HF id or local path to the ancestor / base model.
        arch:       "auto" (detect) or a model_type family string.
        T, alpha, top_k_kl, top_k_cov: intervention hyperparameters.
        continuation_mode: "temperature" (production) | "greedy" (analysis).
        cache_dir:  base dir for the LogitStore.
        cache_format: "parquet" (efficient) | "pt".
        task / model_tag: cache cell identifiers.
    """

    def __init__(
        self,
        specialist: str,
        ancestor: str,
        arch: str = "auto",
        T: float = 0.6,
        alpha: float = 0.7,
        top_k_kl: int = 100,
        top_k_cov: int = 20,
        w: int = 1,
        warmup: int = 20,
        max_new_tokens: int = 1024,
        continuation_mode: str = "temperature",
        T_cont: Optional[float] = None,
        cache_dir: str = "./feature_cache",
        cache_format: str = "parquet",
        task: str = "task",
        model_tag: str = "spec",
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        seed: int = 42,
        sampling: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.specialist = _resolve_local(specialist)
        self.ancestor = _resolve_local(ancestor)
        self.arch = _detect_arch(self.specialist) if arch == "auto" else arch
        self.T = T
        self.alpha = alpha
        self.top_k_kl = top_k_kl
        self.top_k_cov = top_k_cov
        self.w = w
        self.warmup = warmup
        self.max_new_tokens = max_new_tokens
        self.continuation_mode = continuation_mode
        self.T_cont = T_cont
        self.cache_dir = cache_dir
        self.cache_format = cache_format
        self.task = task
        self.model_tag = model_tag
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.seed = seed
        self.sampling = sampling

        self.store = LogitStore(cache_dir)
        # Register the dual backbone for this arch (idempotent).
        reg = _ARCH_REGISTER.get(self.arch, register.register_dual_qwen)
        reg()
        register.register_repair_dual_qwen()

        self._tokenizer = None

    # ── helpers ─────────────────────────────────────────────────────────────

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.specialist, trust_remote_code=True)
        return self._tokenizer

    def _prompt_ids_map(self, problems: List[dict]) -> Dict[str, List[int]]:
        tok = self.tokenizer
        out: Dict[str, List[int]] = {}
        for p in problems:
            pid = str(p["problem_id"])
            out[pid] = tok.encode(p["prompt"], add_special_tokens=False)
        return out

    @staticmethod
    def _failed(rollouts: List[dict], correctness_fn: Optional[Callable],
                problems_by_id: Dict[str, dict]) -> List[dict]:
        """Select rollouts that are failures (is_correct False, or correctness_fn False)."""
        out = []
        for r in rollouts:
            if "is_correct" in r and r["is_correct"] is not None:
                ok = bool(r["is_correct"])
            elif correctness_fn is not None:
                prob = problems_by_id.get(str(r.get("problem_id", r.get("pid", ""))), {})
                ok = correctness_fn(prob, r.get("generated_text", ""))
            else:
                ok = False  # unknown → treat as failure (cache it)
            if not ok:
                out.append(r)
        return out

    # ── stage 1: cache_logits ────────────────────────────────────────────────

    def cache_logits(
        self,
        problems: List[dict],
        rollouts: List[dict],
        correctness_fn: Optional[Callable] = None,
        chunk_size: int = 500,
    ) -> int:
        """Run the cache_logits stage over the FAILED rollouts. Returns # written."""
        problems_by_id = {str(p["problem_id"]): p for p in problems}
        prompt_ids_map = self._prompt_ids_map(problems)
        failed = self._failed(rollouts, correctness_fn, problems_by_id)
        # Normalize keys for the engine (pid / rollout_idx / generated_text).
        recs = []
        for r in failed:
            recs.append({
                "pid": str(r.get("problem_id", r.get("pid", ""))),
                "rollout_idx": int(r.get("rollout_idx", r.get("rollout_id", 0))),
                "generated_text": r.get("generated_text", r.get("answer", "")),
            })
        engine = VllmCacheLogitsEngine(
            specialist_path=self.specialist,
            ancestor_path=self.ancestor,
            top_k_kl=self.top_k_kl,
            top_k_cov=self.top_k_cov,
            T=self.T,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
        )
        return engine.compute_and_store(
            recs, prompt_ids_map, self.store, self.task, self.model_tag,
            chunk_size=chunk_size, cache_format=self.cache_format,
        )

    # ── stage 2: repair ──────────────────────────────────────────────────────

    def repair(
        self,
        problems: List[dict],
        rollouts: List[dict],
        correctness_fn: Callable,
        operators: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
        run_dense: bool = True,
    ) -> List[dict]:
        """Run the repair stage. Requires cache_logits to have run for these pids.

        For each (pid, k): loads the first k failed-rollout caches, estimates
        background, runs the operator sweep via LogitRepairEngine. Returns the
        per-(pid, k) result dicts.
        """
        from .repair import LogitRepairEngine

        operators = operators or ["geo", "rand", "dense", "local_temp"]
        k_values = k_values or [1]
        problems_by_id = {str(p["problem_id"]): p for p in problems}
        prompt_ids_map = self._prompt_ids_map(problems)

        # Group failed rollouts per pid (ordered by rollout_idx).
        failed = self._failed(rollouts, correctness_fn, problems_by_id)
        by_pid: Dict[str, List[dict]] = {}
        for r in sorted(failed, key=lambda x: int(x.get("rollout_idx", x.get("rollout_id", 0)))):
            by_pid.setdefault(str(r.get("problem_id", r.get("pid", ""))), []).append(r)

        # Build (pid, k) cache lists + background map from the cached features.
        pid_k_caches: List[tuple] = []
        background_map: Dict[tuple, dict] = {}
        for pid, rs in by_pid.items():
            if pid not in prompt_ids_map:
                continue
            for k in k_values:
                caches = []
                for r in rs[:k]:
                    ridx = int(r.get("rollout_idx", r.get("rollout_id", 0)))
                    c = self.store.load_feature_cache(self.task, self.model_tag, pid, ridx)
                    if c is not None and "Delta_path" in c:
                        caches.append(c)
                if len(caches) < k:
                    continue  # not enough cached failures for this k
                pid_k_caches.append((pid, k, caches))
                background_map[(pid, k)] = estimate_background(caches, w=self.w)

        if not pid_k_caches:
            print("[LogitPipeline] No (pid, k) cells with enough cached failures; "
                  "did cache_logits run?", flush=True)
            return []

        temp_interventions = []
        if "local_temp" in operators:
            temp_interventions = [1.0, 1.5, 2.0]

        engine = LogitRepairEngine(
            specialist_path=self.specialist,
            ancestor_path=self.ancestor,
            T=self.T, alpha=self.alpha, w=self.w,
            top_k_kl=self.top_k_kl, top_k_cov=self.top_k_cov,
            warmup=self.warmup, max_new_tokens=self.max_new_tokens,
            seed=self.seed,
            gpu_memory_utilization=self.gpu_memory_utilization,
            temp_interventions=temp_interventions,
            sampling=self.sampling,
            continuation_mode=self.continuation_mode,
            T_cont=self.T_cont,
        )

        def verifier_fn(token_ids: List[int], pid: str) -> bool:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            return correctness_fn(problems_by_id.get(pid, {}), text)

        results = engine.run_all_conditions(
            pid_k_caches, background_map, prompt_ids_map, verifier_fn)

        if run_dense and "dense" in operators:
            fired = []
            for res in results:
                gp = res.get("geo_pred", {})
                if gp.get("fired") and gp.get("tau_hat"):
                    fired.append((res["pid"], res["k"], gp["tau_hat"]))
            dense_results = engine.run_dense_batch(
                fired, prompt_ids_map, verifier_fn, background_map)
            dense_by_pk = {(d["pid"], d["k"]): d for d in dense_results}
            for res in results:
                d = dense_by_pk.get((res["pid"], res["k"]))
                if d is not None:
                    res["dense"] = {"correct": d["correct"], "tau_hat": d["tau_hat"],
                                    "n_tokens": d["n_tokens"]}

        engine.unload()
        return results

    # ── one-shot: cache_logits → repair ──────────────────────────────────────

    def run(
        self,
        problems: List[dict],
        rollouts: List[dict],
        correctness_fn: Callable,
        operators: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
    ) -> List[dict]:
        """End-to-end: cache_logits over failures, then the repair operator sweep."""
        self.cache_logits(problems, rollouts, correctness_fn)
        return self.repair(problems, rollouts, correctness_fn,
                           operators=operators, k_values=k_values)
