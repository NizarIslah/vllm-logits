"""src/vllm_logits/scoring.py — scoring math + reference engines + HF reference engines.


`compute_scores_at_t` and `estimate_background` are the core scoring math
used by the repair processor. `CacheLogitsEngine` / `ProspectiveRepairEngine` are
the single-GPU HF reference path (the production path is the vLLM
`features.VllmCacheLogitsEngine` + `repair.LogitRepairEngine`).
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .junctions import RepairContext  # noqa: F401  (re-exported for parity)
from .storage import LogitStore

# ── Constants ─────────────────────────────────────────────────────────────────

_TOP_K_KL = 100   # vocabulary top-k for KL / Fisher info approximation
_TOP_K_COV = 20   # ancestor top-k set for coverage score
_WARMUP = 20      # positions before scan statistic is armed


# ══════════════════════════════════════════════════════════════════════════════
# Score computation (per token, stateless)
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores_at_t(
    logit_S: torch.Tensor,
    logit_A: torch.Tensor,
    x_t: int,
    background: dict,
    T: float = 0.6,
    top_k_kl: int = _TOP_K_KL,
    top_k_cov: int = _TOP_K_COV,
) -> dict:
    """Compute Z_path, Z_cov, J_t, tau_t, V_t, entropy_sensitive at one token.

    All inputs available from prefix only — leakage-free (prefix only).

    Returns dict with keys:
        Z_path, Z_cov, J_t, tau_t, V_t, Z_V, entropy_sensitive,
        mu_r, I_r, pS_on_set, pA_on_set, G_cov, r_chosen
    """
    log_pS = F.log_softmax(logit_S / T, dim=-1)
    log_pA = F.log_softmax(logit_A / T, dim=-1)
    pS = log_pS.exp()
    pA = log_pA.exp()

    # ── Path score (e-geodesic) ───────────────────────────────────────────────
    r_chosen = (log_pS[x_t] - log_pA[x_t]).item()

    topk_kl_idx = pS.topk(top_k_kl).indices
    r_topk = log_pS[topk_kl_idx] - log_pA[topk_kl_idx]
    pS_topk_kl = pS[topk_kl_idx]
    pS_topk_kl = pS_topk_kl / (pS_topk_kl.sum() + 1e-12)

    mu_r = (pS_topk_kl * r_topk).sum().item()
    I_r = (pS_topk_kl * (r_topk - mu_r) ** 2).sum().item()

    Z_path = max(0.0, (r_chosen - mu_r) / (math.sqrt(I_r) + 1e-6))

    # ── Coverage score (m-geodesic) ───────────────────────────────────────────
    topk_cov_idx = pA.topk(top_k_cov).indices
    pA_on_set = pA[topk_cov_idx].sum().item()
    pS_on_set = pS[topk_cov_idx].sum().item()

    G_cov = math.log((pA_on_set + 1e-8) / (pS_on_set + 1e-8))
    Z_cov_raw = (G_cov - background["mu_cov"]) / (background["sig_cov"] + 1e-6)
    Z_cov = max(0.0, Z_cov_raw)

    # ── Union test ────────────────────────────────────────────────────────────
    J_t = max(Z_path, Z_cov)
    tau_t = "path" if Z_path >= Z_cov else "cov"

    # ── Temperature susceptibility (Fisher info of temp submodel) ─────────────
    z_topk = logit_S[topk_kl_idx].float()
    Ez = (pS_topk_kl * z_topk).sum()
    V_t = (pS_topk_kl * (z_topk - Ez) ** 2).sum().item()

    log_V_t = math.log(max(V_t, 1e-8))
    Z_V = (log_V_t - background["mu_logV"]) / (background["sig_logV"] + 1e-6)
    entropy_sensitive = Z_V > background["lambda_V"]

    return {
        "Z_path": Z_path, "Z_cov": Z_cov,
        "J_t": J_t, "tau_t": tau_t,
        "V_t": V_t, "Z_V": Z_V, "entropy_sensitive": entropy_sensitive,
        "mu_r": mu_r, "I_r": I_r,
        "pS_on_set": pS_on_set, "pA_on_set": pA_on_set,
        "G_cov": G_cov, "r_chosen": r_chosen,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Background estimation from k failed rollout caches
# ══════════════════════════════════════════════════════════════════════════════

def estimate_background(faifeature_caches: List[dict], w: int = 1,
                         q_J: float = 0.99, q_V: float = 0.75) -> dict:
    """Estimate problem-specific background stats from k failed rollout caches.

    Each cache is a dict from LogitStore.load_feature_cache(), with keys
    Delta_path, G_cov, V, J_approx (float Tensors of length L_i). Pooling is over
    all (rollout, position) pairs so differing rollout lengths are handled.
    """
    if not faifeature_caches:
        raise ValueError("estimate_background requires at least 1 failed cache")

    all_path: List[float] = []
    all_cov: List[float] = []
    all_logV: List[float] = []
    all_scan: List[float] = []

    for cache in faifeature_caches:
        dp = cache["Delta_path"].tolist()
        gc = cache["G_cov"].tolist()
        vv = cache["V"].tolist()
        ja = cache.get("J_approx", cache.get("V")).tolist()

        all_path.extend(dp)
        all_cov.extend(gc)
        all_logV.extend(math.log(max(v, 1e-8)) for v in vv)

        for t in range(len(ja)):
            start = max(0, t - w + 1)
            s_t = sum(ja[start: t + 1])
            all_scan.append(s_t)

    mu_path = float(np.mean(all_path))
    sig_path = float(np.std(all_path) + 1e-6)
    mu_cov = float(np.mean(all_cov))
    sig_cov = float(np.std(all_cov) + 1e-6)
    mu_logV = float(np.mean(all_logV))
    sig_logV = float(np.std(all_logV) + 1e-6)

    lambda_J = float(np.quantile(all_scan, q_J))

    z_V_vals = [(lv - mu_logV) / sig_logV for lv in all_logV]
    lambda_V = float(np.quantile(z_V_vals, q_V))

    return {
        "mu_path": mu_path, "sig_path": sig_path,
        "mu_cov": mu_cov, "sig_cov": sig_cov,
        "mu_logV": mu_logV, "sig_logV": sig_logV,
        "lambda_J": lambda_J, "lambda_V": lambda_V,
        "w": w, "q_J": q_J, "q_V": q_V,
        "k": len(faifeature_caches),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CacheLogitsEngine (HF reference path) — forward passes, write feature cache
# ══════════════════════════════════════════════════════════════════════════════

class CacheLogitsEngine:
    """HF reference: forward passes through specialist + ancestor over each rollout.

    Writes per-rollout feature cache files via LogitStore. Idempotent. Both
    models on the same device. (Production uses features.VllmCacheLogitsEngine.)
    """

    def __init__(
        self,
        specialist_path: str,
        ancestor_path: str,
        device: str = "cuda",
        hf_home: Optional[str] = None,
        top_k_kl: int = _TOP_K_KL,
        top_k_cov: int = _TOP_K_COV,
        T: float = 0.6,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.specialist_path = specialist_path
        self.ancestor_path = ancestor_path
        self.device = torch.device(device)
        self.hf_home = hf_home
        self.top_k_kl = top_k_kl
        self.top_k_cov = top_k_cov
        self.T = T

        self._AutoModelForCausalLM = AutoModelForCausalLM
        print(f"[CacheLogitsEngine] Loading tokenizer from {ancestor_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            ancestor_path, cache_dir=hf_home, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model_ancestor = self._load_model(ancestor_path)
        self.model_specialist = self._load_model(specialist_path)
        print("[CacheLogitsEngine] Both models loaded.")

    def _load_model(self, name: str):
        for attn in ("sdpa", "eager"):
            try:
                print(f"[CacheLogitsEngine] Loading {name} → {self.device} (attn={attn})")
                model = self._AutoModelForCausalLM.from_pretrained(
                    name,
                    dtype=torch.bfloat16,
                    device_map=str(self.device),
                    attn_implementation=attn,
                    cache_dir=self.hf_home,
                    trust_remote_code=True,
                ).eval()
                return model
            except ImportError as e:
                if "FlashAttention" in str(e) and attn == "sdpa":
                    continue
                raise

    @torch.no_grad()
    def cache_rollout(
        self,
        prompt_ids: List[int],
        rollout_ids: List[int],
    ) -> dict:
        """Forward pass through both models over the full rollout prefix."""
        T = self.T
        top_k_kl = self.top_k_kl
        top_k_cov = self.top_k_cov
        dev = self.device

        full_ids = torch.tensor(
            prompt_ids + rollout_ids, dtype=torch.long, device=dev
        ).unsqueeze(0)

        prompt_len = len(prompt_ids)
        gen_len = len(rollout_ids)

        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            out_S = self.model_specialist(full_ids)
            out_A = self.model_ancestor(full_ids)

        logits_S_all = out_S.logits[0].float()
        logits_A_all = out_A.logits[0].float()

        Delta_path_list: List[float] = []
        G_cov_list: List[float] = []
        V_list: List[float] = []
        J_approx_list: List[float] = []
        log_pS_rows: List[torch.Tensor] = []
        log_pA_rows: List[torch.Tensor] = []
        logits_S_rows: List[torch.Tensor] = []
        logits_A_rows: List[torch.Tensor] = []
        pS_on_set_list: List[float] = []
        pA_on_set_list: List[float] = []

        for s in range(gen_len):
            pos = prompt_len - 1 + s
            x_t = rollout_ids[s]

            logit_S = logits_S_all[pos]
            logit_A = logits_A_all[pos]

            log_pS = F.log_softmax(logit_S / T, dim=-1)
            log_pA = F.log_softmax(logit_A / T, dim=-1)
            pS = log_pS.exp()
            pA = log_pA.exp()

            r_chosen = (log_pS[x_t] - log_pA[x_t]).item()
            Delta_path_list.append(r_chosen)

            topk_cov_idx = pA.topk(top_k_cov).indices
            pA_on_set = pA[topk_cov_idx].sum().item()
            pS_on_set = pS[topk_cov_idx].sum().item()
            G_cov = math.log((pA_on_set + 1e-8) / (pS_on_set + 1e-8))
            G_cov_list.append(G_cov)
            pA_on_set_list.append(pA_on_set)
            pS_on_set_list.append(pS_on_set)

            topk_kl_idx = pS.topk(top_k_kl).indices
            pS_topk = pS[topk_kl_idx]
            pS_topk = pS_topk / (pS_topk.sum() + 1e-12)
            z_topk = logit_S[topk_kl_idx]
            Ez = (pS_topk * z_topk).sum()
            V_t = (pS_topk * (z_topk - Ez) ** 2).sum().item()
            V_list.append(V_t)

            J_approx_list.append(max(0.0, r_chosen) + max(0.0, G_cov))

            log_pS_rows.append(log_pS[topk_kl_idx].cpu().half())
            log_pA_rows.append(log_pA[topk_kl_idx].cpu().half())
            logits_S_rows.append(logit_S[topk_kl_idx].cpu().half())
            logits_A_rows.append(logit_A[topk_kl_idx].cpu().half())

        return {
            "Delta_path":     torch.tensor(Delta_path_list),
            "G_cov":          torch.tensor(G_cov_list),
            "V":              torch.tensor(V_list),
            "J_approx":       torch.tensor(J_approx_list),
            "log_pS_topk":    torch.stack(log_pS_rows),
            "log_pA_topk":    torch.stack(log_pA_rows),
            "logits_S_topk":  torch.stack(logits_S_rows),
            "logits_A_topk":  torch.stack(logits_A_rows),
            "pS_on_set":      torch.tensor(pS_on_set_list),
            "pA_on_set":      torch.tensor(pA_on_set_list),
        }

    def run(
        self,
        rollouts: List[dict],
        store: LogitStore,
        task: str,
        model_tag: str,
        prompt_ids_map: Dict[str, List[int]],
    ) -> int:
        """Cache all rollouts. Returns count of newly cached (skipped if exists)."""
        new_count = 0
        for i, r in enumerate(rollouts):
            pid = str(r["pid"])
            ridx = int(r.get("rollout_id", r.get("rollout_idx", i)))
            if store.feature_cache_exists(task, model_tag, pid, ridx):
                continue

            prompt_ids = prompt_ids_map[pid]
            rollout_ids = r["token_ids"]

            payload = self.cache_rollout(prompt_ids, rollout_ids)
            store.save_feature_cache(task, model_tag, pid, ridx, payload)
            new_count += 1
            if new_count % 10 == 0:
                print(f"[CacheLogitsEngine] cached {new_count} rollouts ...", flush=True)

        print(f"[CacheLogitsEngine] Done. {new_count} new, "
              f"{len(rollouts) - new_count} skipped (already cached).")
        return new_count

    def unload(self) -> None:
        del self.model_specialist, self.model_ancestor
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# ProspectiveRepairEngine (HF reference path) — online detection + 4 conditions
# ══════════════════════════════════════════════════════════════════════════════

_CONDITIONS = ("retry", "rand", "geo_pred", "geo_wrong")
_ABLATION_OFFSETS = (-10, -5, 0, 5, 10)


class ProspectiveRepairEngine:
    """HF reference: rollout k+1 under 4 conditions with online scan statistic.

    Conditions: retry / rand / geo_pred / geo_wrong. Dense via run_dense().
    (Production uses repair.LogitRepairEngine on vLLM.)
    """

    def __init__(
        self,
        specialist_path: str,
        ancestor_path: str,
        device: str = "cuda",
        hf_home: Optional[str] = None,
        T: float = 0.6,
        alpha: float = 0.7,
        w: int = 1,
        q_J: float = 0.99,
        q_V: float = 0.75,
        top_k_kl: int = _TOP_K_KL,
        top_k_cov: int = _TOP_K_COV,
        warmup: int = _WARMUP,
        max_new_tokens: int = 1024,
        seed: int = 42,
    ) -> None:
        self.specialist_path = specialist_path
        self.ancestor_path = ancestor_path
        self.device = torch.device(device)
        self.T = T
        self.alpha = alpha
        self.w = w
        self.q_J = q_J
        self.q_V = q_V
        self.top_k_kl = top_k_kl
        self.top_k_cov = top_k_cov
        self.warmup = warmup
        self.max_new_tokens = max_new_tokens
        self.rng = random.Random(seed)

        self._cache_engine = CacheLogitsEngine(
            specialist_path=specialist_path,
            ancestor_path=ancestor_path,
            device=device,
            hf_home=hf_home,
            top_k_kl=top_k_kl,
            top_k_cov=top_k_cov,
            T=T,
        )
        self.model_specialist = self._cache_engine.model_specialist
        self.model_ancestor = self._cache_engine.model_ancestor
        self.tokenizer = self._cache_engine.tokenizer

        raw_eos = self.model_specialist.generation_config.eos_token_id
        if isinstance(raw_eos, int):
            raw_eos = [raw_eos]
        if self.tokenizer.eos_token_id not in raw_eos:
            raw_eos = list(raw_eos) + [self.tokenizer.eos_token_id]
        self._eos_ids: set = set(raw_eos)

    def _is_eos(self, tok: int) -> bool:
        return tok in self._eos_ids

    def _sample(self, logit_S: torch.Tensor) -> int:
        probs = F.softmax(logit_S / self.T, dim=-1)
        return torch.multinomial(probs, 1).item()

    def _greedy(self, logit_S: torch.Tensor) -> int:
        return logit_S.argmax().item()

    def _mix_token(self, logit_S: torch.Tensor, logit_A: torch.Tensor,
                    tau: str) -> int:
        a = self.alpha
        if tau == "path":
            mixed = a * logit_S + (1 - a) * logit_A
            return mixed.argmax().item()
        else:
            pS = F.softmax(logit_S / self.T, dim=-1)
            pA = F.softmax(logit_A / self.T, dim=-1)
            return (a * pS + (1 - a) * pA).argmax().item()

    def _init_kv(self, prompt_ids: List[int]) -> Tuple:
        dev = self.device
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=dev).unsqueeze(0)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            out_S = self.model_specialist(prompt_tensor, use_cache=True)
            out_A = self.model_ancestor(prompt_tensor, use_cache=True)
        return (
            out_S.past_key_values,
            out_A.past_key_values,
            out_S.logits[0, -1].float(),
            out_A.logits[0, -1].float(),
        )

    def _step_S(self, tok: int, past_S) -> Tuple[torch.Tensor, Any]:
        t = torch.tensor([[tok]], dtype=torch.long, device=self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            out = self.model_specialist(t, past_key_values=past_S, use_cache=True)
        return out.logits[0, -1].float(), out.past_key_values

    def _step_SA(self, tok: int, past_S, past_A) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
        t = torch.tensor([[tok]], dtype=torch.long, device=self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            out_S = self.model_specialist(t, past_key_values=past_S, use_cache=True)
            out_A = self.model_ancestor(t, past_key_values=past_A, use_cache=True)
        return (out_S.logits[0, -1].float(), out_A.logits[0, -1].float(),
                out_S.past_key_values, out_A.past_key_values)

    @torch.no_grad()
    def _generate_condition(
        self,
        prompt_ids: List[int],
        background: dict,
        condition: str,
        t_force: Optional[int] = None,
        tau_force: Optional[str] = None,
    ) -> dict:
        lambda_J = background["lambda_J"]
        w = self.w

        past_S, past_A, logit_S, logit_A = self._init_kv(prompt_ids)

        generated: List[int] = []
        J_window: List[float] = []
        scores: List[dict] = []
        intervened = False
        t_hat: Optional[int] = None
        tau_hat: Optional[str] = None
        V_hat: Optional[float] = None
        entropy_sensitive: Optional[bool] = None
        fire_score: Optional[dict] = None

        rand_t = rand_tau = None
        if condition == "rand":
            rand_t = self.rng.randint(self.warmup,
                                      max(self.warmup + 1, self.max_new_tokens // 2))
            rand_tau = self.rng.choice(["path", "cov"])

        t = 0
        while t < self.max_new_tokens:
            if not intervened:
                x_t = self._sample(logit_S)
                sc = compute_scores_at_t(
                    logit_S, logit_A, x_t, background, T=self.T,
                    top_k_kl=self.top_k_kl, top_k_cov=self.top_k_cov,
                )
                sc["t"] = t
                scores.append(sc)

                J_window.append(sc["J_t"])
                if len(J_window) > w:
                    J_window.pop(0)
                S_t = sum(J_window)

                fire = False
                if condition == "retry":
                    fire = False
                elif condition == "rand":
                    fire = (t == rand_t)
                elif condition in ("geo_pred", "geo_wrong"):
                    fire = (len(J_window) == w and S_t > lambda_J
                            and t >= self.warmup)
                    if t_force is not None:
                        fire = (t == t_force)

                if fire:
                    t_hat = t
                    if condition == "rand":
                        tau_hat = rand_tau
                    elif condition == "geo_wrong":
                        tau_hat = "cov" if sc["tau_t"] == "path" else "path"
                    else:
                        tau_hat = tau_force if tau_force is not None else sc["tau_t"]
                    V_hat = sc["V_t"]
                    entropy_sensitive = sc["entropy_sensitive"]
                    fire_score = {
                        "Z_path": sc["Z_path"], "Z_cov": sc["Z_cov"],
                        "J_t": sc["J_t"], "S_t": float(S_t),
                        "t": t, "t_frac": t / max(1, self.max_new_tokens),
                        "pS_on_set": sc["pS_on_set"], "pA_on_set": sc["pA_on_set"],
                    }
                    intervened = True

                    for step_i in range(w):
                        tok = self._mix_token(logit_S, logit_A, tau_hat)
                        generated.append(tok)
                        if self._is_eos(tok):
                            break
                        if step_i < w - 1:
                            logit_S, logit_A, past_S, past_A = self._step_SA(
                                tok, past_S, past_A)
                        else:
                            logit_S, past_S = self._step_S(tok, past_S)
                    t += w
                    continue
                else:
                    logit_S, logit_A, past_S, past_A = self._step_SA(
                        x_t, past_S, past_A)
                    generated.append(x_t)
            else:
                next_tok = self._greedy(logit_S)
                generated.append(next_tok)
                logit_S, past_S = self._step_S(next_tok, past_S)

            t += 1
            if self._is_eos(generated[-1]):
                break

        return {
            "token_ids": generated,
            "t_hat": t_hat,
            "tau_hat": tau_hat,
            "V_hat": V_hat,
            "entropy_sensitive": entropy_sensitive,
            "fired": t_hat is not None,
            "fire_score": fire_score,
            "scores": scores,
        }

    @torch.no_grad()
    def run_problem(
        self,
        pid: str,
        prompt_ids: List[int],
        k_caches: List[dict],
        verifier_fn,
        k: int,
    ) -> dict:
        background = estimate_background(k_caches, w=self.w,
                                          q_J=self.q_J, q_V=self.q_V)

        results: dict = {"pid": pid, "k": k, "background": {
            kk: vv for kk, vv in background.items()
            if isinstance(vv, (float, int))
        }}

        for cond in _CONDITIONS:
            out = self._generate_condition(prompt_ids, background, cond)
            correct = verifier_fn(out["token_ids"], pid)
            results[cond] = {
                "correct": correct,
                "t_hat": out["t_hat"],
                "tau_hat": out["tau_hat"],
                "V_hat": out["V_hat"],
                "entropy_sensitive": out["entropy_sensitive"],
                "fired": out["fired"],
                "fire_score": out.get("fire_score"),
                "n_tokens": len(out["token_ids"]),
            }

        geo = results["geo_pred"]
        ablation: Dict[int, bool] = {}
        if geo["fired"] and geo["t_hat"] is not None:
            t_hat = geo["t_hat"]
            tau_hat = geo["tau_hat"]
            for offset in _ABLATION_OFFSETS:
                t_abl = max(self.warmup, t_hat + offset)
                out_abl = self._generate_condition(
                    prompt_ids, background,
                    condition="geo_pred",
                    t_force=t_abl,
                    tau_force=tau_hat,
                )
                ablation[offset] = verifier_fn(out_abl["token_ids"], pid)
        results["ablation"] = ablation

        return results

    @torch.no_grad()
    def run_dense(
        self,
        pid: str,
        prompt_ids: List[int],
        tau_hat: str,
        verifier_fn,
        k: int,
    ) -> dict:
        past_S, past_A, logit_S, logit_A = self._init_kv(prompt_ids)
        generated: List[int] = []
        t = 0
        while t < self.max_new_tokens:
            tok = self._mix_token(logit_S, logit_A, tau_hat)
            generated.append(tok)
            if self._is_eos(tok):
                break
            logit_S, logit_A, past_S, past_A = self._step_SA(tok, past_S, past_A)
            t += 1
        return {
            "pid": pid, "k": k,
            "condition": "dense",
            "tau_hat": tau_hat,
            "correct": verifier_fn(generated, pid),
            "n_tokens": len(generated),
        }

    def unload(self) -> None:
        self._cache_engine.unload()
