"""src/vllm_logits/features.py — vLLM-native feature-cache engine (cache_logits).

The ONLY repo coupling severed: `_resolve_hf_cache_snapshot` / `_ensure_local`
(the HF-snapshot resolver) is dropped — the caller passes resolved LOCAL paths
(or HF repo ids that transformers/vLLM resolve natively).

Computes per-token logit features for each rollout via two batched vLLM prefill
passes (specialist, then ancestor) using prompt_logprobs at temperature T.

Cache keys written (matching estimate_background expectations):
    Delta_path  [L]  log_pS_T[x_t] - log_pA_T[x_t]
    G_cov       [L]  log(pA_on_set / pS_on_set), ancestor top-k_cov set
    V           [L]  exp(log_pS_T[x_t])  — specialist probability of x_t
    J_approx    [L]  max(0, Delta_path) + max(0, G_cov)
    pA_on_set   [L]  ancestor mass on its own top-k_cov
    logit_var   [L]  Var_{y~p_S^T,K}[z_S(y)] — probability-weighted logit variance
    entropy     [L]  H(p_S^T) over top-K
    kl_div      [L]  KL(p_S^T ‖ p_A^T) over specialist top-K
    logit_skew  [L]  κ3 third cumulant of the top-K renormalized logit distribution
    logit_kurt  [L]  κ4 fourth cumulant (curvature of V(β))

Also exports population-level feature helpers (`FEATS`, `_rollout_feats`,
`_load_cell_rollouts`) used by `showcase_clustering.py` and the parquet round-trip
"""
from __future__ import annotations

import collections
import glob
import math
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .storage import LogitStore

_LOG_FLOOR = -1000.0


class VllmCacheLogitsEngine:
    """Compute per-token feature caches for failed rollouts via two vLLM prefill passes.

    Model paths must be LOCAL dirs or HF repo ids resolvable by transformers/vLLM.
    (The repo's `_resolve_hf_cache_snapshot` indirection is intentionally NOT
    included — pass an already-resolved path.)
    """

    def __init__(
        self,
        specialist_path: str,
        ancestor_path: str,
        top_k_kl: int = 100,
        top_k_cov: int = 20,
        T: float = 0.6,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        store_topk_logits: bool = False,
        force_recompute: bool = False,
        prefill_token_budget: int = 51200,
    ) -> None:
        self.specialist_path = specialist_path
        self.ancestor_path = ancestor_path
        self.top_k_kl = top_k_kl
        self.top_k_cov = top_k_cov
        self.T = T
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = int(max_model_len) if max_model_len else None
        self.prefill_token_budget = int(prefill_token_budget)
        self.store_topk_logits = bool(store_topk_logits)
        self.force_recompute = bool(force_recompute)
        self.tokenizer = AutoTokenizer.from_pretrained(
            specialist_path, trust_remote_code=True
        )

    def _run_prefill(self, model_path: str, full_seqs: List[List[int]]) -> List:
        """Load model, run batched prefill with prompt_logprobs at T, unload.

        Sequence-length-aware sub-batching: caps each `generate` call by a
        TOTAL-TOKEN budget (`prefill_token_budget`) so long-sequence sub-batches
        shrink automatically while short-sequence batches stay large. The model
        is loaded ONCE; only the `generate` calls are sub-batched.

        Returns list of prompt_logprobs (one per sequence), in input order.
        """
        tp = torch.cuda.device_count()
        llm_kwargs = dict(
            model=model_path,
            enforce_eager=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            tensor_parallel_size=tp,
            disable_custom_all_reduce=True,
            disable_log_stats=True,
            enable_prefix_caching=True,
            max_num_batched_tokens=65536,
            max_logprobs=self.top_k_kl,
        )
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len
        llm = LLM(**llm_kwargs)
        # vLLM V1 returns raw T=1 logprobs; temperature rescaling is applied
        # post-hoc in _compute_feature_tensors via logsumexp over top-K.
        params = SamplingParams(
            max_tokens=1,
            temperature=1.0,
            prompt_logprobs=self.top_k_kl,
        )

        budget = max(1, int(self.prefill_token_budget))
        sub_batches: List[List[int]] = []
        cur: List[int] = []
        cur_tokens = 0
        for i, ids in enumerate(full_seqs):
            n = len(ids)
            if cur and cur_tokens + n > budget:
                sub_batches.append(cur)
                cur, cur_tokens = [], 0
            cur.append(i)
            cur_tokens += n
        if cur:
            sub_batches.append(cur)

        results: List = [None] * len(full_seqs)
        n_sub = len(sub_batches)
        if n_sub > 1:
            print(f"[VllmCacheLogitsEngine] _run_prefill: {len(full_seqs)} seqs "
                  f"split into {n_sub} sub-batch(es) (token budget {budget})",
                  flush=True)
        for sb_idx, idxs in enumerate(sub_batches):
            prompts = [{"prompt_token_ids": full_seqs[i]} for i in idxs]
            if n_sub > 1:
                tok = sum(len(full_seqs[i]) for i in idxs)
                print(f"[VllmCacheLogitsEngine]   sub-batch {sb_idx+1}/{n_sub}: "
                      f"{len(idxs)} seqs, {tok} tokens", flush=True)
            outputs = llm.generate(prompts, sampling_params=params, use_tqdm=True)
            for i, out in zip(idxs, outputs):
                results[i] = out.prompt_logprobs
            torch.cuda.empty_cache()
        del llm
        torch.cuda.empty_cache()
        return results

    def _compute_feature_tensors(
        self,
        rec: dict,
        spec_plp: list,
        anc_plp: list,
    ) -> Optional[tuple]:
        """Compute per-token feature tensors for one rollout. None if no valid positions."""
        prompt_len = rec["prompt_len"]
        output_ids = rec["output_ids"]
        top_k_cov  = self.top_k_cov

        dp_list, gc_list, v_list, pa_list, lv_list, ent_list, kl_list, lsk_list, lk_list = \
            [], [], [], [], [], [], [], [], []
        store_raw = self.store_topk_logits
        sp_lp_rows, sp_id_rows, ap_lp_rows, ap_id_rows = [], [], [], []

        T = self.T

        for i, x_t in enumerate(output_ids):
            pos = prompt_len + i
            if pos >= len(spec_plp) or pos >= len(anc_plp):
                break
            sp = spec_plp[pos]
            ap = anc_plp[pos]
            if sp is None or ap is None:
                continue

            sp_lps1 = [lp.logprob for lp in sp.values()]
            ap_lps1 = [lp.logprob for lp in ap.values()]
            if store_raw:
                sp_lp_rows.append(sp_lps1)
                sp_id_rows.append(list(sp.keys()))
                ap_lp_rows.append(ap_lps1)
                ap_id_rows.append(list(ap.keys()))
            sp_lse_T = math.log(sum(math.exp(lp / T) for lp in sp_lps1) + 1e-300)
            ap_lse_T = math.log(sum(math.exp(lp / T) for lp in ap_lps1) + 1e-300)

            # Delta_path = log pS_T(x_t) - log pA_T(x_t)
            log_pS_xt = (sp[x_t].logprob if x_t in sp else _LOG_FLOOR) / T - sp_lse_T
            log_pA_xt = (ap[x_t].logprob if x_t in ap else _LOG_FLOOR) / T - ap_lse_T
            dp_list.append(log_pS_xt - log_pA_xt)
            v_list.append(math.exp(log_pS_xt))

            # V_t: probability-weighted variance of logits = Fisher info of temp submodel
            pS_T_topk = [math.exp(lp / T - sp_lse_T) for lp in sp_lps1]
            sum_pS_T = sum(pS_T_topk) + 1e-300
            pS_T_norm = [p / sum_pS_T for p in pS_T_topk]
            Ez_logit = sum(p * lp for p, lp in zip(pS_T_norm, sp_lps1))
            lv = sum(p * (lp - Ez_logit) ** 2 for p, lp in zip(pS_T_norm, sp_lps1))
            lv_list.append(lv)

            # logit_skew κ3 (third central moment)
            lsk = sum(p * (lp - Ez_logit) ** 3 for p, lp in zip(pS_T_norm, sp_lps1))
            lsk_list.append(lsk)

            # logit_kurt κ4 = μ4 − 3·κ2² (curvature of V(β))
            mu4 = sum(p * (lp - Ez_logit) ** 4 for p, lp in zip(pS_T_norm, sp_lps1))
            lk_list.append(mu4 - 3.0 * lv ** 2)

            # entropy[t] = H(p_S^T)
            ent = -sum(p * math.log(p + 1e-300) for p in pS_T_norm)
            ent_list.append(ent)

            # kl_div[t] = KL(p_S^T || p_A^T) over specialist top-K
            sp_tok_ids = list(sp.keys())
            kl = sum(
                p * (math.log(p + 1e-300)
                     - ((ap[tok].logprob if tok in ap else _LOG_FLOOR) / T - ap_lse_T))
                for tok, p in zip(sp_tok_ids, pS_T_norm)
            )
            kl_list.append(max(kl, 0.0))

            # G_cov: ancestor top-k_cov set identified via Logprob.rank (O(K) scan)
            top_anc = [(tok, lp) for tok, lp in ap.items()
                       if lp.rank is not None and lp.rank <= top_k_cov]
            if not top_anc:  # rank unavailable — fall back to O(K log K) sort
                top_anc = sorted(ap.items(), key=lambda kv: -kv[1].logprob)[:top_k_cov]
                top_anc = [(tok, lp) for tok, lp in top_anc]

            pA_on_set = sum(
                math.exp(lp.logprob / T - ap_lse_T) for _, lp in top_anc
            )
            pS_on_set = sum(
                math.exp((sp[tok].logprob if tok in sp else _LOG_FLOOR) / T - sp_lse_T)
                for tok, _ in top_anc
            )
            gc_list.append(math.log((pA_on_set + 1e-8) / (pS_on_set + 1e-8)))
            pa_list.append(pA_on_set)

        if not dp_list:
            return None

        dp_t  = torch.tensor(dp_list,  dtype=torch.float32)
        gc_t  = torch.tensor(gc_list,  dtype=torch.float32)
        v_t   = torch.tensor(v_list,   dtype=torch.float32)
        pa_t  = torch.tensor(pa_list,  dtype=torch.float32)
        lv_t  = torch.tensor(lv_list,  dtype=torch.float32)
        ent_t = torch.tensor(ent_list, dtype=torch.float32)
        kl_t  = torch.tensor(kl_list,  dtype=torch.float32)
        lsk_t = torch.tensor(lsk_list, dtype=torch.float32)
        lk_t  = torch.tensor(lk_list,  dtype=torch.float32)

        raw = None
        if store_raw and sp_lp_rows:
            def _pad(rows, pad_val, dtype):
                k = max(len(r) for r in rows)
                arr = torch.full((len(rows), k), pad_val, dtype=dtype)
                for i, r in enumerate(rows):
                    if r:
                        arr[i, :len(r)] = torch.tensor(r, dtype=dtype)
                return arr
            raw = {
                "spec_topk_logprobs": _pad(sp_lp_rows, _LOG_FLOOR, torch.float16),
                "spec_topk_token_ids": _pad(sp_id_rows, -1, torch.int32),
                "anc_topk_logprobs":  _pad(ap_lp_rows, _LOG_FLOOR, torch.float16),
                "anc_topk_token_ids": _pad(ap_id_rows, -1, torch.int32),
                "T_extracted": float(self.T),
            }
        return dp_t, gc_t, v_t, pa_t, lv_t, ent_t, kl_t, lsk_t, lk_t, raw

    def compute_and_store(
        self,
        failed_rollouts: List[dict],
        prompt_ids_map: Dict[str, List[int]],
        store: LogitStore,
        task: str,
        model_tag: str,
        chunk_size: int = 500,
        cache_format: str = "pt",
    ) -> int:
        """Compute logit features per token and write cache files (resumable).

        Skips rollouts whose cache already exists. cache_format: "pt" → one .pt
        per rollout (legacy); "parquet" → one parquet shard per chunk.
        Returns count of new caches written this run.
        """
        parquet = (cache_format == "parquet")
        done = (store.feature_parquet_keys(task, model_tag)
                if parquet and not self.force_recompute else set())
        records = []
        for r in failed_rollouts:
            pid = str(r.get("pid") or r.get("problem_id") or r.get("id", ""))
            if not pid or pid not in prompt_ids_map:
                continue
            ridx = int(r.get("rollout_id", r.get("rollout_idx", r.get("sample_id", 0))))
            if not self.force_recompute:
                if parquet:
                    if (pid, ridx) in done:
                        continue
                elif store.feature_cache_exists(task, model_tag, pid, ridx):
                    continue

            prompt_ids = prompt_ids_map[pid]
            if "output_ids" in r:
                output_ids = list(r["output_ids"])
            elif "token_ids" in r:
                output_ids = list(r["token_ids"])
            elif "generated_text" in r:
                output_ids = self.tokenizer.encode(
                    r["generated_text"], add_special_tokens=False
                )
            else:
                continue
            if not output_ids:
                continue

            records.append({
                "pid": pid,
                "ridx": ridx,
                "prompt_len": len(prompt_ids),
                "output_ids": output_ids,
                "full_ids": prompt_ids + output_ids,
            })

        if not records:
            print("[VllmCacheLogitsEngine] All caches already exist — nothing to do.",
                  flush=True)
            return 0

        records.sort(key=lambda r: r["pid"])

        n_chunks = math.ceil(len(records) / chunk_size)
        print(f"[VllmCacheLogitsEngine] {len(records)} rollouts → "
              f"{n_chunks} chunk(s) of ≤{chunk_size}", flush=True)

        n_written = 0
        n_skipped_overlong = 0
        max_len = self.max_model_len
        shard_offset = store.n_feature_shards(task, model_tag) if parquet else 0
        for chunk_idx in range(n_chunks):
            chunk_payloads = []
            chunk = records[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]
            if max_len is not None:
                kept = [rec for rec in chunk if len(rec["full_ids"]) <= max_len]
                n_skipped_overlong += len(chunk) - len(kept)
                if not kept:
                    print(f"[VllmCacheLogitsEngine] Chunk {chunk_idx+1}/{n_chunks} — "
                          f"all {len(chunk)} sequences exceed max_model_len={max_len}; "
                          f"skipping chunk", flush=True)
                    continue
                if len(kept) < len(chunk):
                    print(f"[VllmCacheLogitsEngine] Chunk {chunk_idx+1}/{n_chunks} — "
                          f"dropped {len(chunk)-len(kept)}/{len(chunk)} overlong "
                          f"(>max_model_len={max_len})", flush=True)
                chunk = kept
            full_seqs = [rec["full_ids"] for rec in chunk]
            print(f"[VllmCacheLogitsEngine] Chunk {chunk_idx+1}/{n_chunks} — "
                  f"Pass 1/2 (specialist): {len(chunk)} rollouts", flush=True)
            spec_plp_list = self._run_prefill(self.specialist_path, full_seqs)

            print(f"[VllmCacheLogitsEngine] Chunk {chunk_idx+1}/{n_chunks} — "
                  f"Pass 2/2 (ancestor):   {len(chunk)} rollouts", flush=True)
            anc_plp_list = self._run_prefill(self.ancestor_path, full_seqs)

            for rec, spec_plp, anc_plp in zip(chunk, spec_plp_list, anc_plp_list):
                pid, ridx = rec["pid"], rec["ridx"]
                tensors = self._compute_feature_tensors(rec, spec_plp, anc_plp)
                if tensors is None:
                    continue
                dp_t, gc_t, v_t, pa_t, lv_t, ent_t, kl_t, lsk_t, lk_t, raw = tensors
                ja_t = (dp_t.clamp(min=0) + gc_t.clamp(min=0))

                payload = {
                    "pid":        pid,
                    "rollout_idx": ridx,
                    "Delta_path": dp_t,
                    "G_cov":      gc_t,
                    "V":          v_t,
                    "J_approx":   ja_t,
                    "pA_on_set":  pa_t,
                    "logit_var":  lv_t,
                    "entropy":    ent_t,
                    "kl_div":     kl_t,
                    "logit_skew": lsk_t,
                    "logit_kurt": lk_t,
                }
                if raw is not None:
                    payload.update(raw)
                if parquet:
                    chunk_payloads.append(payload)
                else:
                    store.save_feature_cache(task, model_tag, pid, ridx, payload)
                n_written += 1

            if parquet and chunk_payloads:
                store.save_feature_shard(task, model_tag, chunk_payloads,
                                     shard_offset + chunk_idx)
            print(f"[VllmCacheLogitsEngine] Chunk {chunk_idx+1}/{n_chunks} done — "
                  f"{n_written} total written so far"
                  f"{' (parquet shard)' if parquet else ''}.", flush=True)

        if n_skipped_overlong:
            print(f"[VllmCacheLogitsEngine] Skipped {n_skipped_overlong} rollouts "
                  f"with full_ids length > max_model_len={max_len}.", flush=True)
        print(f"[VllmCacheLogitsEngine] All done — {n_written} new caches written.",
              flush=True)
        return n_written


# ══════════════════════════════════════════════════════════════════════════════
# Population-level trajectory feature helpers
# Used by examples/showcase_clustering.py and tests/test_cache_logits_parquet.py.
# ══════════════════════════════════════════════════════════════════════════════

FEATS = ["logit_var", "J_approx", "kl_div", "Delta_path", "G_cov", "entropy"]
FEAT_COLS = [f"{f}_{sc}" for f in FEATS for sc in ("trace", "junc")]


def _rollout_feats(d: dict, w: int) -> Optional[dict]:
    """Per-rollout trace/junction features from a cache feature dict.

    Trims by kl_div (leading zeros mirror real caches); junction window = ±w
    around argmax(J_approx). Returns {f_trace, f_junc} for each f in FEATS.
    """
    kl = d["kl_div"].float().numpy()
    valid = kl != 0.0
    hi = int(np.where(valid)[0][-1]) + 1 if valid.any() else len(kl)
    arrs = {f: d[f].float().numpy()[:hi] for f in FEATS}
    J = arrs["J_approx"]
    if len(J) == 0:
        return None
    j = int(np.argmax(J)); lo, up = max(0, j - w), min(len(J), j + w + 1)
    out = {}
    for f, a in arrs.items():
        out[f"{f}_trace"] = float(a.mean())
        out[f"{f}_junc"] = float(a[lo:up].mean())
    return out


_CELL_CACHE: dict = {}


def _load_cell_rollouts(cache_dir: str) -> dict:
    """{pid: [rollout tensor-dicts]} for a cell, from parquet shards if present else .pt.

    Cached per cache_dir. Feature values returned as float32 tensors so
    _rollout_feats works unchanged for both formats.
    """
    if cache_dir in _CELL_CACHE:
        return _CELL_CACHE[cache_dir]
    out = collections.defaultdict(list)
    keys = FEATS + ["kl_div"]
    shards = sorted(glob.glob(f"{cache_dir}/shard_*.parquet"))
    if shards:
        import polars as pl
        for sh in shards:
            for row in pl.read_parquet(sh).to_dicts():
                d = {k: torch.tensor(row[k], dtype=torch.float32)
                     for k in keys if row.get(k) is not None}
                out[str(row.get("pid"))].append(d)
    else:
        for fp in glob.glob(f"{cache_dir}/*_r*.pt"):
            try:
                d = torch.load(fp, map_location="cpu", weights_only=False)
            except Exception:
                continue
            pid = fp.split("/")[-1].rsplit("_r", 1)[0]   # <safe_pid>_rNNNN.pt
            out[pid].append(d)
    _CELL_CACHE[cache_dir] = dict(out)
    return _CELL_CACHE[cache_dir]


def _pid_feats(cache_dir: str, pid: str, w: int) -> Optional[dict]:
    """Mean trace/junction features over all rollouts of one pid."""
    rows = []
    for d in _load_cell_rollouts(cache_dir).get(str(pid), []):
        try:
            r = _rollout_feats(d, w)
            if r:
                rows.append(r)
        except Exception:
            continue
    if not rows:
        return None
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
