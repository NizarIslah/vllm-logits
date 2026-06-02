"""High-performance storage for per-token feature caches + per-token feature caches.

(zero repo coupling). Two formats are supported for the feature cache:

  - legacy `.pt` (one file per rollout), and
  - parquet shards (one file holding many rollouts; opt-in via `cache_format`).

Readers (`load_feature_cache`, `iter_feature_rollouts`) auto-detect per cell.
"""
import torch
import json
import os
from pathlib import Path
from typing import Optional, List, Tuple
from safetensors.torch import save_file, load_file


class LogitStore:
    """High-performance storage for per-token feature caches with detector-aware hierarchy."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.metadata_path = self.base_dir / "metadata.jsonl"

    def get_logit_dir(self, detector_id: str) -> Path:
        path = self.base_dir / detector_id / "logits"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_logits(self, detector_id: str, key: str, logits: torch.Tensor, prefix_ids: List[int] = None, specialist_logits: torch.Tensor = None):
        """Saves [W, V] logits and metadata."""
        path = self.get_logit_dir(detector_id) / f"{key}.safetensors"
        # Ensure float16 for storage efficiency, but we compute in float32
        data = {"logits": logits.half()}
        if specialist_logits is not None:
            data["specialist_logits"] = specialist_logits.half()
        if prefix_ids is not None:
            data["prefix_ids"] = torch.tensor(prefix_ids, dtype=torch.long)
        save_file(data, str(path))

    def load_logits_with_prefix(self, detector_id: str, key: str) -> Tuple[Optional[torch.Tensor], Optional[List[int]], Optional[torch.Tensor]]:
        path = self.base_dir / detector_id / "logits" / f"{key}.safetensors"
        if not path.exists():
            return None, None, None
        data = load_file(str(path))
        prefix_ids = data.get("prefix_ids")
        if prefix_ids is not None:
            prefix_ids = prefix_ids.tolist()

        logits = data["logits"].float()
        spec_logits = data.get("specialist_logits")
        if spec_logits is not None:
            spec_logits = spec_logits.float()

        return logits, prefix_ids, spec_logits

    def append_metadata(self, metadata: dict):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "a") as f:
            f.write(json.dumps(metadata) + "\n")

    def get_all_metadata(self) -> List[dict]:
        if not self.metadata_path.exists():
            return []
        with open(self.metadata_path, "r") as f:
            return [json.loads(line) for line in f]

    # ── per-token feature cache ──────────────────────────────────

    def feature_cache_dir(self, task: str, model_tag: str) -> Path:
        path = self.base_dir / "feature_cache" / f"{task}_{model_tag}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def feature_cache_path(self, task: str, model_tag: str, pid: str, rollout_idx: int) -> Path:
        safe_pid = str(pid).replace("/", "_").replace(" ", "_")
        return self.feature_cache_dir(task, model_tag) / f"{safe_pid}_r{rollout_idx:04d}.pt"

    def feature_cache_exists(self, task: str, model_tag: str, pid: str, rollout_idx: int) -> bool:
        return self.feature_cache_path(task, model_tag, pid, rollout_idx).exists()

    def save_feature_cache(self, task: str, model_tag: str, pid: str, rollout_idx: int,
                       payload: dict) -> None:
        """Save feature cache tensors for one rollout. payload keys match RepairContext fields."""
        path = self.feature_cache_path(task, model_tag, pid, rollout_idx)
        # Store everything as float16 to save disk; caller casts back on load
        out = {}
        for k, v in payload.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.half() if v.is_floating_point() else v
            elif isinstance(v, (int, float)) and v is not None:
                out[k] = torch.tensor(v)
            # str/None values are not stored (path already encodes pid/ridx)
        out = {k: v for k, v in out.items() if v is not None}
        torch.save(out, str(path))

    def load_feature_cache(self, task: str, model_tag: str, pid: str,
                       rollout_idx: int) -> Optional[dict]:
        """Load feature cache for one rollout. Returns None if not cached yet.

        Reads .pt if present; else falls back to the parquet shards (lazy in-memory
        index built once per cell), so repair stages that read their own cache back
        work regardless of write format."""
        path = self.feature_cache_path(task, model_tag, pid, rollout_idx)
        if path.exists():
            data = torch.load(str(path), map_location="cpu", weights_only=True)
            return {k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
                    for k, v in data.items()}
        # parquet fallback
        idx = self._parquet_index(task, model_tag)
        return idx.get((str(pid), int(rollout_idx)))

    # ── parquet-shard feature cache (efficient format; opt-in via cache_format) ──

    _TENSOR_KEYS = ("Delta_path", "G_cov", "V", "J_approx", "pA_on_set", "logit_var",
                    "entropy", "kl_div", "logit_skew", "logit_kurt")

    def feature_shards(self, task: str, model_tag: str) -> List[Path]:
        return sorted(self.feature_cache_dir(task, model_tag).glob("shard_*.parquet"))

    def n_feature_shards(self, task: str, model_tag: str) -> int:
        return len(self.feature_shards(task, model_tag))

    def save_feature_shard(self, task: str, model_tag: str, payloads: List[dict],
                       shard_idx: int) -> None:
        """Write one parquet shard holding many rollouts (one row each). Per-token
        tensors → float32 list columns; scalars (pid/rollout_idx) kept as-is."""
        import polars as pl
        if not payloads:
            return
        d = self.feature_cache_dir(task, model_tag)
        rows = []
        for p in payloads:
            row = {}
            for k, v in p.items():
                if isinstance(v, torch.Tensor):
                    row[k] = v.float().cpu().tolist()   # 1-D → list; 2-D → list-of-lists
                else:
                    row[k] = v
            rows.append(row)
        pl.DataFrame(rows).write_parquet(str(d / f"shard_{shard_idx:04d}.parquet"))

    def feature_parquet_keys(self, task: str, model_tag: str) -> set:
        """{(pid, rollout_idx)} already present in parquet shards (for resume-skip)."""
        import polars as pl
        keys = set()
        for sh in self.feature_shards(task, model_tag):
            df = pl.read_parquet(str(sh), columns=["pid", "rollout_idx"])
            keys.update((str(a), int(b)) for a, b in zip(df["pid"].to_list(),
                                                          df["rollout_idx"].to_list()))
        return keys

    def _row_to_payload(self, row: dict) -> dict:
        """One parquet row → payload dict with float32 tensors (matching .pt loads)."""
        out = {}
        for k, v in row.items():
            if k in self._TENSOR_KEYS and v is not None:
                out[k] = torch.tensor(v, dtype=torch.float32)
            else:
                out[k] = v
        return out

    def _parquet_index(self, task: str, model_tag: str) -> dict:
        """Lazy {(pid,ridx): payload} index over all shards (cached on the instance)."""
        import polars as pl
        cache = getattr(self, "_pq_idx_cache", None)
        if cache is None:
            cache = self._pq_idx_cache = {}
        key = (task, model_tag)
        if key not in cache:
            idx = {}
            for sh in self.feature_shards(task, model_tag):
                for row in pl.read_parquet(str(sh)).to_dicts():
                    idx[(str(row.get("pid")), int(row.get("rollout_idx", 0)))] = \
                        self._row_to_payload(row)
            cache[key] = idx
        return cache[key]

    def iter_feature_rollouts(self, task: str, model_tag: str):
        """Yield per-rollout payload dicts from EITHER parquet shards or .pt files
        (parquet preferred when present). Unified reader for analysis/aggregation."""
        import polars as pl
        shards = self.feature_shards(task, model_tag)
        if shards:
            for sh in shards:
                for row in pl.read_parquet(str(sh)).to_dicts():
                    yield self._row_to_payload(row)
            return
        for f in sorted(self.feature_cache_dir(task, model_tag).glob("*_r*.pt")):
            data = torch.load(str(f), map_location="cpu", weights_only=True)
            yield {k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
                   for k, v in data.items()}
