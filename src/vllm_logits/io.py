"""src/vllm_logits/io.py — the I/O contract that decouples the library from a repo.

Replaces the repo's `get_datamodule` / `ModelRegistry` / `ArtifactStore` tag
conventions with plain typed dicts + jsonl helpers + correctness defaults. The
engines already take problems / prompt_ids_map / rollouts as plain arguments;
this module just provides the shapes and a few ready-made correctness checkers.

Shapes:
    Problem = {"problem_id": str, "prompt": str}
    Rollout = {"problem_id": str, "rollout_idx": int,
               "generated_text": str, "is_correct": bool?}

Correctness defaults (a (problem, text) -> bool callback is the ONLY task-
specific input the library needs):
    exact_match(answer_key="answer")
    numeric_answer(answer_key="answer")   # \\boxed{} or last number
    regex(pattern)
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, TypedDict


class Problem(TypedDict):
    problem_id: str
    prompt: str


class Rollout(TypedDict, total=False):
    problem_id: str
    rollout_idx: int
    generated_text: str
    is_correct: bool


# ── jsonl helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def save_jsonl(rows: List[dict], path: str) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_problems(path: str) -> List[Problem]:
    """Load problems from jsonl; normalizes id/prompt key aliases."""
    raw = load_jsonl(path)
    out: List[Problem] = []
    for r in raw:
        pid = str(r.get("problem_id", r.get("pid", r.get("id", ""))))
        prompt = r.get("prompt", r.get("question", r.get("input", "")))
        out.append({"problem_id": pid, "prompt": prompt, **{
            k: v for k, v in r.items() if k not in ("problem_id", "prompt")}})
    return out


def load_rollouts(path: str) -> List[Rollout]:
    """Load rollouts from jsonl; normalizes id / rollout_idx key aliases."""
    raw = load_jsonl(path)
    out: List[Rollout] = []
    for i, r in enumerate(raw):
        pid = str(r.get("problem_id", r.get("pid", r.get("id", ""))))
        ridx = int(r.get("rollout_idx", r.get("rollout_id", r.get("sample_id", i))))
        rec: Rollout = {"problem_id": pid, "rollout_idx": ridx}
        if "generated_text" in r:
            rec["generated_text"] = r["generated_text"]
        elif "answer" in r:
            rec["generated_text"] = r["answer"]
        if "is_correct" in r:
            rec["is_correct"] = bool(r["is_correct"])
        for k, v in r.items():
            if k not in rec:
                rec[k] = v
        out.append(rec)
    return out


# ── correctness defaults ─────────────────────────────────────────────────────

CorrectnessFn = Callable[[Dict[str, Any], str], bool]


def _gold(problem: Dict[str, Any], answer_key: str) -> Optional[str]:
    v = problem.get(answer_key)
    return None if v is None else str(v)


def exact_match(answer_key: str = "answer") -> CorrectnessFn:
    """(problem, text) -> bool: gold string appears verbatim in the generation."""
    def _fn(problem: Dict[str, Any], text: str) -> bool:
        gold = _gold(problem, answer_key)
        if gold is None:
            return False
        return gold.strip() in text
    return _fn


_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_number(text: str) -> Optional[str]:
    m = list(_BOXED.finditer(text))
    if m:
        inner = m[-1].group(1)
        n = list(_NUM.finditer(inner))
        if n:
            return n[-1].group(0).replace(",", "")
    nums = list(_NUM.finditer(text))
    if nums:
        return nums[-1].group(0).replace(",", "")
    return None


def numeric_answer(answer_key: str = "answer") -> CorrectnessFn:
    """(problem, text) -> bool: last \\boxed{} / last number equals the gold number."""
    def _fn(problem: Dict[str, Any], text: str) -> bool:
        gold = _gold(problem, answer_key)
        if gold is None:
            return False
        gold_num = _extract_number(gold) or gold.strip()
        pred = _extract_number(text)
        if pred is None:
            return False
        try:
            return abs(float(pred) - float(gold_num)) < 1e-6
        except ValueError:
            return pred == gold_num
    return _fn


def regex(pattern: str, answer_key: str = "answer") -> CorrectnessFn:
    """(problem, text) -> bool: a regex match of `pattern` on the text equals gold.

    If `pattern` contains a capture group, the first group is compared to gold;
    otherwise the presence of a match is the criterion.
    """
    rx = re.compile(pattern)
    def _fn(problem: Dict[str, Any], text: str) -> bool:
        m = rx.search(text)
        if m is None:
            return False
        gold = _gold(problem, answer_key)
        if gold is None:
            return True  # presence-only criterion
        captured = m.group(1) if m.groups() else m.group(0)
        return captured.strip() == gold.strip()
    return _fn
