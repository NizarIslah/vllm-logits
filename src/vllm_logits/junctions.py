"""Junction detectors + offline firing on cached features.

`find_firing_position` is folded in at
the bottom of this module.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict, Tuple
import math
import torch


@dataclass
class JunctionContext:
    """Aggregates all possible data sources for detectors."""
    problem_id: str
    rollout_id: int
    fail_token_ids: List[int] # Entire rollout ids
    succ_token_ids: Optional[List[int]] = None
    ancestor_logits: Optional[torch.Tensor] = None # Current step logits [V]
    specialist_logits: Optional[torch.Tensor] = None # Current step logits [V]
    step: int = 0 # Relative step in generation
    pos: int = 0  # Absolute position in sequence
    # Relative metrics (populated by engine)
    demotion_z_score: float = 0.0
    coverage_z_score: float = 0.0
    is_top_percentile: bool = False
    is_top_coverage_percentile: bool = False
    # Raw demotion score before z-normalisation (Δ_t = log p_S(x_t) - log p_A(x_t))
    d_t_raw: float = 0.0
    # Raw coverage gap score before z-normalisation
    c_t_raw: float = 0.0
    # Logit variance for entropy bottleneck detection: Var_{y ~ p_S}(z_S(y))
    v_t_raw: float = 0.0


@dataclass
class RepairContext:
    """Per-position data for the Local Excess Deformation scan statistic.

    Populated by the cache_logits engine from pre-run forward passes; consumed
    by the scoring layer during online generation to estimate background
    stats and run the scan statistic on rollout k+1.

    All Tensor fields are 1-D with length = rollout_token_length.
    """
    pid: str
    rollout_idx: int
    # e-geodesic: D_t^path = log p_S(x_t) - log p_A(x_t) per position
    Delta_path: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    # m-geodesic: log(p_A(A_t^k) / (p_S(A_t^k) + ε)) per position
    G_cov: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    # Probability-weighted logit variance (NOT .var()) per position
    V: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    # Approximate J_t before full normalisation (used for quantile estimation)
    J_approx: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    # Top-K log-probs and raw logits for first-order approximation check
    log_pS_topk: Optional[torch.Tensor] = None   # [L, K]
    log_pA_topk: Optional[torch.Tensor] = None   # [L, K]
    logits_S_topk: Optional[torch.Tensor] = None  # [L, K]
    logits_A_topk: Optional[torch.Tensor] = None  # [L, K]
    # Specialist / ancestor mass on ancestor top-k set
    pS_on_set: Optional[torch.Tensor] = None  # [L]
    pA_on_set: Optional[torch.Tensor] = None  # [L]


class JunctionDetector(ABC):
    @abstractmethod
    def detect(self, context: JunctionContext) -> bool:
        """Returns True if a junction is identified at current context.step."""
        pass

    def reset_state(self, problem_id: str, rollout_id: int) -> None:
        """Called before each new rollout. Override for stateful detectors."""
        pass


class OutcomeContrastDetector(JunctionDetector):
    """Detects junction where fail and success rollouts first diverge, skipping prefix matches."""
    def __init__(self, skip_thinking: bool = True):
        self.skip_thinking = skip_thinking

    def detect(self, context: JunctionContext) -> bool:
        if context.succ_token_ids is None:
            return False

        # Optionally skip early divergence in the thinking tag
        # Justification: Stylistic jitter in boilerplate.
        if self.skip_thinking and context.step < 5:
            return False

        if context.step >= len(context.succ_token_ids) or context.step >= len(context.fail_token_ids):
            return False

        return context.fail_token_ids[context.step] != context.succ_token_ids[context.step]


class AncestorDivergenceDetector(JunctionDetector):
    """Detects where Specialist greedy choice diverges from Ancestor."""
    def detect(self, context: JunctionContext) -> bool:
        if context.ancestor_logits is None or context.specialist_logits is None:
            return False
        return context.specialist_logits.argmax() != context.ancestor_logits.argmax()


class UncertaintyForkDetector(JunctionDetector):
    """Detects where Specialist is uncertain (Reference-free)."""
    def __init__(self, entropy_threshold: float = 2.0):
        self.tau = entropy_threshold

    def detect(self, context: JunctionContext) -> bool:
        if context.specialist_logits is None:
            return False
        p = torch.softmax(context.specialist_logits.float(), dim=-1)
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        return entropy > self.tau


class DemotionScoreDetector(JunctionDetector):
    """Fires at first token where Specialist deviates from Ancestor.

    Modes:
    - fixed: D_t = log_S(x_t) - log_A(x_t) > threshold (absolute)
    - relative: D_t is in the top 90th percentile of the current trace.
    """
    def __init__(self, threshold: float = 1.0, min_step: int = 20, mode: str = "relative"):
        self.threshold = threshold
        self.min_step = min_step
        self.mode = mode

    def detect(self, context: JunctionContext) -> bool:
        if context.step < self.min_step:
            return False

        if self.mode == "relative":
            return context.is_top_percentile

        # Legacy Fixed Mode
        if context.ancestor_logits is None or context.specialist_logits is None:
            return False

        ls = torch.log_softmax(context.specialist_logits.float(), dim=-1)
        la = torch.log_softmax(context.ancestor_logits.float(), dim=-1)
        tok_id = context.fail_token_ids[context.step]
        return (ls[tok_id] - la[tok_id].to(ls.device)).item() > self.threshold

    def select_top_junctions(self, contexts: List[JunctionContext], k: int = 1) -> List[int]:
        """Ranked selection for Iterative Repair."""
        scores = []
        for ctx in contexts:
            if ctx.ancestor_logits is None or ctx.specialist_logits is None:
                scores.append(-float("inf"))
                continue
            ls = torch.log_softmax(ctx.specialist_logits.float(), dim=-1)
            la = torch.log_softmax(ctx.ancestor_logits.float(), dim=-1)
            tok_id = ctx.fail_token_ids[ctx.step]
            # Ensure tensors are on the same device
            scores.append((ls[tok_id] - la[tok_id].to(ls.device)).item())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return top_indices[:k]


class ProbCoverageDetector(JunctionDetector):
    """Fires at first token where Specialist has a significant coverage gap vs Ancestor.

    C_t = sum_{x in TopK_A} [p_A(x) - p_S(x)]_+
    """
    def __init__(self, threshold: float = 1.0, min_step: int = 20, mode: str = "relative", top_k: int = 5):
        self.threshold = threshold
        self.min_step = min_step
        self.mode = mode
        self.top_k = top_k

    def detect(self, context: JunctionContext) -> bool:
        if context.step < self.min_step:
            return False

        if self.mode == "relative":
            return context.is_top_coverage_percentile

        return context.coverage_z_score > self.threshold


class MatchedPositionDetector(JunctionDetector):
    """Fires at a random position within a specific depth bucket relative to a reference.

    The window is typically 10% of the total sequence length.
    """
    def __init__(self, ref_step: int, total_len: int, window_pct: float = 0.10, seed: int = 42):
        self.ref_step = ref_step
        self.total_len = total_len
        self.window_pct = window_pct
        self.seed = seed
        self._cache = {} # (pid, rid, ref) -> step

    def detect(self, context: JunctionContext) -> bool:
        key = (context.problem_id, context.rollout_id, self.ref_step)
        if key not in self._cache:
            import random
            # Deterministic for the same (problem, rollout, ref_step)
            rng = random.Random(f"{self.seed}_{context.problem_id}_{context.rollout_id}_{self.ref_step}")

            # 10% of sequence length (e.g. 100 tokens for length 1000)
            window = max(2, int(self.window_pct * self.total_len))
            low = max(0, self.ref_step - window // 2)
            high = min(self.total_len - 1, self.ref_step + window // 2)

            self._cache[key] = rng.randint(low, high)

        return context.step == self._cache[key]


class RandomPositionDetector(JunctionDetector):
    """Fires at a single random position in the rollout.

    To ensure we only get ONE junction per trace during extraction,
    the engine's extract loop should call this.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._cache = {} # (pid, rid) -> step

    def detect(self, context: JunctionContext) -> bool:
        key = (context.problem_id, context.rollout_id)
        if key not in self._cache:
            import random
            rng = random.Random(f"{self.seed}_{context.problem_id}_{context.rollout_id}")
            # Sample from [0, L-1]
            max_len = len(context.fail_token_ids)
            if max_len <= 1:
                self._cache[key] = 0
            else:
                self._cache[key] = rng.randint(0, max_len - 1)

        return context.step == self._cache[key]


class PageHinkleyDetector(JunctionDetector):
    """Online EWMA + Page-Hinkley detector on Δ_t = log p_S(x_t) - log p_A(x_t).

    Fires once per rollout at the first token where the cumulative excess above
    the running EWMA mean exceeds the calibrated threshold λ.

    β=0.05 fixed: effective window ~20 tokens ≈ one reasoning clause.
    δ=0: increment is pure excess above running mean (no free offset parameter).
    warmup = ceil(1/β) = 20: skips the initial transient while EWMA settles.
    λ: set by FAR calibration on correct rollouts (target FAR ≤ 5%).

    Call reset_state() before each new rollout.
    """

    def __init__(self, lam: float, beta: float = 0.05, min_step: int = 20, signal_type: str = "demotion"):
        self.lam = lam
        self.beta = beta
        self.warmup = math.ceil(1.0 / beta)  # 20 for beta=0.05
        self.min_step = min_step
        self.signal_type = signal_type
        self._reset()

    def _reset(self):
        self._mu: Optional[float] = None
        self._g: float = 0.0
        self._t: int = 0
        self._fired: bool = False
        self.last_fire_info: dict = {}  # populated at fire time: {g, mu, d_t, step}

    def reset_state(self, problem_id: str, rollout_id: int) -> None:
        self._reset()

    def detect(self, context: JunctionContext) -> bool:
        if self._fired:
            return False

        # Select the primary signal - ALWAYS use raw values for PH accumulator
        # PH internally maintains an EWMA mean of these raw values.
        if self.signal_type == "coverage":
            val_t = context.c_t_raw
        else:
            val_t = context.d_t_raw

        self._mu = val_t if self._mu is None else (1 - self.beta) * self._mu + self.beta * val_t
        self._t += 1
        if self._t >= self.warmup and context.step >= self.min_step:
            self._g = max(0.0, self._g + (val_t - self._mu))
            if self._g > self.lam:
                self._fired = True
                self.last_fire_info = {
                    "g_at_fire": self._g,
                    "mu_at_fire": self._mu,
                    "signal_at_fire": val_t,
                    "d_t_at_fire": context.d_t_raw,
                    "v_t_at_fire": context.v_t_raw,
                    "c_t_at_fire": context.c_t_raw,
                    "step": context.step,
                }
                return True
        return False


def calibrate_ph_threshold(
    correct_rollout_d_sequences: List[List[float]],
    beta: float = 0.05,
    lam_grid: Optional[List[float]] = None,
    target_far: float = 0.05,
    min_step: int = 20,
    failed_rollout_d_sequences: Optional[List[List[float]]] = None,
) -> dict:
    """Calibrate λ for PageHinkleyDetector.

    When failed_rollout_d_sequences is provided, uses joint calibration:
    picks λ* = argmax Youden's J = TPR(λ) − FAR(λ), subject to FAR(λ) ≤ target_far.
    This directly maximises discrimination between correct and failed rollouts.

    When failed_rollout_d_sequences is None, falls back to FAR-only: smallest λ
    where FAR ≤ target_far (original behaviour, fully backward compatible).

    Args:
        correct_rollout_d_sequences: Δ_t sequences for correct rollouts.
        failed_rollout_d_sequences: Δ_t sequences for failed rollouts (joint calib).
        beta: EWMA decay (fixed at 0.05).
        lam_grid: Candidate λ values.
        target_far: Maximum allowable FAR on correct rollouts.
        min_step: Minimum step before detector can fire (warmup guard).

    Returns dict with keys: chosen_lam, far_curve, beta, warmup, n_correct,
        target_far, g_maxes, lam_at_limit. Joint calibration also adds:
        tpr_curve, g_maxes_fail, n_fail, youdens_j_at_chosen.
    """
    if lam_grid is None:
        lam_grid = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0]
    warmup = math.ceil(1.0 / beta)

    def _compute_g_maxes(d_sequences):
        g_maxes = []
        for d_seq in d_sequences:
            mu = None
            g = 0.0
            g_max = 0.0
            for t, d_t in enumerate(d_seq):
                mu = d_t if mu is None else (1 - beta) * mu + beta * d_t
                if t >= warmup:
                    g = max(0.0, g + (d_t - mu))
                    g_max = max(g_max, g)
            g_maxes.append(g_max)
        return g_maxes

    g_maxes = _compute_g_maxes(correct_rollout_d_sequences)
    n = len(g_maxes)

    far_curve = {}
    for lam in lam_grid:
        far_curve[lam] = sum(1 for gm in g_maxes if gm > lam) / n if n > 0 else 0.0

    result = {
        "far_curve": {str(k): v for k, v in far_curve.items()},
        "beta": beta,
        "warmup": warmup,
        "n_correct": n,
        "target_far": target_far,
        "g_maxes": g_maxes,
    }

    if failed_rollout_d_sequences is not None:
        # Joint calibration: Youden's J within FAR budget
        g_maxes_fail = _compute_g_maxes(failed_rollout_d_sequences)
        n_fail = len(g_maxes_fail)
        tpr_curve = {}
        for lam in lam_grid:
            tpr_curve[lam] = sum(1 for gm in g_maxes_fail if gm > lam) / n_fail if n_fail > 0 else 0.0

        result["g_maxes_fail"] = g_maxes_fail
        result["n_fail"] = n_fail
        result["tpr_curve"] = {str(k): v for k, v in tpr_curve.items()}

        # Pick λ* = argmax J subject to FAR ≤ target_far
        if n == 0 or n_fail == 0:
            chosen_lam = max(lam_grid)
            lam_at_limit = True
            youdens_j = float("nan")
        else:
            feasible = [(lam, tpr_curve[lam] - far_curve[lam]) for lam in sorted(lam_grid)
                        if far_curve[lam] <= target_far]
            if not feasible:
                chosen_lam = max(lam_grid)
                lam_at_limit = True
                youdens_j = float("nan")
            else:
                # argmax J; ties broken by smaller λ (already sorted ascending)
                chosen_lam, youdens_j = max(feasible, key=lambda x: x[1])
                lam_at_limit = False

        result["chosen_lam"] = chosen_lam
        result["lam_at_limit"] = lam_at_limit
        result["youdens_j_at_chosen"] = youdens_j

    else:
        # FAR-only: smallest λ where FAR ≤ target_far
        if n == 0:
            chosen_lam = max(lam_grid)
            lam_at_limit = True
        else:
            lam_at_limit = False
            chosen_lam = max(lam_grid)
            for lam in sorted(lam_grid):
                if far_curve[lam] <= target_far:
                    chosen_lam = lam
                    break
            else:
                lam_at_limit = True

        result["chosen_lam"] = chosen_lam
        result["lam_at_limit"] = lam_at_limit

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Offline firing on cached per-rollout features
# ══════════════════════════════════════════════════════════════════════════════

def find_firing_position(
    cache: dict,
    background: dict,
    w: int = 1,
    warmup: int = 20,
) -> Optional[Tuple[int, str]]:
    """Return (t_hat, tau_hat) for the first firing in this cache, or None.

    Walks the windowed scan statistic S_t = sum_{i=t-w+1..t} J_approx[i] over a
    cached failed-rollout feature dict and returns the first position where
    S_t > lambda_J (with t >= warmup and a full window). Mirrors the firing rule
    in the repair processor (geo_pred branch) but operates on cached tensors
    instead of live generation.

    Args:
        cache: dict loaded by LogitStore.load_feature_cache() — must have
               Delta_path, G_cov, J_approx as 1-D float tensors of length T.
        background: dict returned by estimate_background() — must have
                    lambda_J, mu_path, sig_path, mu_cov, sig_cov.
        w: window size for S_t (matches the configured window).
        warmup: minimum token index before firing is allowed.
    """
    J = cache["J_approx"].tolist()
    Z_path_raw = cache["Delta_path"].tolist()
    G_cov_raw = cache["G_cov"].tolist()

    lambda_J = float(background["lambda_J"])
    mu_path = float(background["mu_path"])
    sig_path = float(background["sig_path"]) + 1e-6
    mu_cov = float(background["mu_cov"])
    sig_cov = float(background["sig_cov"]) + 1e-6

    window = []
    for t, j in enumerate(J):
        window.append(j)
        if len(window) > w:
            window.pop(0)
        if t < warmup or len(window) < w:
            continue
        S_t = sum(window)
        if S_t > lambda_J:
            zp = max(0.0, (Z_path_raw[t] - mu_path) / sig_path)
            zc = max(0.0, (G_cov_raw[t] - mu_cov) / sig_cov)
            tau = "path" if zp >= zc else "cov"
            return t, tau
    return None
