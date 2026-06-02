"""
Adaptive alpha signals for position-wise logit/prob mixing.

All functions are pure (no I/O, no vLLM imports). Inputs are raw logits tensors.
Callers are responsible for softmax before passing to chi2_divergence / prob mixing.
"""

import torch
import torch.nn.functional as F

EPS = 1e-10


def entropy_gap(logits_spec: torch.Tensor, logits_anc: torch.Tensor) -> torch.Tensor:
    """
    Entropy gap signal for logit mixing (e-geodesic).

    Positive where specialist is sharper than ancestor — those positions get
    more ancestor weight (alpha_t closer to 1).

    Args:
        logits_spec: (..., vocab_size) raw specialist logits
        logits_anc:  (..., vocab_size) raw ancestor logits

    Returns:
        (...,) T-invariant scalar per position
    """
    p_spec = F.softmax(logits_spec.float(), dim=-1)
    p_anc  = F.softmax(logits_anc.float(),  dim=-1)
    H_spec = -(p_spec * torch.log(p_spec + EPS)).sum(dim=-1)
    H_anc  = -(p_anc  * torch.log(p_anc  + EPS)).sum(dim=-1)
    return H_spec - H_anc  # positive → spec sharper → more ancestor injection


def chi2_divergence(logits_spec: torch.Tensor, logits_anc: torch.Tensor) -> torch.Tensor:
    """
    Chi-squared divergence chi2(p_anc || p_spec) for prob mixing (m-geodesic).

    Weights tokens where the ancestor placed mass the specialist suppressed
    (the support-contraction signal). Asymmetry is intentional.

    Args:
        logits_spec: (..., vocab_size) raw specialist logits
        logits_anc:  (..., vocab_size) raw ancestor logits

    Returns:
        (...,) T-invariant non-negative scalar per position
    """
    p_spec = F.softmax(logits_spec.float(), dim=-1)
    p_anc  = F.softmax(logits_anc.float(),  dim=-1)
    return (p_anc.pow(2) / (p_spec + EPS)).sum(dim=-1) - 1.0


import math as _math

# Logit of the default base alpha — used as the sigmoid bias so that
# adaptive_alpha returns base_alpha when the signal is exactly zero.
_LOGIT_07 = _math.log(0.7 / 0.3)  # ≈ 0.8473


def adaptive_alpha(
    signal: torch.Tensor,
    T: float = 1.0,
    base_alpha: float = 0.7,
) -> torch.Tensor:
    """
    Convert a raw signal to a per-position alpha (specialist weight) via
    sigmoid(clamp(signal/T + logit(base_alpha), -4.6, 4.6)).

    The bias logit(base_alpha) shifts the sigmoid centre so that alpha_t equals
    base_alpha when the signal is zero — making adaptive alpha directly
    comparable to the fixed alpha=base_alpha baseline.

    Convention (consistent with existing fixed-alpha code):
      alpha_t → 1.0 = pure specialist, alpha_t → 0.0 = pure ancestor.

    Callers must negate the raw signal when the signal increases with forgetting:
      - Logit mixing: pass  -entropy_gap  (= H_anc - H_spec)
      - Prob mixing:  pass  -chi2

    Args:
        signal:     (...,) pre-negated signal values
        T:          temperature; larger T → softer alpha range
        base_alpha: centre of the sigmoid (default 0.7 to match fixed-alpha baseline)

    Returns:
        (..., 1) broadcast-ready alpha in (0, 1)
    """
    bias = _math.log(base_alpha / (1.0 - base_alpha))
    clamped = torch.clamp(signal / T + bias, min=-4.6, max=4.6)
    return torch.sigmoid(clamped).unsqueeze(-1)


def mixing_diagnostics(
    logits_spec: torch.Tensor,
    logits_anc: torch.Tensor,
    alpha_t: torch.Tensor,
) -> dict:
    """
    Compute entropy and alpha statistics for a batch of positions.

    Args:
        logits_spec: (..., vocab_size)
        logits_anc:  (..., vocab_size)
        alpha_t:     (..., 1) as returned by adaptive_alpha

    Returns:
        dict with H_anc, H_spec, H_mix (means), alpha_mean, alpha_std,
        frac_dominant (fraction of positions where alpha_t > 0.5)
    """
    p_spec = F.softmax(logits_spec.float(), dim=-1)
    p_anc  = F.softmax(logits_anc.float(),  dim=-1)
    p_mix  = alpha_t * p_anc + (1.0 - alpha_t) * p_spec

    def H(p):
        return -(p * torch.log(p + EPS)).sum(dim=-1)

    a = alpha_t.squeeze(-1)
    return dict(
        H_anc=H(p_anc).mean().item(),
        H_spec=H(p_spec).mean().item(),
        H_mix=H(p_mix).mean().item(),
        alpha_mean=a.mean().item(),
        alpha_std=a.std().item(),
        frac_dominant=(a > 0.5).float().mean().item(),
    )
