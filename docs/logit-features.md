# Logit features

`cache_logits` stores, for **every generated token** of a failed trace, the features
below. They are computed at decode temperature `T` over the top-`K` logprobs that
vLLM returns, comparing the **specialist** (`p_S`) against the **ancestor** (`p_A`).
`x_t` is the token the rollout actually took at step `t`.

| Feature (cache key) | Definition | What it measures |
|---|---|---|
| `Delta_path` | `log p_S^T(x_t) − log p_A^T(x_t)` | **Path deformation.** How much more (or less) the specialist favored the *taken* token than the ancestor did. Positive ⇒ the specialist is over-confident on this token relative to its base. |
| `G_cov` | `log( Σ_{y∈A_t} p_A^T(y) / Σ_{y∈A_t} p_S^T(y) )`, where `A_t` = ancestor's top-`k_cov` tokens | **Coverage deformation.** How much probability mass the specialist pulled *off* the ancestor's preferred tokens. Large positive ⇒ the specialist contracted away from the base's support (lost coverage). |
| `logit_var` | `Var_{y∼p_S^T}[ z_S(y) ]` over the specialist's top-`K` (probability-weighted variance of the logits) | **Logit dispersion / temperature sensitivity.** The Fisher information of the temperature submodel. High ⇒ a peaked, confident distribution that responds strongly to a temperature change. |
| `J_approx` | `max(0, Delta_path) + max(0, G_cov)` | **Demotion pressure.** The combined per-token signal whose peak across the trace marks the **junction**. Not a Jacobian. |
| `kl_div` | `KL( p_S^T ‖ p_A^T )` over the specialist's top-`K` | **Distributional drift.** How far the specialist's whole next-token distribution has moved from the ancestor's at this step. |
| `entropy` | `H( p_S^T )` | **Local uncertainty.** How spread out the specialist's next-token distribution is at this step. |

**The junction** is the window of tokens around `argmax_t J_approx`: the place where
the failure was decided, and where the targeted operators (sparse logit steering,
local temperature lift) act.

## From per-token features to routing features

The per-problem routing features used by the prospective rule (see the README's
worked example 2) are trajectory aggregates of the two features above:

| Routing feature | Aggregate | Routes to |
|---|---|---|
| **spread** | `J_frac+`: fraction of tokens with `J_approx > 0` | dense steer |
| **concentration** | `log10( max_t J_approx / mean_t J_approx )` | sparse steer |
| **logit dispersion** | `log10( logit_var )` at the junction | local temperature lift |
