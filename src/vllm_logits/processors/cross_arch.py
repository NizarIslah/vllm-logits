"""Cross-architecture logit-injection processor.

logits from a DIFFERENT architecture (different tokenizer), handling dynamic
vocabulary mapping and out-of-vocab token masking.
"""
import torch
from typing import Dict, List


class CrossArchLogitInjectionAdapter:
    """vLLM LogitsProcessor injecting Ancestor logits from a different architecture."""

    def __init__(
        self,
        vocab_map: torch.Tensor,
        target_positions: Dict[int, torch.Tensor],
        alpha: float = 1.0,
        device: str = "cuda",
    ):
        """
        Args:
            vocab_map: Tensor of shape (V_spec,) where M[i] = anc_id. -1 for UNK.
            target_positions: Dict mapping sequence_length -> raw_anc_logits (V_anc,)
            alpha: Mixing coefficient. 1.0 = hard replacement, <1.0 = soft mixing.
        """
        self.device = device
        self.alpha = float(alpha)
        self.vocab_map = vocab_map.to(self.device)
        self.target_positions = {k: v.to(self.device) for k, v in target_positions.items()}

        self.safe_vocab_map = self.vocab_map.clone()
        self.unmapped_mask = (self.vocab_map == -1)

        if self.target_positions:
            v_anc = next(iter(self.target_positions.values())).shape[0]
            self.safe_vocab_map[self.unmapped_mask] = 0
            self.safe_vocab_map = torch.clamp(self.safe_vocab_map, min=0, max=v_anc - 1)
        else:
            self.safe_vocab_map[self.unmapped_mask] = 0

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        seq_len = len(token_ids)

        if seq_len not in self.target_positions:
            return logits

        raw_anc_logits = self.target_positions[seq_len]
        mapped_logits = raw_anc_logits[self.safe_vocab_map]

        if self.alpha >= 1.0:
            mapped_logits[self.unmapped_mask] = -torch.inf
            return mapped_logits
        else:
            safe_mapped_logits = mapped_logits.clone()
            safe_mapped_logits[self.unmapped_mask] = -1e4

            mixed_logits = (1.0 - self.alpha) * logits + self.alpha * safe_mapped_logits
            mixed_logits[self.unmapped_mask] = logits[self.unmapped_mask]

            return mixed_logits
