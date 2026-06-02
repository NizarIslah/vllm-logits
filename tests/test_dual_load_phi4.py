"""GPU arch-fix canary: confirm DualQwen2ForCausalLM (phi3 branch) loads
Phi-4-mini dual-stack and produces the exact pinned 5-token greedy output
' John. I am a'.

Phi-3/Phi-4-mini
use the LlamaModel backbone (selected via model_type=="phi3" in _compat); the
dot-anchored stacked-params mapping is what this test guards.

Run (GPU):
    export VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
    PYTHONPATH=src python -m pytest tests/test_dual_load_phi4.py -s
"""
import os

PHI4_ID = os.environ.get("VLLM_LOGITS_PHI4_PATH", "microsoft/Phi-4-mini-instruct")
# Resolve HF id -> local snapshot (this canary builds the dual config via filesystem
# ops). NOTE: the pinned greedy output was validated on a pre-split local copy; with
# the stock HF snapshot the phi3-backbone path is exercised but the exact token string
# may differ — treat a non-empty, non-EOS dual generation as the portability check and
# the exact-string assert as local (set VLLM_LOGITS_PHI4_PATH to the split copy).
if os.path.isdir(PHI4_ID):
    PHI4_PATH = PHI4_ID
else:
    from huggingface_hub import snapshot_download
    PHI4_PATH = snapshot_download(PHI4_ID)
EXPECTED_TEXT = " John. I am a"

os.environ["VLLM_LOGIT_MIX_ANCESTOR"] = PHI4_PATH
os.environ["VLLM_LOGIT_MIX_SPECIALIST"] = PHI4_PATH
os.environ["VLLM_LOGIT_MIX_ALPHA"] = "0.5"
os.environ["VLLM_LOGIT_MIX_A_LAYERS"] = "32"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

import shutil
import tempfile

from transformers import AutoConfig, AutoTokenizer

import vllm_logits.register as register
register.register_dual_qwen()  # registers DualQwen2ForCausalLM (covers phi3)


def _run():
    print("[verify] Building Frankenstein config (32+32=64 layers)...")
    tmp = tempfile.mkdtemp()
    try:
        cfg = AutoConfig.from_pretrained(PHI4_PATH, trust_remote_code=True)
        cfg.architectures = ["DualQwen2ForCausalLM"]
        if hasattr(cfg, "layer_types") and cfg.layer_types:
            cfg.layer_types = list(cfg.layer_types) * 2
        cfg.num_hidden_layers = 64
        cfg.save_pretrained(tmp)
        tok = AutoTokenizer.from_pretrained(PHI4_PATH, trust_remote_code=True)
        tok.save_pretrained(tmp)
        for fname in os.listdir(PHI4_PATH):
            src = os.path.join(PHI4_PATH, fname)
            dst = os.path.join(tmp, fname)
            if os.path.exists(dst):
                continue
            if fname.endswith((".safetensors", ".bin")):
                os.symlink(src, dst)

        from vllm import LLM, SamplingParams
        print(f"[verify] Initializing vLLM with dual Phi-4-mini from {PHI4_PATH}...")
        llm = LLM(
            model=tmp,
            tokenizer=PHI4_PATH,
            dtype="bfloat16",
            enforce_eager=True,
            disable_custom_all_reduce=True,
            gpu_memory_utilization=0.5,
            max_model_len=512,
            tensor_parallel_size=1,
            trust_remote_code=True,
        )
        sp = SamplingParams(temperature=0.0, max_tokens=5)
        out = llm.generate(["Hello, my name is"], sp)
        text = out[0].outputs[0].text
        ids = list(out[0].outputs[0].token_ids)
        print(f"[verify] Generated text: {text!r}")
        print(f"[verify] Generated token_ids: {ids}")

        assert len(text) > 0, "Empty generation - dual-load broken"
        assert ids[0] != tok.eos_token_id, f"First token was EOS ({ids[0]}) - misload"
        assert text == EXPECTED_TEXT, f"Greedy output {text!r} != expected {EXPECTED_TEXT!r}"
        print("[verify] PASS - Phi-4-mini dual-load exact greedy output reproduced.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dual_load_phi4():
    _run()


if __name__ == "__main__":
    _run()
