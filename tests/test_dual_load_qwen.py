"""GPU arch-fix canary: confirm DualQwen2ForCausalLM loads Qwen3 dual-stack
and produces the exact pinned 5-token greedy output " Lina. I'm".

This is the
vLLM-version-bump canary for the qwen3 backbone: if vLLM moves the internals
that `backbones.py`/`_compat.py` touch, this test fails loudly.

Run (GPU):
    export VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
    PYTHONPATH=src python -m pytest tests/test_dual_load_qwen.py -s
"""
import os

QWEN3_ID = os.environ.get("VLLM_LOGITS_QWEN3_PATH", "Qwen/Qwen3-0.6B")
# This canary does manual filesystem ops to build the dual ("Frankenstein") config,
# so it needs a real directory. Resolve an HF id -> local snapshot (no-op if already
# a local path). The library's normal path (LogitPipeline) resolves ids via vLLM.
if os.path.isdir(QWEN3_ID):
    QWEN3_PATH = QWEN3_ID
else:
    from huggingface_hub import snapshot_download
    QWEN3_PATH = snapshot_download(QWEN3_ID)
EXPECTED_TEXT = " Lina. I'm"

os.environ["VLLM_LOGIT_MIX_ANCESTOR"] = QWEN3_PATH
os.environ["VLLM_LOGIT_MIX_SPECIALIST"] = QWEN3_PATH
os.environ["VLLM_LOGIT_MIX_ALPHA"] = "0.5"
os.environ["VLLM_LOGIT_MIX_A_LAYERS"] = "28"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

import shutil
import tempfile

from transformers import AutoConfig, AutoTokenizer

import vllm_logits.register as register
register.register_dual_qwen()  # registers DualQwen2ForCausalLM


def _run():
    print("[verify] Building Frankenstein config (28+28=56 layers)...")
    tmp = tempfile.mkdtemp()
    try:
        cfg = AutoConfig.from_pretrained(QWEN3_PATH, trust_remote_code=True)
        cfg.architectures = ["DualQwen2ForCausalLM"]
        if hasattr(cfg, "layer_types") and cfg.layer_types:
            cfg.layer_types = list(cfg.layer_types) * 2
        cfg.num_hidden_layers = 56
        cfg.save_pretrained(tmp)
        tok = AutoTokenizer.from_pretrained(QWEN3_PATH, trust_remote_code=True)
        tok.save_pretrained(tmp)
        for fname in os.listdir(QWEN3_PATH):
            src = os.path.join(QWEN3_PATH, fname)
            dst = os.path.join(tmp, fname)
            if os.path.exists(dst):
                continue
            if fname.endswith((".safetensors", ".bin")):
                os.symlink(src, dst)

        from vllm import LLM, SamplingParams
        print("[verify] Initializing vLLM with dual Qwen3-0.6B...")
        llm = LLM(
            model=tmp,
            tokenizer=QWEN3_PATH,
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
        print("[verify] PASS - Qwen3 dual-load exact greedy output reproduced.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dual_load_qwen():
    _run()


if __name__ == "__main__":
    _run()
