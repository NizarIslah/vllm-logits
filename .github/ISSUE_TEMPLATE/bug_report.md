---
name: Bug report
about: Something does not work as documented
labels: bug
---

**What happened**

**What you expected**

**Reproduction**: the smallest thing that shows it. If it involves a model, include both model ids
(specialist and ancestor) and whether they are local paths or HF ids.

```python
```

**Environment**
- `vllm-logits` version:
- install: core only / `[engine]` / `[demo]`
- `python -c "import vllm; print(vllm.__version__)"` (if the engine path is involved):
- GPU (if relevant):

**Notes**
If this is a vLLM version issue, the dual-load tests are the fastest signal:
`pytest tests/test_dual_load_qwen.py -s`.
