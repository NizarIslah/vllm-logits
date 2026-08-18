---
name: Feature request
about: Propose a capability or an architecture
labels: enhancement
---

**What you are trying to do**: the problem, not the implementation.

**Why the current API cannot do it**

**Sketch of what the API would look like**

```python
```

**Adding a model family?** That path is short and documented: subclass the dual backbone, add a
`register_*` line, add the `model_type` entry. Say which `model_type` you need, and note that the
specialist and ancestor must share an architecture since both are loaded into one vLLM model.
