#!/usr/bin/env python3
"""Fail if `vllm_logits` grows a dependency it must not have.

Why this exists
---------------
This package is meant to stay usable as a library: importable on a laptop, embeddable in someone
else's pipeline, and free of any assumption about how *we* run experiments. That property is easy to
state and easy to lose. One convenient import of a config framework or an experiment-harness helper
and it is gone. This check makes the boundary mechanical.

Rules
-----
* Every third-party import must be in ALLOWED.
* Tier-0 modules (importable with numpy alone) may not import torch, vLLM, or anything heavier.
* No module may import a research-harness module, a config framework, or read a private env var.

Run: python tools/check_import_boundary.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "vllm_logits"

#: Third-party packages this library may depend on, at any tier.
ALLOWED = {
    "numpy", "torch", "vllm", "transformers", "polars", "pyarrow", "safetensors",
    "huggingface_hub",
    "matplotlib",  # demo plotting only, imported inside a function
}

#: Modules that must import with numpy alone: no torch, no vLLM.
TIER0 = {"routing.py", "io.py", "demo.py", "__init__.py"}
TIER0_FORBIDDEN = {"torch", "vllm", "transformers", "safetensors", "polars", "pyarrow"}

#: Things that would tie this library to one lab's infrastructure.
FORBIDDEN_ALWAYS = {
    "hydra", "omegaconf",          # a library must not own a config framework
    "src",                         # research-harness packages
    "slurm", "submitit",
}
#: Environment variables that belong to a private experiment harness, not a public library.
FORBIDDEN_ENV_PREFIXES = ("SFR_", "SCRATCH", "SLURM_")

STDLIB = set(sys.stdlib_module_names)


def top(name: str | None) -> str:
    return (name or "").split(".")[0]


def check_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    errs: list[str] = []
    tier0 = path.name in TIER0

    for node in ast.walk(tree):
        mods: list[tuple[str, int]] = []
        if isinstance(node, ast.Import):
            mods = [(top(a.name), node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:           # relative import, always fine
                continue
            mods = [(top(node.module), node.lineno)]

        for mod, line in mods:
            where = f"{path.name}:{line}"
            if mod in FORBIDDEN_ALWAYS:
                errs.append(f"{where}: forbidden import {mod!r}. That belongs to the caller, "
                            f"not to this library")
            elif mod in STDLIB or mod == "vllm_logits":
                continue
            elif mod not in ALLOWED:
                errs.append(f"{where}: import {mod!r} is not in the allowlist. Add it to ALLOWED "
                            f"in tools/check_import_boundary.py and to pyproject, or drop it.")
            elif tier0 and mod in TIER0_FORBIDDEN:
                errs.append(f"{where}: {path.name} is Tier 0 (must import with numpy alone) but "
                            f"imports {mod!r}. Move the symbol behind the lazy loader in "
                            f"__init__.py instead.")

        # os.environ["SFR_..."] / os.getenv("SCRATCH") style reads
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith(FORBIDDEN_ENV_PREFIXES) and node.value.isupper():
                errs.append(f"{path.name}:{node.lineno}: references {node.value!r}, a private "
                            f"harness env var. Take a path or value as an argument instead.")
    return errs


def main() -> int:
    files = sorted(SRC.rglob("*.py"))
    if not files:
        print(f"no modules found under {SRC}")
        return 1
    errs = [e for f in files for e in check_file(f)]
    if errs:
        print(f"import-boundary violations ({len(errs)}):\n")
        for e in errs:
            print(f"  {e}")
        print("\nThis library must stay embeddable: no config framework, no harness modules, "
              "no private env vars, and Tier 0 stays numpy-only.")
        return 1
    print(f"import boundary OK: {len(files)} modules, "
          f"{len(TIER0)} of them Tier 0 (numpy-only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
