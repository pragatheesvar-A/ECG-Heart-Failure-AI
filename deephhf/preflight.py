"""Non-mutating runtime checks for DeepHHF-Inspired."""
from __future__ import annotations

import importlib
import platform
import sys


REQUIRED_MODULES = ("numpy", "pandas", "yaml", "scipy", "sklearn", "matplotlib", "torch", "streamlit")


def verify_setup() -> dict:
    modules = {}
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            modules[name] = {"available": True, "version": getattr(module, "__version__", "unknown")}
        except Exception as error:
            modules[name] = {"available": False, "error": str(error)}
    torch = importlib.import_module("torch") if modules["torch"]["available"] else None
    return {
        "ready": all(item["available"] for item in modules.values()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "modules": modules,
        "cuda_available": bool(torch and torch.cuda.is_available()),
    }
