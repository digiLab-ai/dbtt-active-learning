"""Which uncertainty-engine is installed — the library, or the SDK?

Both distributions are published under the name ``uncertainty-engine`` and both
provide a top-level ``uncertainty_engine`` package, but they are different
things:

- the **library** (``uncertainty_engine.models``, ``.simulators``,
  ``.optimisers``) runs probabilistic modelling *in this process*;
- the **SDK** (``uncertainty_engine.client``, ``.nodes``, ``.graph``) is a thin
  client that builds workflows and runs them on the *remote* Uncertainty Engine.

They therefore **cannot be installed in the same environment** — one shadows the
other. This module reports which one is present so a notebook can take the right
path, and raises a message that says what to install when the wrong one is.

See the README for the two-environment layout this implies.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Literal, Optional

Backend = Literal["library", "sdk", "none"]

_LIBRARY_HINT = (
    "The in-process uncertainty-engine library is not installed in this "
    "environment. It is a sister repo, installed editable:\n"
    '    uv pip install -e "../uncertainty-engine[gptorch,metrics]"\n'
    "Note it cannot share an environment with the uncertainty-engine SDK: both "
    "distributions provide the top-level `uncertainty_engine` package."
)

_SDK_HINT = (
    "The uncertainty-engine SDK is not installed in this environment. Install it "
    "into a separate virtualenv:\n"
    "    uv venv .venv-sdk --python 3.12\n"
    '    uv pip install --python .venv-sdk -e ".[sdk]"\n'
    "It cannot share an environment with the in-process library: both "
    "distributions provide the top-level `uncertainty_engine` package."
)


def detect_backend() -> Backend:
    """Report which flavour of ``uncertainty_engine`` is importable.

    Returns ``"library"``, ``"sdk"`` or ``"none"``. Detection is by module
    layout rather than version, because the two share a distribution name.
    """
    if importlib.util.find_spec("uncertainty_engine") is None:
        return "none"
    for module, backend in (
        ("uncertainty_engine.models", "library"),
        ("uncertainty_engine.client", "sdk"),
    ):
        try:
            if importlib.util.find_spec(module) is not None:
                return backend  # type: ignore[return-value]
        except (ImportError, ValueError):
            continue
    return "none"


def backend_version() -> Optional[str]:
    """Installed ``uncertainty-engine`` distribution version, if any."""
    try:
        from importlib.metadata import version

        return version("uncertainty-engine")
    except Exception:
        return None


def has_library() -> bool:
    """Whether the in-process modelling library is available."""
    return detect_backend() == "library"


def has_sdk() -> bool:
    """Whether the remote-API SDK is available."""
    return detect_backend() == "sdk"


def require_library() -> None:
    """Raise a directive error unless the in-process library is installed."""
    if not has_library():
        raise ImportError(_LIBRARY_HINT)


def require_sdk() -> None:
    """Raise a directive error unless the SDK is installed."""
    if not has_sdk():
        raise ImportError(_SDK_HINT)


def describe() -> dict:
    """Summarise the environment, for printing at the top of a notebook."""
    backend = detect_backend()
    return {
        "backend": backend,
        "version": backend_version(),
        "in_process_modelling": backend == "library",
        "remote_workflows": backend == "sdk",
    }
