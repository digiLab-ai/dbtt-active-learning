"""Is the uncertainty-engine SDK installed?

This repo talks to the Uncertainty Engine over its **SDK** — a remote API client
that builds workflow graphs and runs them on the Engine. It is published to PyPI
as ``uncertainty-engine`` and provides a top-level ``uncertainty_engine``
package.

A second, unpublished distribution of the same name provides an *in-process*
modelling library (``uncertainty_engine.models``, ``.optimisers``). This repo no
longer uses it, but the two still cannot share an environment — one shadows the
other — so detection is by module layout rather than by version, and finding the
library where the SDK should be is reported as such rather than as a confusing
``ImportError`` deep in a workflow builder.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Literal, Optional

Backend = Literal["sdk", "library", "none"]

_SDK_HINT = (
    "The uncertainty-engine SDK is not installed in this environment. Install it "
    "into a separate virtualenv:\n"
    "    uv venv .venv-sdk --python 3.12\n"
    '    uv pip install --python .venv-sdk -e ".[sdk]"\n'
    "If the in-process modelling library is installed here instead, it shadows "
    "the SDK: both distributions provide the top-level `uncertainty_engine` "
    "package, so they cannot share an environment."
)


def detect_backend() -> Backend:
    """Report which flavour of ``uncertainty_engine`` is importable.

    Returns ``"sdk"``, ``"library"`` or ``"none"``. ``"library"`` means the
    in-process distribution is installed where the SDK is wanted, which this
    repo cannot use.
    """
    if importlib.util.find_spec("uncertainty_engine") is None:
        return "none"
    for module, backend in (
        ("uncertainty_engine.client", "sdk"),
        ("uncertainty_engine.models", "library"),
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


def has_sdk() -> bool:
    """Whether the remote-API SDK is available."""
    return detect_backend() == "sdk"


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
        "remote_workflows": backend == "sdk",
    }
