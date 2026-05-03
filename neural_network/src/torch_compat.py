from __future__ import annotations

import sys
import types


def ensure_torch_dynamo_stub() -> None:
    try:
        import torch._dynamo  # type: ignore  # noqa: F401
        return
    except ModuleNotFoundError:
        sys.modules.pop("torch._dynamo", None)
        pass

    if "torch._dynamo" in sys.modules:
        return

    stub = types.ModuleType("torch._dynamo")

    def _identity_disable(fn=None, recursive: bool = True, wrapping: bool = True):
        if fn is None:
            def decorator(inner):
                return inner
            return decorator
        return fn

    def _identity_optimize(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def _noop(*args, **kwargs):
        return None

    def _is_compiling() -> bool:
        return False

    stub.disable = _identity_disable  # type: ignore[attr-defined]
    stub.optimize = _identity_optimize  # type: ignore[attr-defined]
    stub.graph_break = _noop  # type: ignore[attr-defined]
    stub.is_compiling = _is_compiling  # type: ignore[attr-defined]
    stub.allow_in_graph = _noop  # type: ignore[attr-defined]
    stub.disable_nested_graph_breaks = _noop  # type: ignore[attr-defined]
    stub.disallow_in_graph = _noop  # type: ignore[attr-defined]
    stub.dont_skip_tracing = _noop  # type: ignore[attr-defined]
    stub.error_on_graph_break = _noop  # type: ignore[attr-defined]
    stub.forbid_in_graph = _noop  # type: ignore[attr-defined]
    stub.mark_dynamic = _noop  # type: ignore[attr-defined]
    stub.mark_static = _noop  # type: ignore[attr-defined]
    stub.mark_static_address = _noop  # type: ignore[attr-defined]
    stub.maybe_mark_dynamic = _noop  # type: ignore[attr-defined]
    stub.nonstrict_trace = _noop  # type: ignore[attr-defined]
    stub.run = _noop  # type: ignore[attr-defined]
    stub.set_stance = _noop  # type: ignore[attr-defined]
    stub.skip_frame = _noop  # type: ignore[attr-defined]
    stub.step_unsupported = _noop  # type: ignore[attr-defined]
    stub.substitute_in_graph = _noop  # type: ignore[attr-defined]
    stub.config = types.SimpleNamespace()  # type: ignore[attr-defined]
    sys.modules["torch._dynamo"] = stub
