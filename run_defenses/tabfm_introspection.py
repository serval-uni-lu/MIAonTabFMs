"""Small runtime introspection helpers for TabFM wrappers."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any


def _as_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except Exception:
        return None


def infer_num_thinking_rows(model: Any, default: int | None = 0) -> int | None:
    """Best-effort lookup of a loaded TabPFN model's thinking-token count.

    TabPFN exposes the value either in the architecture config or on the
    ``add_thinking_tokens`` module.  The estimator can be wrapped by local
    defenses, so this walks the common wrapper/model attributes and ensemble
    containers without touching arbitrary attributes.
    """
    seen: set[int] = set()
    queue: deque[Any] = deque([model])
    attr_links = (
        "model",
        "model_",
        "models_",
        "base_model",
        "estimator",
        "estimator_",
        "module",
        "net",
        "network",
        "architecture",
        "model_type",
        "executor",
    )
    config_links = (
        "config",
        "config_",
        "model_config",
        "model_config_",
        "architecture_config",
        "architecture_config_",
    )

    while queue:
        obj = queue.popleft()
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        value = _as_int(getattr(obj, "num_thinking_rows", None))
        if value is not None:
            return max(0, value)

        add_tokens = getattr(obj, "add_thinking_tokens", None)
        value = _as_int(getattr(add_tokens, "num_thinking_rows", None))
        if value is not None:
            return max(0, value)
        if add_tokens is not None:
            queue.append(add_tokens)

        for attr in config_links:
            cfg = getattr(obj, attr, None)
            value = _as_int(getattr(cfg, "num_thinking_rows", None))
            if value is not None:
                return max(0, value)
            if isinstance(cfg, dict):
                value = _as_int(cfg.get("num_thinking_rows"))
                if value is not None:
                    return max(0, value)

        for attr in attr_links:
            child = getattr(obj, attr, None)
            if child is None:
                continue
            if isinstance(child, dict):
                queue.extend(child.values())
            elif isinstance(child, Iterable) and not isinstance(child, (str, bytes)):
                queue.extend(child)
            else:
                queue.append(child)

    if default is None:
        return None
    return max(0, int(default))
