"""Config loading with path resolution (mirrors the fusion package style)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]
    config_dir: str
    config_path: str = field(default="")

    def get(self, *keys, default=None):
        node = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def path(self, *keys, default=None) -> str | None:
        """Resolve a (possibly relative) path value from the config tree."""
        p = self.get(*keys, default=default)
        if p is None:
            return None
        if os.path.isabs(p):
            return os.path.normpath(p)
        return os.path.normpath(os.path.join(self.config_dir, p))

    def abspath(self, p: str | None) -> str | None:
        if p is None:
            return None
        if os.path.isabs(p):
            return os.path.normpath(p)
        return os.path.normpath(os.path.join(self.config_dir, p))


def load_config(config_path: str) -> Config:
    config_path = os.path.abspath(config_path)
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, config_dir=os.path.dirname(config_path),
                  config_path=config_path)
