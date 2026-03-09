from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=64)
def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **kwargs: Any) -> str:
    template = load_prompt(name)
    return template.format(**kwargs)

