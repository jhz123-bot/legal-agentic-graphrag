from __future__ import annotations

import locale
import sys
from typing import Tuple


def configure_stdio_encoding() -> Tuple[bool, str]:
    configured = False
    encoding = "unknown"
    try:
        encoding = getattr(sys.stdout, "encoding", None) or locale.getpreferredencoding(False) or "unknown"
        for stream in (sys.stdin, sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
                configured = True
        encoding = getattr(sys.stdout, "encoding", None) or encoding
    except Exception:
        configured = False
        encoding = getattr(sys.stdout, "encoding", None) or locale.getpreferredencoding(False) or "unknown"
    return configured, encoding


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoded = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        print(encoded)
