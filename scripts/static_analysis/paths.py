"""Wspólne dla konwerterów: układ ścieżek Mavena i klucze z dwukropkiem."""

from __future__ import annotations

import re

JAVA_SOURCE_DIR_RE = re.compile(
    r"^(?:(?P<module>.*?)/)?src/(?P<source_set>main|test)/java/(?P<package_path>.*)/(?P<sourcefile>[^/]+\.java)$"
)


def after_colon(value: str) -> str:
    """Część po pierwszym dwukropku; całość, gdy dwukropka nie ma."""
    return value.split(":", 1)[1] if ":" in value else value
