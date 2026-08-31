from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.base import _build_code_context

from analysis.dataset import Finding

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MIN_LENGTH = 4
NEAR_RADIUS = 10

@dataclass
class Grounding:
    identifiers: frozenset[str]
    near_identifiers: frozenset[str]
    message_identifiers: frozenset[str]
    shown_full_file: bool
    context_available: bool


def _looks_like_identifier(token: str) -> bool:
    return "_" in token or any(char.isupper() for char in token[1:])


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in IDENTIFIER.findall(text or "")
        if len(token) >= MIN_LENGTH and _looks_like_identifier(token)
    }


COMMENT_OR_LITERAL = re.compile(
    r'"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])*'"
    r"|//[^\n]*"
    r"|/\*.*?\*/",
    re.S,
)


def _strip_comments(code: str) -> str:
    # Trafienia zamieniane na spacje tej samej długości — numeracja linii musi zostać nietknięta.
    return COMMENT_OR_LITERAL.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), code)


def _strip_line_prefixes(context: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for line in context.splitlines():
        match = re.match(r"\s*(\d+)\s\|(?:>>>|\s{3})\s?(.*)$", line)
        if match:
            rows.append((int(match.group(1)), match.group(2)))
    return rows


def build_grounding(finding: Finding, source_root: Path) -> Grounding:
    context = _build_code_context(finding.raw or _as_raw(finding), source_root)
    if context.startswith("("):
        return Grounding(frozenset(), frozenset(), frozenset(), False, False)

    rows = _strip_line_prefixes(context)
    numery = [n for n, _ in rows]
    oczyszczone = _strip_comments("\n".join(code for _, code in rows)).split("\n")
    rows = list(zip(numery, oczyszczone))
    code_tokens: set[str] = set()
    near_tokens: set[str] = set()
    try:
        flagged = int(finding.start_line)
    except (TypeError, ValueError):
        flagged = 0
    for line_number, code in rows:
        tokens = _tokens(code)
        code_tokens |= tokens
        if flagged and abs(line_number - flagged) <= NEAR_RADIUS:
            near_tokens |= tokens

    message_tokens = _tokens(f"{finding.message} {finding.description} {finding.type}")
    return Grounding(
        identifiers=frozenset(code_tokens),
        near_identifiers=frozenset(near_tokens),
        message_identifiers=frozenset(message_tokens),
        shown_full_file="complete file" in context.splitlines()[0],
        context_available=True,
    )


def _as_raw(finding: Finding) -> dict[str, Any]:
    return {
        "sourcefile": finding.sourcefile,
        "start_line": finding.start_line,
        "source_path": finding.source_path,
        "tool_specific": finding.metadata,
    }


def context_kind(finding: Finding, source_root: Path) -> str:
    context = _build_code_context(finding.raw or _as_raw(finding), source_root)
    if context.startswith("("):
        return "no context"
    return "full file" if "complete file" in context.splitlines()[0] else "excerpt"


def mentions(text: str, identifiers: frozenset[str]) -> set[str]:
    if not identifiers:
        return set()
    return _tokens(text) & identifiers


def score_text(text: str, grounding: Grounding) -> dict[str, Any]:
    all_hits = mentions(text, grounding.identifiers)
    beyond_message = all_hits - grounding.message_identifiers
    near_hits = mentions(text, grounding.near_identifiers) - grounding.message_identifiers
    return {
        "grounded": bool(all_hits),
        "grounded_beyond_message": bool(beyond_message),
        "grounded_near_flagged_line": bool(near_hits),
        "hits": len(all_hits),
        "hits_beyond_message": len(beyond_message),
        "identifiers": sorted(beyond_message)[:10],
    }


def reference_precision(text: str, grounding: Grounding) -> dict[str, Any]:
    """Jaka część nazw przywołanych w uzasadnieniu pochodzi z pokazanego fragmentu."""
    if not grounding.context_available:
        return {"available": False}
    candidates = set(_tokens(text))
    candidates -= grounding.message_identifiers
    if not candidates:
        return {"available": True, "referenced": 0, "supported": 0, "unsupported": 0, "precision": None}
    supported = candidates & grounding.identifiers
    unsupported = candidates - supported
    return {
        "available": True,
        "referenced": len(candidates),
        "supported": len(supported),
        "unsupported": len(unsupported),
        "precision": len(supported) / len(candidates),
        "unsupported_examples": sorted(unsupported)[:5],
    }


