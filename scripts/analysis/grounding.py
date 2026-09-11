from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

def mentions(text: str, identifiers: frozenset[str]) -> set[str]:
    if not identifiers:
        return set()
    return _tokens(text) & identifiers


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


def tool_roots(
    findings: Sequence[Finding], source_root: Path
) -> dict[str, tuple[Path, bool]]:
    """Ustala dla kazdego narzedzia katalog, wzgledem ktorego podaje ono sciezki."""
    candidate_paths = [source_root] + sorted(p for p in source_root.iterdir() if p.is_dir())
    roots: dict[str, tuple[Path, bool]] = {}

    for tool in sorted({f.tool for f in findings}):
        paths = [f.source_path for f in findings if f.tool == tool and f.source_path]
        if not paths:
            roots[tool] = (source_root, False)
            continue
        hits = [sum((k / s).exists() for s in paths) for k in candidate_paths]
        best_item = max(hits)
        if best_item == 0 or hits.count(best_item) > 1:
            roots[tool] = (source_root, False)
            continue
        root = candidate_paths[hits.index(best_item)]
        roots[tool] = (root, root != source_root)
    return roots


def build_grounding(
    finding: Finding, source_root: Path, root: tuple[Path, bool] | None = None
) -> Grounding:
    directory, relative_item = root or (source_root, False)
    context = _build_code_context(
        finding.raw or _as_raw(finding), directory, paths_relative_to_root=relative_item
    )
    if context.startswith("("):
        return Grounding(frozenset(), frozenset(), frozenset(), False, False)

    rows = _strip_line_prefixes(context)
    numbers = [n for n, _ in rows]
    cleaned = _strip_comments("\n".join(code for _, code in rows)).split("\n")
    rows = list(zip(numbers, cleaned))
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


