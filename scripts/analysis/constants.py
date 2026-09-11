"""Nazwy modeli, narzędzi i kategorii — jedno źródło dla całej paczki."""

from __future__ import annotations

MODEL_DIRS = ("claude", "gpt", "qwen", "oss", "luna", "oss20", "sonnet")

# Narzędzia występują w dwóch zapisach: z podkreśleniem w nazwach plików,
# z myślnikiem w polach danych.
TOOL_FILES = ("spotbugs", "error_prone", "sonarqube")
TOOL_ORDER = ("spotbugs", "error-prone", "sonarqube")
TOOL_FILE_TO_NAME = dict(zip(TOOL_FILES, TOOL_ORDER))
TOOL_NAME_TO_FILE = dict(zip(TOOL_ORDER, TOOL_FILES))

CATEGORIES = ("CORRECTNESS", "CONCURRENCY", "SECURITY", "PERFORMANCE", "MAINTAINABILITY", "OTHER")

ALL_TOOLS = "wszystkie"

TRUNCATION_REASONS = {"max_tokens", "length", "MAX_TOKENS"}
