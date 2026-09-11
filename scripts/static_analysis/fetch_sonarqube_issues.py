#!/usr/bin/env python3
"""Pobiera wszystkie ostrzeżenia z API SonarQube, stronicowaniem."""

import argparse
import json
import urllib.request
import urllib.parse
import base64
import sys

API_RESULT_LIMIT = 10_000
PAGE_SIZE = 500

ISSUE_TYPES = ["BUG", "VULNERABILITY", "CODE_SMELL"]
SEVERITIES = ["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"]

HOTSPOT_PROBABILITY_TO_SEVERITY = {"HIGH": "CRITICAL", "MEDIUM": "MAJOR", "LOW": "MINOR"}


def make_auth_header(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


def get(url: str, auth: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": auth})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def fetch_bucket(base_url: str, auth: str, project_key: str, extra: dict) -> tuple[list[dict], int]:
    page = 1
    all_issues: list[dict] = []
    reported_total = 0

    while True:
        params: dict = {"componentKeys": project_key, "ps": PAGE_SIZE, "p": page}
        params.update(extra)
        data = get(f"{base_url}/api/issues/search?{urllib.parse.urlencode(params)}", auth)
        all_issues.extend(data.get("issues", []))
        reported_total = data.get("total", 0)
        if page * PAGE_SIZE >= min(reported_total, API_RESULT_LIMIT):
            break
        page += 1

    return all_issues, reported_total


def _merge(new_issues: list[dict], target: list[dict], seen: set[str]) -> None:
    for issue in new_issues:
        key = issue.get("key")
        if key not in seen:
            seen.add(key)
            target.append(issue)


def _fetch_version(base_url: str, auth: str) -> str:
    try:
        req = urllib.request.Request(f"{base_url}/api/server/version", headers={"Authorization": auth})
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode().strip()
    except Exception:
        return "unknown"


def _hotspot_to_issue(hotspot: dict) -> dict:
    """Sprowadza hotspot do postaci zwykłego ostrzeżenia."""
    text_range = hotspot.get("textRange") or {}
    line = hotspot.get("line")
    probability = hotspot.get("vulnerabilityProbability", "")
    security_category = hotspot.get("securityCategory")
    return {
        "key": hotspot.get("key"),
        "rule": hotspot.get("ruleKey"),
        "component": hotspot.get("component"),
        "message": hotspot.get("message"),
        "type": "SECURITY_HOTSPOT",
        "severity": HOTSPOT_PROBABILITY_TO_SEVERITY.get(probability),
        "status": hotspot.get("status"),
        "effort": None,
        "tags": [security_category] if security_category else [],
        "textRange": text_range if text_range else ({"startLine": line, "endLine": line} if line else {}),
    }


def fetch_all_hotspots(base_url: str, auth: str, project_key: str) -> tuple[list[dict], int]:
    """Pobiera hotspoty bezpieczeństwa z osobnego punktu końcowego API."""
    page = 1
    all_hotspots: list[dict] = []
    reported_total = 0

    while True:
        params = {"projectKey": project_key, "ps": PAGE_SIZE, "p": page}
        data = get(f"{base_url}/api/hotspots/search?{urllib.parse.urlencode(params)}", auth)
        hotspots = data.get("hotspots", [])
        all_hotspots.extend(hotspots)
        paging = data.get("paging", {})
        reported_total = paging.get("total", 0)
        if page * PAGE_SIZE >= min(reported_total, API_RESULT_LIMIT):
            break
        page += 1

    return [_hotspot_to_issue(h) for h in all_hotspots], reported_total


def fetch_all_issues(base_url: str, user: str, password: str, project_key: str) -> tuple[list[dict], str, int]:
    auth = make_auth_header(user, password)
    issues, reported_total = fetch_bucket(base_url, auth, project_key, {})

    if reported_total <= API_RESULT_LIMIT:
        hotspots, hotspot_total = fetch_all_hotspots(base_url, auth, project_key)
        if hotspot_total > 0:
            print(f"  SECURITY_HOTSPOT: {hotspot_total}", file=sys.stderr)
            seen_keys: set[str] = {i.get("key") for i in issues}
            _merge(hotspots, issues, seen_keys)
        return issues, _fetch_version(base_url, auth), reported_total

    print(
        f"Reported total {reported_total} exceeds API limit {API_RESULT_LIMIT}. "
        "Switching to divide-and-conquer strategy (type × severity).",
        file=sys.stderr,
    )

    seen_keys: set[str] = set()
    all_issues: list[dict] = []

    for issue_type in ISSUE_TYPES:
        _, type_total = fetch_bucket(base_url, auth, project_key, {"types": issue_type, "ps": 1, "p": 1})
        if type_total <= API_RESULT_LIMIT:
            bucket_issues, _ = fetch_bucket(base_url, auth, project_key, {"types": issue_type})
            _merge(bucket_issues, all_issues, seen_keys)
            print(f"  {issue_type}: {type_total}", file=sys.stderr)
        else:
            for severity in SEVERITIES:
                _, sev_total = fetch_bucket(
                    base_url, auth, project_key,
                    {"types": issue_type, "severities": severity, "ps": 1, "p": 1},
                )
                if sev_total > API_RESULT_LIMIT:
                    print(
                        f"  WARNING: {issue_type}/{severity} has {sev_total} warnings, "
                        f"still exceeds {API_RESULT_LIMIT}. Only first {API_RESULT_LIMIT} fetched.",
                        file=sys.stderr,
                    )
                bucket_issues, _ = fetch_bucket(
                    base_url, auth, project_key,
                    {"types": issue_type, "severities": severity},
                )
                _merge(bucket_issues, all_issues, seen_keys)
                print(f"  {issue_type}/{severity}: {sev_total}", file=sys.stderr)

    hotspots, hotspot_total = fetch_all_hotspots(base_url, auth, project_key)
    if hotspot_total > 0:
        print(f"  SECURITY_HOTSPOT: {hotspot_total}", file=sys.stderr)
    _merge(hotspots, all_issues, seen_keys)

    return all_issues, _fetch_version(base_url, auth), reported_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    issues, version, reported_total = fetch_all_issues(args.url, args.user, args.password, args.project)

    output = {
        "sonarqube_version": version,
        "project_key": args.project,
        "total": len(issues),
        "reported_total": reported_total,
        "truncated": len(issues) < reported_total,
        "issues": issues,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Fetched {len(issues)} / {reported_total} issues -> {args.output}")


if __name__ == "__main__":
    main()