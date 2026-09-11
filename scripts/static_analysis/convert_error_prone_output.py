#!/usr/bin/env python3

"""Przekształca wyjście Error Prone z kompilacji Mavena do wspólnego formatu."""

import sys
import json
import os
import re
from pathlib import Path
from static_analysis.paths import JAVA_SOURCE_DIR_RE




def normalize_source_path(filepath):
    normalized = filepath.replace('\\', '/')
    target_project_dir = os.environ.get('TARGET_PROJECT_DIR')
    if target_project_dir:
        target_project_dir = str(Path(target_project_dir).resolve()).replace('\\', '/')
        if normalized.startswith(f"{target_project_dir}/"):
            return normalized[len(target_project_dir) + 1:]

    path = Path(filepath)
    if path.is_absolute():
        project_root = infer_project_root(path)
        if project_root:
            try:
                return path.resolve().relative_to(project_root).as_posix()
            except ValueError:
                pass

    return normalized


def infer_project_root(path):
    """Ustala korzeń projektu bez opierania się na nazwie repozytorium."""
    directory = path if path.is_dir() else path.parent
    ancestors = [directory, *directory.parents]

    for ancestor in ancestors:
        if (ancestor / '.git').exists():
            return ancestor.resolve()

    pom_roots = [ancestor for ancestor in ancestors if (ancestor / 'pom.xml').exists()]
    if pom_roots:
        return pom_roots[-1].resolve()

    return None


def parse_path_metadata(filepath):
    """Wyprowadza metadane lokalizacji ze ścieżki źródeł w układzie Mavena."""
    normalized = normalize_source_path(filepath)
    sourcefile = Path(normalized).name
    classname = Path(sourcefile).stem
    metadata = {
        'sourcefile': sourcefile,
        'classname': classname,
        'fully_qualified_classname': classname,
        'source_path': normalized,
        'source_set': None,
        'package': None,
        'module': None,
    }

    match = JAVA_SOURCE_DIR_RE.search(normalized)
    if not match:
        return metadata

    package_path = match.group('package_path')
    package_name = package_path.replace('/', '.') if package_path else None
    metadata.update({
        'source_path': match.group(0),
        'source_set': match.group('source_set'),
        'package': package_name,
        'fully_qualified_classname': f"{package_name}.{classname}" if package_name else classname,
        'module': (match.group('module') or '').strip('/') or None,
    })
    return metadata


def parse_error_prone_output(output_file):
    """Wydobywa ostrzeżenia Error Prone z wyjścia Mavena."""

    warnings = []

    with open(output_file, 'r') as f:
        content = f.read()

    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    content = ansi_escape.sub('', content)
    lines = content.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i]

        if '[WARNING]' in line and '.java:' in line:
            match = re.search(
                r'\[WARNING\]\s+(.+?\.java):\[(\d+),(\d+)\]\s+\[(\w+)\]\s+(.*)',
                line
            )

            if match:
                filepath = match.group(1)
                line_num = int(match.group(2))
                col_num = int(match.group(3))
                check_name = match.group(4)
                message = match.group(5).strip()

                path_metadata = parse_path_metadata(filepath)

                warning = {
                    'type': check_name,
                    'abbrev': check_name[:3].upper(),
                    'sourcefile': path_metadata['sourcefile'],
                    'start_line': str(line_num),
                    'end_line': str(line_num),
                    'classname': path_metadata['classname'],
                    'fully_qualified_classname': path_metadata['fully_qualified_classname'],
                    'source_path': path_metadata['source_path'],
                    'source_set': path_metadata['source_set'],
                    'package': path_metadata['package'],
                    'message': message,
                    'description': f"{check_name}: {message}",
                    'tool_specific': {
                        'column': str(col_num),
                        'module': path_metadata['module'],
                        'filepath': filepath,
                        'bugpattern_url': None,
                        'suggestion': None,
                    }
                }

                j = i + 1
                while j < len(lines):
                    next_line = lines[j]

                    if '[WARNING]' in next_line and '.java:' in next_line:
                        break
                    if next_line.startswith('[INFO]') or next_line.startswith('[ERROR]'):
                        break

                    if 'errorprone.info/bugpattern' in next_line:
                        url_match = re.search(r'(https://errorprone\.info/bugpattern/[\w]+)', next_line)
                        if url_match:
                            warning['tool_specific']['bugpattern_url'] = url_match.group(1)

                    if 'Did you mean' in next_line:
                        suggestion_match = re.search(r"Did you mean '(.+?)'", next_line)
                        if suggestion_match:
                            warning['tool_specific']['suggestion'] = suggestion_match.group(1)

                    j += 1

                warnings.append(warning)

        i += 1

    return warnings

def main():
    if len(sys.argv) < 2:
        print("Usage: convert_error_prone_output.py <output_file>", file=sys.stderr)
        sys.exit(1)

    output_file = sys.argv[1]

    try:
        warnings = parse_error_prone_output(output_file)
    except Exception as e:
        print(f"Error processing the file: {e}", file=sys.stderr)
        sys.exit(1)

    result = {
        'tool': 'error-prone',
        'version': '2.26.1',
        'total_findings': len(warnings),
        'findings': warnings
    }

    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
