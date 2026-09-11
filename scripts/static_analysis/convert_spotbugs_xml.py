#!/usr/bin/env python3
import re
import html
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def simple_classname(sourcefile, fully_qualified_classname):
    if sourcefile and sourcefile.endswith('.java'):
        return Path(sourcefile).stem
    if fully_qualified_classname:
        return fully_qualified_classname.rsplit('.', 1)[-1]
    return None


def package_from_classname(fully_qualified_classname):
    if fully_qualified_classname and '.' in fully_qualified_classname:
        return fully_qualified_classname.rsplit('.', 1)[0]
    return None


def source_path_from_spotbugs(module, sourcepath):
    if not sourcepath:
        return None
    source_root = 'src/main/java'
    if module:
        return f"{module}/{source_root}/{sourcepath}"
    return f"{source_root}/{sourcepath}"


def strip_html(text):
    if not text:
        return text
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    return ' '.join(text.split())


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    version = root.get('version', '4.7.3')
    module = Path(xml_file).name
    if module.endswith('__target__spotbugsXml.xml'):
        module = module[:-len('__target__spotbugsXml.xml')]
    elif module == 'spotbugsXml.xml':
        module = ''
    module = module.replace('__', '/') if module else ''

    bug_pattern_details = {}
    for bp in root.findall('BugPattern'):
        details = bp.find('Details')
        if details is not None and details.text:
            bug_pattern_details[bp.get('type')] = strip_html(details.text)

    bugs = []

    for bug_instance in root.findall('.//BugInstance'):
        bug = {
            'type': bug_instance.get('type'),
            'abbrev': bug_instance.get('abbrev'),
            'sourcefile': None,
            'start_line': None,
            'end_line': None,
            'classname': None,
            'fully_qualified_classname': None,
            'source_path': None,
            'source_set': None,
            'package': None,
            'message': None,
            'description': None,
        }

        source_lines = bug_instance.findall('SourceLine')
        source_line = next((sl for sl in source_lines if sl.get('primary') == 'true'), None)
        if source_line is None:
            source_line = source_lines[0] if source_lines else None
        if source_line is not None:
            sourcefile = source_line.get('sourcefile')
            sourcepath = source_line.get('sourcepath')
            fully_qualified_classname = source_line.get('classname')
            package_name = package_from_classname(fully_qualified_classname)

            bug['sourcefile'] = sourcefile
            bug['start_line'] = source_line.get('start')
            bug['end_line'] = source_line.get('end')
            bug['classname'] = simple_classname(sourcefile, fully_qualified_classname)
            bug['fully_qualified_classname'] = fully_qualified_classname or bug['classname']
            bug['source_path'] = source_path_from_spotbugs(module, sourcepath)
            bug['source_set'] = 'main' if sourcepath else None
            bug['package'] = package_name

        short_message = bug_instance.find('ShortMessage')
        if short_message is not None:
            bug['message'] = html.unescape(short_message.text) if short_message.text else short_message.text

        long_message = bug_instance.find('LongMessage')
        if long_message is not None:
            bug['description'] = html.unescape(long_message.text) if long_message.text else long_message.text

        tool_specific = {
            'category': bug_instance.get('category'),
            'priority': bug_instance.get('priority'),
            'instance_hash': bug_instance.get('instanceHash'),
            'rank': bug_instance.get('rank'),
            'cweid': bug_instance.get('cweid'),
            'module': module or None,
            'method': None,
            'details': None,
            'class_start_line': None,
            'class_end_line': None,
        }

        class_source_line = bug_instance.find('Class/SourceLine')
        if class_source_line is not None:
            tool_specific['class_start_line'] = class_source_line.get('start')
            tool_specific['class_end_line'] = class_source_line.get('end')

        method = bug_instance.find('Method[@primary="true"]')
        if method is None:
            method = bug_instance.find('Method')
        if method is not None:
            tool_specific['method'] = method.get('name')

        bug_type = bug_instance.get('type')
        if bug_type in bug_pattern_details:
            tool_specific['details'] = bug_pattern_details[bug_type]

        bug['tool_specific'] = tool_specific

        bugs.append(bug)

    return version, bugs


def main():
    xml_files = sys.argv[1:]
    if not xml_files:
        print(f"Usage: {sys.argv[0]} <spotbugs.xml> [<spotbugs.xml> ...]", file=sys.stderr)
        sys.exit(1)

    try:
        versions = []
        bugs = []
        for xml_file in xml_files:
            version, parsed_bugs = parse_xml(xml_file)
            versions.append(version)
            bugs.extend(parsed_bugs)

        version = versions[0] if versions else 'unknown'
        if len(set(versions)) > 1:
            version = ','.join(sorted(set(versions)))

        result = {
            'tool': 'spotbugs',
            'version': version,
            'total_findings': len(bugs),
            'findings': bugs
        }

        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
