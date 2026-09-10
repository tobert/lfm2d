"""Separate wire-format, schema and expected-field checks; never execute model output."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate key: {key}')
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f'non-JSON constant: {value}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def field_mismatches(expected, actual):
    if not isinstance(actual, dict):
        return {'$': {'expected': 'object', 'actual': actual}}
    return {key: {'expected': value, 'actual': actual.get(key), 'present': key in actual}
            for key, value in expected.items()
            if key not in actual or type(actual[key]) is not type(value) or actual[key] != value}


def assess_json(case, output):
    expected = case.get('expected_fields')
    if expected is None:
        return dict(format='ungraded', semantic_status='ungraded')
    text = output.strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    form = 'fenced_json' if fence else 'raw_json'
    try:
        actual = strict_json(fence[1] if fence else text)
    except ValueError:
        return dict(format='empty' if not text else 'unparseable', semantic_status='unscorable')
    mismatches = field_mismatches(expected, actual)
    return dict(format=form, semantic_status='fail' if mismatches else 'pass',
                mismatches=mismatches, actual=actual)


def validate_tool_case(case):
    tools = case.get('tools')
    if not isinstance(tools, list) or len(tools) != 1:
        raise ValueError('typed cases require exactly one report_analysis tool')
    tool = tools[0]
    function = tool.get('function', {})
    schema = function.get('parameters', {})
    properties = schema.get('properties', {})
    if (tool.get('type') != 'function' or function.get('name') != 'report_analysis'
            or function.get('strict', False) is not False
            or schema.get('type') != 'object' or schema.get('additionalProperties') is not False
            or not isinstance(properties, dict) or not properties
            or set(schema.get('required', [])) != set(properties)):
        raise ValueError('unsupported report_analysis declaration')
    for spec in properties.values():
        if spec.get('type') not in ('string', 'boolean'):
            raise ValueError('this scorer supports flat required string/boolean fields only')
        if 'enum' in spec and (not isinstance(spec['enum'], list) or not spec['enum']
                              or not all(isinstance(x, str) for x in spec['enum'])):
            raise ValueError('invalid enum')
    if set(case.get('expected_fields', {})) - set(properties):
        raise ValueError('expected fields missing from tool schema')
    return schema


def check_tool_output(case, calls):
    schema = validate_tool_case(case)
    result = dict(status='fail', scope='native report_analysis arguments',
                  schema_status='fail', semantic_status='unscorable')
    if not isinstance(calls, list) or len(calls) != 1:
        return dict(result, reason='expected exactly one tool call')
    call = calls[0]
    fn = call.get('function', {})
    if call.get('type') != 'function' or fn.get('name') != 'report_analysis':
        return dict(result, reason='wrong tool')
    try:
        actual = strict_json(fn['arguments'])
    except (KeyError, TypeError, ValueError):
        return dict(result, reason='invalid argument JSON')
    if not isinstance(actual, dict) or set(actual) != set(schema['properties']):
        return dict(result, reason='missing or additional arguments', actual=actual)
    for name, spec in schema['properties'].items():
        value = actual[name]
        kind = str if spec['type'] == 'string' else bool
        if type(value) is not kind or (kind is str and not value.strip()):
            return dict(result, reason=f'invalid type or empty value: {name}', actual=actual)
        if 'enum' in spec and value not in spec['enum']:
            return dict(result, reason=f'invalid enum value: {name}', actual=actual)
    mismatches = field_mismatches(case.get('expected_fields', {}), actual)
    return dict(result, status='fail' if mismatches else 'pass', schema_status='pass',
                semantic_status='fail' if mismatches else 'pass', mismatches=mismatches, actual=actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = []
    for run in args.runs:
        records = []
        for line in (run / 'raw.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row.get('phase') != 'measure':
                continue
            check = (check_tool_output(row['case'], row.get('tool_calls', []))
                     if row['case'].get('tools') else assess_json(row['case'], row['output']))
            records.append(dict(case_id=row['case_id'], repeat=row['repeat'], **check))
        report.append(dict(run=str(run), counts=dict(Counter(x['semantic_status'] for x in records)),
                           records=records))
    with args.output.open('x') as out:
        json.dump(report, out, indent=2)
    for run in report:
        print(run['run'], run['counts'])


if __name__ == '__main__':
    main()
