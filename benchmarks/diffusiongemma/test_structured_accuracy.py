import unittest
import structured_accuracy as accuracy


class AccuracyTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            'expected_fields': {'deletes_files': False},
            'tools': [{'type': 'function', 'function': {'name': 'report_analysis',
                'parameters': {'type': 'object', 'additionalProperties': False,
                    'properties': {'deletes_files': {'type': 'boolean'}, 'reason': {'type': 'string'}},
                    'required': ['deletes_files', 'reason']}}}],
        }
        self.call = {'index': 0, 'id': 'test', 'type': 'function',
                     'function': {'name': 'report_analysis',
                                  'arguments': '{"deletes_files":false,"reason":"quoted text"}'}}

    def test_fence_is_reported_separately(self):
        result = accuracy.assess_json(self.case, '```json\n{"deletes_files":false}\n```')
        self.assertEqual(result['format'], 'fenced_json')
        self.assertEqual(result['semantic_status'], 'pass')

    def test_no_coercion_or_brace_salvage(self):
        self.assertEqual(accuracy.assess_json(self.case, '{"deletes_files":"false"}')['semantic_status'], 'fail')
        self.assertEqual(accuracy.assess_json(self.case, 'Answer: {"deletes_files":false}')['semantic_status'], 'unscorable')
        self.assertEqual(accuracy.assess_json(self.case, '{"deletes_files":true,"deletes_files":false}')['semantic_status'], 'unscorable')

    def test_native_tool_arguments(self):
        self.assertEqual(accuracy.check_tool_output(self.case, [self.call])['status'], 'pass')

    def test_missing_wrong_duplicate_and_bad_schema_calls(self):
        self.assertEqual(accuracy.check_tool_output(self.case, [])['status'], 'fail')
        self.assertEqual(accuracy.check_tool_output(self.case, [self.call, self.call])['status'], 'fail')
        self.call['function']['name'] = 'execute_shell'
        self.assertEqual(accuracy.check_tool_output(self.case, [self.call])['status'], 'fail')
        self.call['function']['name'] = 'report_analysis'
        for args in ['{"deletes_files":"false","reason":"x"}',
                     '{"deletes_files":false}', '{"deletes_files":false,"reason":""}',
                     '{"deletes_files":false,"reason":"x","extra":1}',
                     '{"deletes_files":true,"reason":"x"}']:
            self.call['function']['arguments'] = args
            self.assertEqual(accuracy.check_tool_output(self.case, [self.call])['status'], 'fail', args)

    def test_schema_is_required_and_supported(self):
        accuracy.validate_tool_case(self.case)
        self.case['tools'][0]['function']['parameters']['properties']['reason']['type'] = 'array'
        with self.assertRaises(ValueError):
            accuracy.validate_tool_case(self.case)


class CorpusTests(unittest.TestCase):
    def test_native_corpus_has_typed_schemas_and_preserves_expected_values(self):
        import json
        from pathlib import Path
        root = Path(__file__).parent
        original = {c['id']: c for c in map(json.loads, (root / 'cases.jsonl').read_text().splitlines())}
        native = list(map(json.loads, (root / 'tool_cases.jsonl').read_text().splitlines()))
        self.assertEqual(len(native), 8)
        for case in native:
            accuracy.validate_tool_case(case)
            self.assertEqual(case['expected_fields'], original[case['source_case_id']]['expected_fields'])
            self.assertEqual([m['role'] for m in case['messages']], ['system', 'user'])
            self.assertNotIn('Return only JSON', str(case['messages']))


class NativeResultTests(unittest.TestCase):
    def test_tool_only_response_is_not_empty_or_missing_content(self):
        import benchmark as bench
        import json
        from pathlib import Path
        case = json.loads((Path(__file__).parent / 'tool_cases.jsonl').read_text().splitlines()[0])
        scheduled = bench.schedule_cases([case], 1, 1, 42)[1]
        call = {'index': 0, 'id': 'test', 'type': 'function',
                'function': {'name': 'report_analysis', 'arguments': json.dumps(
                    dict(case['expected_fields'], reason='quoted string'))}}
        row = dict(scheduled, type='request', output='', tool_calls=[call], response_s=3.,
                   first_content_s=2., finish_reason='tool_calls', prompt_tokens=10,
                   completion_tokens=20, canvases=[dict(passes=2, finalized_at_s=1.9,
                   observed_step_intervals_s=[.8], converged=True)])
        valid = bench.validate_result(row, scheduled)
        self.assertEqual(valid['check']['status'], 'pass')
        self.assertEqual(bench.summarize([valid])['cases'][case['id']]['empty_outputs'], 0)


if __name__ == '__main__':
    unittest.main()
