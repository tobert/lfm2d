import io
import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark as bench


class BenchmarkTests(unittest.TestCase):
    def test_hardware_inventory_records_q8_kernel_override(self):
        with patch.dict('os.environ', {'MISTRALRS_ROCM_MOE_Q8_KERNEL': 'wmma'}):
            self.assertEqual(bench.hardware_inventory()['overrides']['MISTRALRS_ROCM_MOE_Q8_KERNEL'],
                             'wmma')

    def setUp(self):
        self.case = {'id': 'a', 'category': 'control',
                     'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 100}
        self.schedule = bench.schedule_cases([self.case], 1, 2, 42)
        self.row = dict(self.schedule[1], type='request', output='hello', response_s=3.,
                        first_content_s=2., finish_reason='stop', prompt_tokens=10,
                        completion_tokens=1, canvases=[dict(passes=2, finalized_at_s=1.9,
                                                          observed_step_intervals_s=[.8], converged=True)])

    def test_partial_jsonl_does_not_get_consumed(self):
        source = io.StringIO('{"a":1}')
        self.assertIsNone(bench.read_record(source))
        self.assertEqual(source.tell(), 0)
        self.assertEqual(bench.read_record(io.StringIO('{"a":1}\n')), {'a': 1})

    def test_schedule_preserves_each_case_and_is_repeatable(self):
        cases = [self.case, {**self.case, 'id': 'b'}]
        schedule = bench.schedule_cases(cases, 3, 5, 42)
        self.assertEqual(schedule, bench.schedule_cases(cases, 3, 5, 42))
        self.assertEqual(len(schedule), 13)
        for repeat in range(5):
            self.assertEqual({r['case_id'] for r in schedule if r['phase'] == 'measure'
                              and r['repeat'] == repeat}, {'a', 'b'})

    def test_native_result_validates_order_timing_usage_and_progress(self):
        self.assertEqual(bench.validate_result(self.row, self.schedule[1])['check']['status'], 'ungraded')
        for overrides in [dict(sequence=99), dict(response_s=float('nan')),
                          dict(first_content_s=9), dict(finish_reason=None),
                          dict(completion_tokens=0), dict(canvases=[]),
                          dict(canvases=[dict(passes=3, finalized_at_s=1.9,
                                              observed_step_intervals_s=[.8])])]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                bench.validate_result({**self.row, **overrides}, self.schedule[1])

    def test_summary_excludes_warmup_and_keeps_cases_separate(self):
        rows = [dict(self.row, phase=phase, case_id=case, response_s=seconds,
                     first_content_s=seconds, check={'status': 'ungraded'})
                for phase, case, seconds in [('warmup', 'a', 99), ('measure', 'a', 1),
                                             ('measure', 'a', 3), ('measure', 'b', 5)]]
        result = bench.summarize(rows)
        self.assertEqual(result['requests'], 3)
        self.assertEqual(result['cases']['a']['response_s']['p50'], 2)
        self.assertEqual(result['cases']['a']['response_s']['p95'], 3)
        self.assertEqual(result['cases']['b']['response_s']['p50'], 5)

    def test_output_checks_do_not_grade_unconfigured_prose(self):
        self.assertEqual(bench.check_output({}, 'great answer')['status'], 'ungraded')
        case = {'expected_fields': {'severity': 'situation-normal'}}
        self.assertEqual(bench.check_output(case, '{"severity":"data-critical"}')['status'], 'fail')
        self.assertEqual(bench.check_output(case, '{"severity":"situation-normal"}')['status'], 'pass')
        self.assertEqual(bench.check_output(case, 'not json')['status'], 'fail')

    def test_summary_counts_unconverged_canvases_and_requests(self):
        canvas = self.row['canvases'][0]
        row = dict(self.row, check={'status': 'ungraded'},
                   canvases=[canvas, dict(canvas, converged=False), dict(canvas, converged=False)])
        result = bench.summarize([row, dict(row, phase='warmup')])['cases']['a']
        self.assertEqual(result['unconverged_canvases'], 2)
        self.assertEqual(result['requests_with_unconverged_canvas'], 1)

    def test_throughput_reports_engine_tokens_including_failed_output(self):
        row = dict(self.row, completion_tokens=3, output='', first_content_s=None,
                   check={'status': 'fail'})
        result = bench.summarize([row])
        self.assertEqual(result['reported_completion_tokens_per_s'], 1.)
        self.assertNotIn('useful_output_tokens_per_s', result)

    def test_canvas_convergence_must_be_explicit_boolean(self):
        for value in (None, 0, 'false'):
            row = dict(self.row, canvases=[dict(self.row['canvases'][0], converged=value)])
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'convergence'):
                bench.validate_result(row, self.schedule[1])

    def test_boolean_output_checks_reject_numeric_lookalikes(self):
        self.assertEqual(bench.check_output({'expected_fields': {'safe': True}}, '{"safe":1}')['status'], 'fail')

    def test_null_requires_field_presence(self):
        case = {'expected_fields': {'note': None}}
        self.assertEqual(bench.check_output(case, '{}')['status'], 'fail')
        self.assertEqual(bench.check_output(case, '{"note":null}')['status'], 'pass')

    def test_empty_answer_is_recorded_as_quality_failure_not_protocol_failure(self):
        row = {**self.row, 'output': '', 'first_content_s': None, 'completion_tokens': 0}
        result = bench.validate_result(row, self.schedule[1])
        self.assertEqual(result['check']['status'], 'fail')
        self.assertEqual(bench.summarize([result])['cases']['a']['empty_outputs'], 1)

    def test_packaged_cases_have_expected_categories(self):
        cases = bench.validate_cases([json.loads(line) for line in
                                     Path(__file__).with_name('cases.jsonl').read_text().splitlines()])
        self.assertEqual(len(cases), 12)
        self.assertIn('adjudicator_candidate', {case['category'] for case in cases})

    def test_case_validation_rejects_duplicate_ids_and_bad_limits(self):
        self.assertEqual(bench.validate_cases([self.case]), [self.case])
        for cases in [[self.case, self.case], [{**self.case, 'max_tokens': 0}], []]:
            with self.assertRaises(ValueError):
                bench.validate_cases(cases)

    def run_fake_driver(self, tail, isq='q4k', ready_isq=None, thinking='off', ready_thinking=None,
                        max_tokens=None, prefix_cache_n=0, ready_prefix_cache_n=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            model.mkdir()
            for name in ('config.json', 'generation_config.json', 'tokenizer_config.json'):
                (model / name).write_text('{}')
            cases = root / 'cases.jsonl'
            cases.write_text(json.dumps(self.case) + '\n')
            args = SimpleNamespace(cases=cases, only=None, binary=Path(__file__), model=model,
                                   output=root / 'result', label='test', warmup=1, repetitions=1,
                                   order_seed=42, seed=42, request_timeout=1, startup_timeout=1,
                                   isq=isq, thinking=thinking, max_tokens=max_tokens,
                                   prefix_cache_n=prefix_cache_n)

            def launch(command, **kwargs):
                self.assertNotIn('serve', command)
                self.assertIn('--isq', command)
                self.assertEqual(command[command.index('--isq') + 1], isq)
                schedule = [json.loads(line) for line in (args.output / 'requests.jsonl').read_text().splitlines()]
                self.assertEqual(command[command.index('--thinking') + 1], str(thinking == 'on').lower())
                self.assertEqual(command[command.index('--prefix-cache-n') + 1], str(prefix_cache_n))
                self.assertTrue(all(item['case']['max_tokens'] == (max_tokens or self.case['max_tokens']) for item in schedule))
                records = [dict(type='ready', startup_s=1, isq=ready_isq or isq,
                                thinking=(thinking == 'on') if ready_thinking is None else ready_thinking,
                                prefix_cache_n=(prefix_cache_n if ready_prefix_cache_n is None
                                                else ready_prefix_cache_n))]
                records.extend({**self.row, **item} for item in schedule)
                text = ''.join(json.dumps(row) + '\n' for row in records) + tail
                (args.output / 'raw.jsonl').write_text(text)
                return SimpleNamespace(poll=lambda: 0, returncode=0)

            with patch.object(bench.subprocess, 'Popen', side_effect=launch), \
                    patch.object(bench.platform, 'platform', return_value='test'), \
                    contextlib.redirect_stdout(io.StringIO()):
                try:
                    bench.run(args)
                finally:
                    self.assertTrue((args.output / 'results.jsonl').exists())
            summary = json.loads((args.output / 'summary.json').read_text())
            self.assertEqual(summary['requests'], 1)
            self.assertEqual(summary['cases']['a']['samples'], 1)
            metadata = json.loads((args.output / 'results.jsonl').read_text().splitlines()[0])
            self.assertEqual(metadata['isq'], isq)
            self.assertEqual(metadata['thinking'], thinking == 'on')
            self.assertEqual(metadata['max_tokens_override'], max_tokens)
            self.assertEqual(metadata['prefix_cache_n'], prefix_cache_n)

    def test_thinking_and_token_override_reach_driver(self):
        self.run_fake_driver('{"type":"complete"}\n', thinking='on', max_tokens=1024)

    def test_thinking_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'thinking'):
            self.run_fake_driver('{"type":"complete"}\n', thinking='on', ready_thinking=False)

    def test_prefix_cache_reaches_driver_and_provenance(self):
        self.run_fake_driver('{"type":"complete"}\n', prefix_cache_n=16)

    def test_prefix_cache_mismatch_fails(self):
        # Without this gate a run could be labelled "cache on" while the driver
        # ran with it off -- and the whole point of the run is the difference
        # between those two, so the label IS the measurement.
        with self.assertRaisesRegex(ValueError, 'prefix cache'):
            self.run_fake_driver('{"type":"complete"}\n', prefix_cache_n=16,
                                 ready_prefix_cache_n=0)

    def test_q8_selection_reaches_driver_and_provenance(self):
        self.run_fake_driver('{"type":"complete"}\n', isq='q8_0')

    def test_driver_quantization_must_match_requested_quantization(self):
        with self.assertRaisesRegex(ValueError, 'quantization'):
            self.run_fake_driver('{"type":"complete"}\n', isq='q8_0', ready_isq='q4k')

    def test_already_exited_driver_results_are_fully_collected(self):
        self.run_fake_driver('{"type":"complete"}\n')

    def test_missing_complete_record_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'without clean completion'):
            self.run_fake_driver('')

    def test_partial_final_record_fails(self):
        with self.assertRaisesRegex(ValueError, 'partial driver output'):
            self.run_fake_driver('{"type":"complete"}')


if __name__ == '__main__':
    unittest.main()
