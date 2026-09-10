"""Run and summarize the resident mistral.rs diffusion_bench library driver."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

from structured_accuracy import check_tool_output, validate_tool_case


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def hardware_inventory():
    version = Path('/opt/rocm/.info/version')
    bundle_dir = os.environ.get('MISTRALRS_ROCM_CANVAS_KERNEL_DIR')
    manifest = Path(bundle_dir) / 'manifest.json' if bundle_dir else None
    cards = []
    for device in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
        fields = {}
        for name in ('vendor', 'device', 'product_name', 'mem_info_vram_total'):
            path = device / name
            if path.is_file():
                fields[name] = path.read_text().strip()
        cards.append(dict(path=str(device.resolve()), **fields))
    return dict(rocm_version=version.read_text().strip() if version.is_file() else None,
                canvas_kernel_manifest_sha256=sha256(manifest) if manifest and manifest.is_file() else None,
                drm_inventory=cards, selected_device='ROCm ordinal 0',
                overrides={name: os.environ[name] for name in
                           ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HSA_OVERRIDE_GFX_VERSION',
                            'MISTRALRS_ROCM_MOE_Q8_KERNEL', 'MISTRALRS_ROCM_CANVAS_ATTN',
                            'MISTRALRS_ROCM_CANVAS_KERNEL_DIR')
                           if name in os.environ})


def validate_cases(cases):
    seen = set()
    if not cases:
        raise ValueError('empty case set')
    for case in cases:
        if not isinstance(case.get('id'), str) or not case['id'] or case['id'] in seen:
            raise ValueError('missing or duplicate case id')
        seen.add(case['id'])
        if not isinstance(case.get('category'), str):
            raise ValueError('case needs a category')
        if type(case.get('max_tokens')) is not int or case['max_tokens'] <= 0:
            raise ValueError('max_tokens must be a positive integer')
        if not case.get('messages') or any(
            m.get('role') not in ('system', 'user', 'assistant')
            or not isinstance(m.get('content'), str) for m in case['messages']
        ):
            raise ValueError('case needs text chat messages')
        if 'expected_fields' in case and not isinstance(case['expected_fields'], dict):
            raise ValueError('expected_fields must be an object')
        if 'tools' in case:
            validate_tool_case(case)
    return cases


def check_output(case, output):
    if not output.strip():
        return dict(status='fail', scope='nonempty answer', reason='empty output')
    expected = case.get('expected_fields')
    if expected is None:
        return dict(status='ungraded', scope='human review required')
    try:
        actual = json.loads(output)
    except json.JSONDecodeError:
        return dict(status='fail', scope='JSON fields only', reason='invalid JSON')
    passed = isinstance(actual, dict) and all(
        k in actual and type(actual[k]) is type(v) and actual[k] == v for k, v in expected.items())
    return dict(status='pass' if passed else 'fail', scope='JSON fields only', expected=expected)


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return dict(p50=statistics.median(values), p95=values[math.ceil(len(values) * .95) - 1],
                minimum=values[0], maximum=values[-1])


def summarize(rows):
    measured = [r for r in rows if r.get('type') == 'request' and r['phase'] == 'measure']
    if not measured:
        raise ValueError('no measured requests')
    cases = {}
    for case_id in sorted({r['case_id'] for r in measured}):
        group = [r for r in measured if r['case_id'] == case_id]
        cases[case_id] = dict(
            samples=len(group),
            response_s=distribution([r['response_s'] for r in group]),
            first_content_s=distribution([r['first_content_s'] for r in group if r['first_content_s'] is not None]),
            empty_outputs=sum(not r['output'].strip() and not r.get('tool_calls') for r in group),
            passes=distribution([sum(c['passes'] for c in r['canvases']) for r in group]),
            canvas_count=distribution([len(r['canvases']) for r in group]),
            unconverged_canvases=sum(not c['converged'] for r in group for c in r['canvases']),
            requests_with_unconverged_canvas=sum(any(not c['converged'] for c in r['canvases'])
                                                for r in group),
            completion_tokens=distribution([r['completion_tokens'] for r in group]),
            truncated=sum(r['finish_reason'] == 'length' for r in group),
            checks={status: sum(r['check']['status'] == status for r in group)
                    for status in ('pass', 'fail', 'ungraded')},
        )
    return dict(requests=len(measured), cases=cases,
                response_s=distribution([r['response_s'] for r in measured]),
                reported_completion_tokens_per_s=sum(r['completion_tokens'] for r in measured)
                / sum(r['response_s'] for r in measured),
                percentile_method='p50 median; p95 nearest rank (small samples are descriptive)',
                timing='native host receive times; inter-progress intervals omit first pass and are not GPU kernel time')


def schedule_cases(cases, warmup, repetitions, seed):
    schedule = []

    def add(phase, repeat, case):
        schedule.append(dict(phase=phase, repeat=repeat, case_id=case['id'],
                             sequence=len(schedule), case=case))
    for i in range(warmup):
        add('warmup', i, cases[i % len(cases)])
    rng = random.Random(seed)
    for repeat in range(repetitions):
        order = list(cases)
        rng.shuffle(order)
        for case in order:
            add('measure', repeat, case)
    return schedule


def validate_result(row, expected):
    for key in ('phase', 'repeat', 'sequence', 'case_id', 'case'):
        if row.get(key) != expected[key]:
            raise ValueError(f'driver result does not match schedule: {key}')
    for key in ('first_content_s', 'response_s'):
        value = row.get(key)
        if key == 'first_content_s' and value is None and row.get('output') == '':
            continue
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'invalid timing: {key}')
    if row['first_content_s'] is not None and row['first_content_s'] > row['response_s']:
        raise ValueError('first content is after response end')
    finish_reasons = ('stop', 'length', 'tool_calls') if expected['case'].get('tools') else ('stop', 'length')
    if not isinstance(row.get('output'), str) or row.get('finish_reason') not in finish_reasons:
        raise ValueError('missing output or invalid finish reason')
    for key in ('prompt_tokens', 'completion_tokens'):
        minimum = 1 if key == 'prompt_tokens' or row['output'] else 0
        if type(row.get(key)) is not int or row[key] < minimum:
            raise ValueError(f'invalid usage: {key}')
    if not row.get('canvases'):
        raise ValueError('missing canvas metrics')
    previous = 0
    for canvas in row['canvases']:
        if type(canvas.get('converged')) is not bool:
            raise ValueError('missing or invalid canvas convergence')
        passes = canvas.get('passes')
        at = canvas.get('finalized_at_s', float('nan'))
        intervals = canvas.get('observed_step_intervals_s', [])
        if type(passes) is not int or passes <= 0 or len(intervals) != passes - 1:
            raise ValueError('invalid canvas pass count or missing progress')
        if not math.isfinite(at) or not previous <= at <= row['response_s']:
            raise ValueError('invalid canvas timestamp')
        if any(not math.isfinite(x) or x < 0 for x in intervals):
            raise ValueError('invalid denoise progress interval')
        previous = at
    check = (check_tool_output(expected['case'], row.get('tool_calls', []))
             if expected['case'].get('tools') else check_output(expected['case'], row['output']))
    return {**row, 'category': expected['case']['category'], 'check': check}


def json_line(sink, value):
    sink.write(json.dumps(value, allow_nan=False) + '\n')
    sink.flush()


def read_record(source):
    offset = source.tell()
    line = source.readline()
    if not line.endswith('\n'):
        source.seek(offset)
        return None
    return json.loads(line)


def run(args):
    cases = validate_cases([json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()])
    if args.only:
        wanted = set(args.only.split(','))
        if wanted - {c['id'] for c in cases}:
            raise ValueError('unknown --only case id')
        cases = [c for c in cases if c['id'] in wanted]
    if args.max_tokens is not None:
        if type(args.max_tokens) is not int or args.max_tokens <= 0:
            raise ValueError('max_tokens override must be positive')
        cases = [{**case, 'max_tokens': args.max_tokens} for case in cases]
    binary = args.binary.resolve(strict=True)
    model = args.model.resolve(strict=True)
    config_names = ('config.json', 'generation_config.json', 'tokenizer_config.json')
    config_hashes = {name: sha256(model / name) for name in config_names}
    forbidden = [k for k in os.environ if k.startswith(('MISTRALRS_DIFFUSION_', 'ROCPROF', 'ROCP_'))]
    if forbidden:
        raise ValueError(f'run without debug-dump/profiler environment variables: {forbidden}')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    schedule = schedule_cases(cases, args.warmup, args.repetitions, args.order_seed)
    with (args.output / 'requests.jsonl').open('x') as sink:
        for item in schedule:
            json_line(sink, item)
    command = [str(binary), '--model', str(model), '--requests', str(args.output / 'requests.jsonl'),
               '--results', str(args.output / 'raw.jsonl'), '--seed', str(args.seed),
               '--request-timeout', str(args.request_timeout), '--isq', args.isq,
               '--thinking', str(args.thinking == 'on').lower(),
               '--prefix-cache-n', str(args.prefix_cache_n)]
    # RUST_LOG defaults to info but is NOT forced: a diagnostic run needs to be
    # able to turn a module up (e.g. mistralrs_core::prefix_cacher=debug) without
    # editing the harness. driver.log is not parsed for measurements, so a
    # louder log cannot change a number.
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'NO_COLOR': '1',
           'RUST_LOG': os.environ.get('RUST_LOG', 'info')}
    metadata = dict(type='metadata', schema_version=3, label=args.label,
                    binary=str(binary), binary_sha256=sha256(binary), model=str(model),
                    model_config_sha256=config_hashes, harness_sha256=sha256(__file__),
                    cases_sha256=sha256(args.cases), cases=cases,
                    process_seed=args.seed, order_seed=args.order_seed,
                    rng_policy='device RNG seeded at startup only; requests are not paired random draws across builds',
                    schedule=schedule, command=command, platform=platform.platform(),
                    hardware=hardware_inventory(),
                    isq=args.isq, thinking=args.thinking == 'on', max_tokens_override=args.max_tokens,
                    prefix_cache_n=args.prefix_cache_n, transport='native library',
                    started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    rows = []
    with (args.output / 'results.jsonl').open('x') as sink, (args.output / 'driver.log').open('xb') as log:
        json_line(sink, metadata)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        ready = complete = False
        raw = None
        last_record = time.monotonic()
        try:
            while True:
                exitcode = process.poll()
                if raw is None and (args.output / 'raw.jsonl').exists():
                    raw = (args.output / 'raw.jsonl').open()
                record = read_record(raw) if raw else None
                if record is not None:
                    last_record = time.monotonic()
                    kind = record.get('type')
                    if kind == 'ready' and not ready and not rows:
                        if record.get('isq') != args.isq:
                            raise ValueError('driver quantization does not match requested quantization')
                        if type(record.get('thinking')) is not bool or record['thinking'] != (args.thinking == 'on'):
                            raise ValueError('driver thinking does not match requested thinking')
                        if record.get('prefix_cache_n') != args.prefix_cache_n:
                            raise ValueError('driver prefix cache does not match requested prefix cache')
                        ready = True
                        json_line(sink, record)
                        print(f'Ready: {args.label}; {len(schedule)} requests including {args.warmup} warmups', flush=True)
                    elif kind == 'request' and ready and not complete and len(rows) < len(schedule):
                        row = validate_result(record, schedule[len(rows)])
                        rows.append(row)
                        json_line(sink, row)
                        print(json.dumps({k: row[k] for k in ('phase', 'case_id', 'response_s', 'completion_tokens', 'check')}), flush=True)
                    elif kind == 'complete' and len(rows) == len(schedule) and not complete:
                        complete = True
                        json_line(sink, record)
                    elif kind == 'error':
                        raise RuntimeError(f'driver failed case {record.get("case_id")}: {record.get("error")}')
                    else:
                        raise ValueError(f'unexpected driver record: {record}')
                    continue
                if exitcode is not None:
                    if raw and raw.tell() < (args.output / 'raw.jsonl').stat().st_size:
                        raise ValueError('partial driver output at exit')
                    if process.returncode != 0 or not complete:
                        raise RuntimeError(f'driver exited {process.returncode} without clean completion; inspect driver.log')
                    break
                timeout = args.request_timeout + 30 if ready else args.startup_timeout
                if time.monotonic() - last_record > timeout:
                    raise TimeoutError('driver made no progress within its deadline')
                time.sleep(.1)
            with (args.output / 'summary.json').open('x') as target:
                json.dump(summarize(rows), target, indent=2, allow_nan=False)
        except BaseException as error:
            json_line(sink, dict(type='error', error=repr(error), completed_requests=len(rows)))
            raise
        finally:
            if raw:
                raw.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            json_line(sink, dict(type='shutdown', returncode=process.returncode))


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True, help='native diffusion_bench example binary')
    parser.add_argument('--model', type=Path, required=True, help='local snapshot directory')
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('cases.jsonl'))
    parser.add_argument('--output', type=Path, required=True, help='new directory; never overwrites')
    parser.add_argument('--label', required=True)
    parser.add_argument('--isq', choices=('q4k', 'q8_0'), default='q4k')
    parser.add_argument('--prefix-cache-n', type=int, default=0,
                        help='sequences held in the prefix cache; 0 disables it. '
                             '0 is the default because it makes every request a '
                             'cold prefill and therefore comparable -- raise it to '
                             'measure what a shared prompt prefix is worth.')
    parser.add_argument('--thinking', choices=('off', 'on'), default='off')
    parser.add_argument('--max-tokens', type=positive, help='override output allowance for every case')
    parser.add_argument('--warmup', type=positive, default=5)
    parser.add_argument('--repetitions', type=positive, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--order-seed', type=int, default=4242)
    parser.add_argument('--only', help='comma-separated case IDs for smoke testing')
    parser.add_argument('--startup-timeout', type=positive, default=600)
    parser.add_argument('--request-timeout', type=positive, default=600)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
