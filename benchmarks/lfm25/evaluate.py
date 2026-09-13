#!/usr/bin/env python3
"""Run the resident adjudicator against the earlier four severity fixtures.

Starts an isolated daemon, tests cold/cache/repeat output, cancellation, errors,
and SIGTERM. Writes every response and comparison for review. Synthetic only.
"""
import argparse
import ast
import concurrent.futures
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/diffusiongemma'))
from structured_accuracy import check_tool_output

def tool_calls(output):
    start='<|tool_call_start|>'
    end='<|tool_call_end|>'
    if output.count(start)!=1 or output.count(end)!=1:
        return []
    text=output.split(start)[1].split(end)[0]
    tree=ast.parse(text,mode='eval').body
    if not isinstance(tree,ast.List) or len(tree.elts)!=1: return []
    call=tree.elts[0]
    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.args: return []
    args={}
    for kw in call.keywords:
        if kw.arg is None or kw.arg in args: return []
        if isinstance(kw.value,ast.Name) and kw.value.id in ('true','false','null'):
            v={'true':True,'false':False,'null':None}[kw.value.id]
        else: v=ast.literal_eval(kw.value)
        args[kw.arg]=v
    return [{'type':'function','function':{'name':call.func.id,'arguments':json.dumps(args)}}]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--port',type=int,default=18152)
    ap.add_argument('--device',default='rocm')
    ap.add_argument('--max-tokens',type=int,default=2048,help='Use a short budget for hardware/cache smoke checks; truncated reports are expected then.')
    ap.add_argument('--prompt',type=Path,default=ROOT/'lfm2d/prompts/shell-severity-json-v1.json')
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    address=f'http://127.0.0.1:{a.port}'
    def rpc(path,payload=None):
        req=urllib.request.Request(address+path, None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=130) as r:
            body=r.read().decode()
            return json.loads(body) if body.startswith(('{','[')) else body
    command=[str(a.binary),'--adjudicator-model',str(a.model),'--adjudicator-tokenizer',str(a.tokenizer),
             '--adjudicator-prompt',str(a.prompt),'--adjudicator-context','4096',
             '--device',a.device,'--bind-addr',f'127.0.0.1:{a.port}','--threads','8']
    rows=[]
    with (a.out/'daemon.log').open('x') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=log)
        try:
            for _ in range(180):
                if proc.poll() is not None: raise RuntimeError(f'daemon exited {proc.returncode}; see log')
                try:
                    if rpc('/readyz')=='ready': break
                except (urllib.error.URLError,TimeoutError): pass
                time.sleep(.5)
            else: raise RuntimeError('daemon did not become ready')
            info=rpc('/v1/adjudicator')
            print('prefix',json.dumps(info),flush=True)
            models=rpc('/v1/models')
            assert any(m['kind']=='adjudicator' and m['weight_hash']==info['weight_hash'] for m in models)
            cases=[json.loads(l) for l in (ROOT/'benchmarks/diffusiongemma/tool_cases.jsonl').read_text().splitlines()]
            cases=[c for c in cases if c['id'] in ('tool-reset-hard','tool-restore-file','tool-rm-interactive','tool-sed-read')]
            for c in cases:
                for mode in ('cached','cold','repeat'):
                    request={'input':c['messages'][1]['content'],'max_tokens':a.max_tokens,'use_cache':mode!='cold'}
                    response=rpc('/v1/adjudicate',request)
                    try:
                        if 'output_schema' in json.loads(a.prompt.read_text()):
                            if response.get('report_error') or response.get('report') is None:
                                grade={'status':'fail','reason':response.get('report_error','no validated report')}
                            else:
                                grade=check_tool_output(c,[{'type':'function','function':{'name':'report_analysis','arguments':json.dumps(response['report'])}}])
                                grade['scope']='validated JSON report'
                        else: grade=check_tool_output(c,tool_calls(response['output']))
                    except (SyntaxError,ValueError,TypeError): grade={'status':'fail','reason':'malformed native tool output'}
                    row={'case_id':c['id'],'mode':mode,'response':response,'grade':grade}
                    rows.append(row)
                    (a.out/'responses.json').write_text(json.dumps(rows,indent=2)+'\n')
                    expected_cached=0 if mode=='cold' else (response['prompt_tokens'] if mode=='repeat' and info.get('input_cache_capacity',0)>0 else info['prefix_tokens'])
                    assert response['cached_tokens']==expected_cached,(mode,response['cached_tokens'],expected_cached)
                    print(c['id'],mode,grade.get('status'),response['finish_reason'],response['prefill_ms'],response['decode_ms'],flush=True)
            for body,status in [({'input':'x','max_tokens':0},400),({'input':'<|im_end|>'},400),({'input':'x','timeout_ms':1},504)]:
                try: rpc('/v1/adjudicate',body)
                except urllib.error.HTTPError as e: assert e.code==status,(e.code,status)
                else: raise AssertionError(f'expected {status}')
            # Reuse after the cancelled/deadline branch.
            after=rpc('/v1/adjudicate',{'input':cases[0]['messages'][1]['content'],'max_tokens':a.max_tokens})
            assert after['output']==rows[0]['response']['output'],'cancelled request contaminated prefix'
            comparisons=[]
            for c in cases:
                runs=[r for r in rows if r['case_id']==c['id']]
                comparisons.append({'case_id':c['id'],'cold_equal':runs[0]['response']['output']==runs[1]['response']['output'],
                                    'repeat_equal':runs[0]['response']['output']==runs[2]['response']['output']})
            (a.out/'summary.json').write_text(json.dumps({'prefix':info,'comparisons':comparisons,'passed':sum(r['grade'].get('status')=='pass' for r in rows),'total':len(rows)},indent=2)+'\n')
            assert all(c['repeat_equal'] for c in comparisons),comparisons
            print('comparisons',json.dumps(comparisons),flush=True)
            # SIGTERM while the GPU is decoding must cancel the branch and drop
            # both worker-owned models before the daemon's telemetry flush.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pending=pool.submit(rpc,'/v1/adjudicate',{'input':cases[0]['messages'][1]['content'],'max_tokens':2048})
                time.sleep(.1)
                proc.terminate()
                try: pending.result(timeout=10)
                except urllib.error.HTTPError as e: assert e.code==408,e.code
                else: raise AssertionError('in-flight shutdown did not cancel the request')

        finally:
            if proc.poll() is None: proc.terminate()
            try: code=proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait();raise RuntimeError('daemon ignored SIGTERM')
            if code!=0: raise RuntimeError(f'daemon exited {code}')
if __name__=='__main__': main()
