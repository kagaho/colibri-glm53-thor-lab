#!/usr/bin/env python3
"""Curate isolated attempts and assess the predeclared paired improvement rule."""
import hashlib
import json
import re
from pathlib import Path
import statistics
import sys

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parent
sys.path.insert(0,str(ROOT/'scripts'))
import summarize as baseline_summary


def assess(pairs,complete,checks_ok,threshold=.20):
    if len(pairs)!=3 or not complete or not checks_ok:
        return {'promote':False,'status':'pending_or_failed_checks','threshold_fraction':threshold}
    throughput=[p['throughput_gain_fraction'] for p in pairs]
    latency=[p['ttft_reduction_fraction'] for p in pairs]
    t,l=statistics.median(throughput),statistics.median(latency)
    gain=((t>=threshold and min(throughput)>0 and l>=-.05) or
          (l>=threshold and min(latency)>0 and t>=-.05))
    return {'promote':gain,'status':'considerable_repeatable_gain' if gain else 'threshold_not_met',
            'median_paired_throughput_gain_fraction':t,'median_paired_ttft_reduction_fraction':l,
            'throughput_gain_range':[min(throughput),max(throughput)],
            'ttft_reduction_range':[min(latency),max(latency)],'threshold_fraction':threshold,
            'interpretation':'Three paired server starts; descriptive evidence, not a statistical significance claim'}


def main():
    plan=json.loads((STUDY/'configs/study.json').read_text())
    runs=[]
    for path in sorted((STUDY/'results/raw').glob('*/run.json')):
        data=json.loads(path.read_text())
        row=baseline_summary.summarize_run(data,path.parent)
        row['quality_workload_sha256']=data.get('quality_workload_sha256')
        row['memory_guard_triggered']=data.get('memory_guard_triggered',False)
        quality=path.parent/'quality.json'
        if quality.exists():
            row['quality_cases']=[{'id':c['id'],'passed':c['passed'],
                'answer':c['response']['choices'][0]['message'].get('content') or ''}
                for c in json.loads(quality.read_text())]
        runs.append(row)
    state_path=STUDY/'runtime/campaign-state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    by_id={r['run_id']:r for r in runs}
    ordered=[by_id[c['run_id']] for c in state.get('completed',[]) if c['run_id'] in by_id]
    gpu=[r for r in ordered if r['profile'] in (plan['baseline_profile'],plan['candidate_profile'])]
    pairs=[]
    for index in range(0,len(gpu)-1,2):
        pair={r['profile']:r for r in gpu[index:index+2]}
        if set(pair)!={plan['baseline_profile'],plan['candidate_profile']}:
            continue
        a,b=pair[plan['baseline_profile']],pair[plan['candidate_profile']]
        ah,bh=a['http_summary'],b['http_summary']
        pairs.append({'pair':index//2+1,'order':[r['profile'] for r in gpu[index:index+2]],
            'baseline_run_id':a['run_id'],'candidate_run_id':b['run_id'],
            'throughput_gain_fraction':bh['successful_completion_tokens_per_second']/ah['successful_completion_tokens_per_second']-1,
            'ttft_reduction_fraction':1-bh['successful_first_output_seconds']['p50']/ah['successful_first_output_seconds']['p50']})
    complete=len(ordered)==len(plan['run_order']) and len(runs)==len(ordered)
    checks_ok=bool(runs) and all(r['status']=='complete' and not r['numerical_failure_detected'] and
        r.get('quality_total') and r['quality_passed']==r['quality_total'] and
        r['protected_containers_unchanged'] and not r['memory_guard_triggered'] and
        r['minimum_global_memavailable_bytes']>=8*2**30 and
        (r['profile']=='cpu-32' or (r.get('observed_device_expert_tier') and
            not r['observed_device_expert_tier']['clamped_below_requested'] and
            r.get('cuda_routed_expert_execution_verified'))) for r in runs)
    gates={}
    for file in ('model-validation.json','build-validation.json','native-expert-check.json','numerical-diagnostics.json'):
        path=STUDY/'results'/file
        gates[file]=json.loads(path.read_text()).get('status')=='passed' if path.exists() else None
    validated=all(v is True for v in gates.values())
    decision=assess(pairs,complete,checks_ok and validated,plan['meaningful_gain_fraction'])
    aggregate={}
    for name in ('cpu-32',plan['baseline_profile'],plan['candidate_profile']):
        selected=[r for r in ordered if r['profile']==name and r['status']=='complete']
        if selected:
            rates=[r['http_summary']['successful_completion_tokens_per_second'] for r in selected]
            first=[r['http_summary']['successful_first_output_seconds']['p50'] for r in selected]
            aggregate[name]={'independent_server_starts':len(selected),'http_requests':sum(len(r['http_attempts']) for r in selected),
                'median_batch_completion_tokens_per_second':statistics.median(rates),
                'throughput_range':[min(rates),max(rates)],'median_of_server_first_output_p50_seconds':statistics.median(first),
                'median_startup_seconds':statistics.median(r['startup_seconds'] for r in selected),
                'median_startup_plus_first_request_output_seconds':statistics.median(r['startup_seconds']+r['http_attempts'][0]['first_output_seconds'] for r in selected),
                'server_first_output_p50_range':[min(first),max(first)],
                'quality_passed':sum(r['quality_passed'] for r in selected),'quality_total':sum(r['quality_total'] for r in selected)}
    result={'study':'Tunning-Enhance-1','status':'measured' if complete else 'partial','plan':plan,
        'metric':'Whole measured HTTP batch completion throughput including prefill/HTTP; TTFT proxy is first nonempty output',
        'validation_gates':gates,'measurement_checks_passed':checks_ok,'runs':runs,'pairs':pairs,
        'aggregate':aggregate,'decision':decision,
        'limitations':['One Thor and two short synthetic prompts; not a model quality benchmark',
            'Four requests within each server start are correlated; independent repetitions are server starts',
            'Three alternated pairs, uncontrolled OS page cache and concurrent user-service I/O; no cold-SSD or statistical-significance claim',
            'Only requested GPU expert-tier budget changes between the two CUDA profiles; shared UMA allocations still interact',
            'CPU reference is one fresh server start; historical baseline is context, not the tuning gain denominator']}
    (STUDY/'results/summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Tunning-Enhance-1 results','',f"Recorded {len(runs)} attempt(s); {len(pairs)} complete comparison pair(s).",'',
        '| Configuration | Starts | HTTP requests | Median batch tok/s | Median server TTFT-proxy p50 (s) | Answer checks |',
        '|---|---:|---:|---:|---:|---:|']
    for name,row in aggregate.items():
        lines.append(f"| {name} | {row['independent_server_starts']} | {row['http_requests']} | {row['median_batch_completion_tokens_per_second']:.3f} | {row['median_of_server_first_output_p50_seconds']:.2f} | {row['quality_passed']}/{row['quality_total']} |")
    lines+=['','## Paired improvements','','| Pair | Throughput change | TTFT-proxy reduction |','|---|---:|---:|']
    for pair in pairs:
        lines.append(f"| {pair['pair']} | {pair['throughput_gain_fraction']:+.1%} | {pair['ttft_reduction_fraction']:+.1%} |")
    lines+=['','## Decision','',f"**{decision['status']}**. Predeclared promotion threshold: at least 20% paired median improvement in throughput or first-output latency, improving in all three pairs for that metric, with no more than 5% paired median regression in the other metric; every correctness, placement, memory, and container check must pass.",
        '', '## Visual evidence','',
        '![Independent server-start throughput and first-output latency](assets/performance.png)','',
        '![Paired percentage improvements against the fixed threshold](assets/paired-gains.png)','',
        '![Separate phase timers and measured storage reads](assets/phases-and-io.png)','',
        '![Requested and reported placement plus available memory](assets/placement-and-memory.png)','',
        '[Raw public metrics](results/summary.json) · [Method](METHODOLOGY.md) · [Reproduce](REPRODUCE.md) · [File index](INDEX.md)',
        '', '## Limits','']+['- '+text for text in result['limitations']]
    (STUDY/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    readme=STUDY/'README.md'
    if readme.exists():
        headline=[f"**Status:** {len(runs)} recorded attempt(s), {len(pairs)}/3 complete pairs; **{decision['status']}**.",'',
            '| Configuration | Independent starts | Median batch completion tok/s | Median server TTFT-proxy p50 | Answer checks |',
            '|---|---:|---:|---:|---:|']
        for name,row in aggregate.items():
            headline.append(f"| {name} | {row['independent_server_starts']} | {row['median_batch_completion_tokens_per_second']:.3f} | {row['median_of_server_first_output_p50_seconds']:.2f} s | {row['quality_passed']}/{row['quality_total']} |")
        if 'median_paired_throughput_gain_fraction' in decision:
            headline+=['',f"**Paired median:** throughput {decision['median_paired_throughput_gain_fraction']:+.1%}; TTFT-proxy reduction {decision['median_paired_ttft_reduction_fraction']:+.1%}. These compare fresh 32 GiB trials with fresh 16 GiB trials, not the historical session."]
        readme.write_text(re.sub(r'<!-- STUDY STATUS START -->.*?<!-- STUDY STATUS END -->',
            '<!-- STUDY STATUS START -->\n\n'+'\n'.join(headline)+'\n\n<!-- STUDY STATUS END -->',readme.read_text(),flags=re.S))
    print(json.dumps({'attempts':len(runs),'pairs':len(pairs),'decision':decision}))


if __name__=='__main__':
    main()
