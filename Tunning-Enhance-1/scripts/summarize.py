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
    gain=((t>=threshold-1e-12 and min(throughput)>0 and l>=-.05) or
          (l>=threshold-1e-12 and min(latency)>0 and t>=-.05))
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
    expected_workload=hashlib.sha256((STUDY/'workloads/smoke.jsonl').read_bytes()).hexdigest()
    expected_quality=hashlib.sha256((STUDY/'workloads/quality.json').read_bytes()).hexdigest()
    expected_profiles=json.loads((STUDY/'configs/profiles.json').read_text())
    checks_ok=bool(runs) and all(r['status']=='complete' and not r['numerical_failure_detected'] and
        r['workload_sha256']==expected_workload and r['quality_workload_sha256']==expected_quality and
        r['profile_settings']==expected_profiles[r['profile']] and
        r.get('quality_total') and r['quality_passed']==r['quality_total'] and
        r['protected_containers_unchanged'] and not r['memory_guard_triggered'] and
        r['minimum_global_memavailable_bytes']>=8*2**30 and
        (r['profile']=='cpu-32' or (r.get('observed_device_expert_tier') and
            not r['observed_device_expert_tier']['clamped_below_requested'] and
            r.get('cuda_routed_expert_execution_verified'))) for r in runs)
    if complete:
        checks_ok=checks_ok and state.get('containers_unchanged') is True
    gates={}
    for file in ('model-validation.json','build-validation.json','native-expert-check.json','numerical-diagnostics.json','regression-validation.json'):
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
                'minimum_global_available_gib_across_starts':min(r['minimum_global_memavailable_bytes'] for r in selected)/2**30,
                'median_measured_read_bytes':statistics.median(r['measurement_read_bytes'] for r in selected),
                'median_mean_engine_phase_seconds':{key:statistics.median(statistics.mean(t[key] for t in r['engine_profile']['turns']) for r in selected)
                    for key in ('expert_matmul_s','expert_disk_s','attention_s','lm_head_s')},
                'server_first_output_p50_range':[min(first),max(first)],
                'quality_passed':sum(r['quality_passed'] for r in selected),'quality_total':sum(r['quality_total'] for r in selected)}
            shares=[]
            for run in selected:
                counters=run.get('native_tier_cumulative_counters_at_http_stage') or []
                if counters and counters[-1]['routed_experts']:
                    shares.append(counters[-1]['device_experts_served']/counters[-1]['routed_experts'])
            if shares:
                aggregate[name]['median_engine_device_served_share_at_http_end']=statistics.median(shares)
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
    if plan['baseline_profile'] in aggregate and plan['candidate_profile'] in aggregate:
        a,b=aggregate[plan['baseline_profile']],aggregate[plan['candidate_profile']]
        lines+=['','## Memory, I/O, and startup trade-offs','',
            '| Measured cost | 16 GiB tier | 32 GiB tier |','|---|---:|---:|',
            f"| Median server startup | {a['median_startup_seconds']:.2f} s | {b['median_startup_seconds']:.2f} s |",
            f"| Approx. startup + first-request output | {a['median_startup_plus_first_request_output_seconds']:.2f} s | {b['median_startup_plus_first_request_output_seconds']:.2f} s |",
            f"| Lowest global available memory across starts | {a['minimum_global_available_gib_across_starts']:.2f} GiB | {b['minimum_global_available_gib_across_starts']:.2f} GiB |",
            f"| Median physical reads over four HTTP requests | {a['median_measured_read_bytes']/1e9:.1f} GB | {b['median_measured_read_bytes']/1e9:.1f} GB |",
            f"| Median per-start mean expert-compute timer | {a['median_mean_engine_phase_seconds']['expert_matmul_s']:.2f} s | {b['median_mean_engine_phase_seconds']['expert_matmul_s']:.2f} s |",'',
            f"The engine's device-served/routed share at the end of the HTTP stage rose from {a.get('median_engine_device_served_share_at_http_end',0):.1%} to {b.get('median_engine_device_served_share_at_http_end',0):.1%}. This is that counter's ratio, not a CPU-cache hit rate. More GPU residency accompanies less host expert work; these observations do not isolate a causal kernel speedup. Phase timers may overlap.",'']
    lines+=['','## Decision','',f"**{decision['status']}**. Predeclared promotion threshold: at least 20% paired median improvement in throughput or first-output latency, improving in all three pairs for that metric, with no more than 5% paired median regression in the other metric; every correctness, placement, memory, and container check must pass.",
        '', '## Visual evidence','',
        '![Independent server-start throughput and first-output latency](assets/performance.png)','',
        '![Paired percentage improvements against the fixed threshold](assets/paired-gains.png)','',
        '![Separate phase timers and measured storage reads](assets/phases-and-io.png)','',
        '![Requested and reported placement plus available memory](assets/placement-and-memory.png)','',
        '![Startup and approximate startup-plus-first-request cost](assets/startup-cost.png)','',
        'The startup-plus-first-request panel adds matching per-start measurements and omits validation and harness gaps. It is not an exact command-to-token measurement. A throughput improvement can coexist with slower loading.','',
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
        if decision['status']=='threshold_not_met':
            headline+=['','The completed measurements did not meet the predeclared considerable-gain target. The parent baseline remains unchanged; this study is documented without a headline upgrade or a new performance release under that rule.']
        text=re.sub(r'<!-- STUDY STATUS START -->.*?<!-- STUDY STATUS END -->',
            '<!-- STUDY STATUS START -->\n\n'+'\n'.join(headline)+'\n\n<!-- STUDY STATUS END -->',readme.read_text(),flags=re.S)
        demo_path=STUDY/'results/rust-fibonacci-demo.json'
        if demo_path.exists():
            demo=json.loads(demo_path.read_text())
            demo_text=['<!-- DEMO START -->','',
                '![Real 32 GiB Rust Fibonacci session with an accelerated elapsed-time clock](assets/rust-fibonacci-30s.gif)','',
                f"The same prompt and output cap as the original demo, using this study's `scripts/chat.py`: **{demo['source_duration_seconds']:.1f} real seconds at {demo['acceleration_factor']:.2f}×**, followed by a 3-second hold, for exactly 30 seconds. Model loading is excluded; this single demonstration is not the paired benchmark.",'',
                '[Source/timing](results/rust-fibonacci-demo.json) · [Terminal output](results/rust-fibonacci-output.txt).','',
                '<!-- DEMO END -->']
            text=re.sub(r'<!-- DEMO START -->.*?<!-- DEMO END -->','\n'.join(demo_text),text,flags=re.S)
        readme.write_text(text)
    print(json.dumps({'attempts':len(runs),'pairs':len(pairs),'decision':decision}))


if __name__=='__main__':
    main()
