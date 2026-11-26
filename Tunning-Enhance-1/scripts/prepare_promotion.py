#!/usr/bin/env python3
"""Add verified tuning history to the README while preserving the original benchmark."""
import hashlib
import json
from pathlib import Path
import re

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parent

def main():
    summary=json.loads((STUDY/'results/summary.json').read_text())
    decision=summary['decision']
    if decision['status'] not in ('considerable_repeatable_gain','threshold_not_met') or len(summary['pairs'])!=3 or not summary['measurement_checks_passed']:
        raise SystemExit('Publication waits for complete paired measurements and correctness/resource gates.')
    figures=json.loads((STUDY/'assets/figures.json').read_text())
    if figures['source_sha256']!=hashlib.sha256((STUDY/'results/summary.json').read_bytes()).hexdigest():
        raise SystemExit('Regenerate the study figures from the final summary before promotion.')
    demo=json.loads((STUDY/'results/rust-fibonacci-demo.json').read_text())
    if demo['duration_ms']!=30000 or demo.get('observed_startup_budget_gib',0)<31.98 or not demo['containers_unchanged']:
        raise SystemExit('The complete, verified 32 GiB demo is required for this promotion.')
    if hashlib.sha256((STUDY/demo['gif']).read_bytes()).hexdigest()!=demo['gif_sha256']:
        raise SystemExit('Demo checksum differs from the verified export.')
    a=summary['aggregate'][summary['plan']['baseline_profile']]
    b=summary['aggregate'][summary['plan']['candidate_profile']]
    t=decision['median_paired_throughput_gain_fraction']
    l=decision['median_paired_ttft_reduction_fraction']
    historical=json.loads((STUDY/'results/historical-summary.json').read_text())
    old=next(r for r in historical['runs'] if r['profile']=='cuda-conservative-16-32' and r['status']=='complete')
    oh=old['http_summary']
    verdict='Meets the predeclared 20% target.' if decision['promote'] else 'Below the predeclared 20% considerable-gain target; retained as a documented tuning result.'
    block=['<!-- TUNING ENHANCE 1 START -->','','## Tuning history','',
        'Every experiment keeps its own inputs, results, charts, and real terminal recording. The original benchmark above is retained.','',
        '| Study | Date (UTC) | Configuration and scope | Evidence |',
        '|---|---|---|---|',
        '| Original benchmark | 2026-11-25 | CPU and 16 GiB GPU tier, one server start each | [Original results](docs/RESULTS.md) |',
        '| Tunning-Enhance-1 | 2026-11-26 | GPU tier 16 → 32 GiB; RAM cache fixed at 32 GB; three matched pairs | [Complete study](Tunning-Enhance-1/README.md) |','',
        '### Tunning-Enhance-1: larger GPU expert tier','',
        f'**32 GiB GPU expert tier: {t:+.1%} paired median completion throughput; {l:+.1%} reduction in first-output latency.**',
        verdict,'',
        'Three fresh paired trials compare 16 versus 32 GiB on the same Thor, with the RAM expert cache held at 32 decimal GB and the conservative CUDA workaround retained.','',
        '![Fresh repeated throughput and first-output comparison](Tunning-Enhance-1/assets/performance.png)','',
        '| Measured metric | Historical 16 GiB | Fresh 16 GiB control | Fresh 32 GiB tuning |',
        '|---|---:|---:|---:|',
        f"| **Batch completion throughput** | **{oh['successful_completion_tokens_per_second']:.3f} tok/s** | **{a['median_batch_completion_tokens_per_second']:.3f} tok/s** | **{b['median_batch_completion_tokens_per_second']:.3f} tok/s** |",
        f"| **Server TTFT-proxy p50** | **{oh['successful_first_output_seconds']['p50']:.2f} s** | **{a['median_of_server_first_output_p50_seconds']:.2f} s** | **{b['median_of_server_first_output_p50_seconds']:.2f} s** |",
        f"| Server starts / HTTP requests | 1 / 4 | {a['independent_server_starts']} / {a['http_requests']} | {b['independent_server_starts']} / {b['http_requests']} |",
        f"| Known-answer checks | 2/2 | {a['quality_passed']}/{a['quality_total']} | {b['quality_passed']}/{b['quality_total']} |",
        f"| Model/server startup | {old['startup_seconds']:.2f} s | {a['median_startup_seconds']:.2f} s | {b['median_startup_seconds']:.2f} s |",
        f"| Approx. startup + first-request output | {old['startup_seconds']+old['http_attempts'][0]['first_output_seconds']:.2f} s | {a['median_startup_plus_first_request_output_seconds']:.2f} s | {b['median_startup_plus_first_request_output_seconds']:.2f} s |",'',
        'The fresh columns summarize three server starts; the historical column is one earlier start. The paired gain uses only the fresh 16/32 GiB runs, not the historical result. Both old and new datasets remain available.','',
        'The startup-plus-first-request row adds matching timings and omits validation/harness gaps; it is not an exact command-to-token measurement. [Startup and memory trade-offs](Tunning-Enhance-1/RESULTS.md#memory-io-and-startup-trade-offs).','',
        'Throughput includes prefill and HTTP overhead. TTFT is time to first nonempty streamed output, not exact first-token timing. The paired percentages and profile medians summarize different quantities; individual observations and the fixed promotion criterion are linked below. These are measurements on this machine/workload, not a general speedup guarantee.','',
        '[Full tuning results](Tunning-Enhance-1/RESULTS.md) · [Paired gains and memory costs](Tunning-Enhance-1/README.md#repeatability-and-cost) · [Method](Tunning-Enhance-1/METHODOLOGY.md) · [All tuning files](Tunning-Enhance-1/INDEX.md)','',
        '### New 30-second Rust demo','',
        '![Real 32 GiB terminal session with accelerated source clock](Tunning-Enhance-1/assets/rust-fibonacci-30s.gif)','',
        f"The same Rust Fibonacci question, recorded with the 32 GiB configuration: **{demo['source_duration_seconds']:.1f} real seconds at {demo['acceleration_factor']:.2f}×**, plus a three-second final hold, for exactly 30 seconds of playback. This single demo is separate from the paired benchmark. [Source and timing](Tunning-Enhance-1/results/rust-fibonacci-demo.json).",'',
        'Try the 32 GiB study configuration after the [normal model setup](docs/REPRODUCE.md):','',
        '```bash','python3 Tunning-Enhance-1/scripts/chat.py --max-tokens 512','```','',
        '<!-- TUNING ENHANCE 1 END -->']
    path=ROOT/'README.md'; text=path.read_text(); replacement='\n'.join(block)
    if '<!-- TUNING ENHANCE 1 START -->' in text:
        text=re.sub(r'<!-- TUNING ENHANCE 1 START -->.*?<!-- TUNING ENHANCE 1 END -->',replacement,text,flags=re.S)
    else:
        anchor='## Watch GLM write Rust in 30 seconds'
        if anchor not in text: raise SystemExit('Expected baseline README section not found; review before changing layout.')
        text=text.replace(anchor,replacement+'\n\n'+anchor,1)
    text=text.replace('[Results](#benchmark-results) · [30-second demo]', '[Results](#benchmark-results) · [Tuning history](#tuning-history) · [30-second demo]',1)
    label='**Original benchmark — 2026-11-25 UTC, one server start per configuration.**'
    if label not in text:
        text=text.replace('## Benchmark results\n','## Benchmark results\n\n'+label+'\n',1)
    demo_label='**Original 16 GiB recording — retained alongside the new 32 GiB demo above.**'
    if demo_label not in text:
        text=text.replace('## Watch GLM write Rust in 30 seconds\n',
            '## Watch GLM write Rust in 30 seconds\n\n'+demo_label+'\n',1)
    scope='The following status block covers the original benchmark dataset; tuning runs are listed separately above.'
    if scope not in text:
        text=text.replace('<!-- STATUS START -->',scope+'\n\n<!-- STATUS START -->',1)
    phase_scope='**Original benchmark phase measurements.** [Tunning-Enhance-1 phase comparison](Tunning-Enhance-1/assets/phases-and-io.png).'
    if phase_scope not in text:
        text=text.replace('## Where the time goes\n','## Where the time goes\n\n'+phase_scope+'\n',1)
    io_scope='**Original benchmark storage/placement measurements.** [Tunning-Enhance-1 memory and I/O costs](Tunning-Enhance-1/RESULTS.md#memory-io-and-startup-trade-offs).'
    if io_scope not in text:
        text=text.replace('## What SSD streaming and memory placement look like\n',
            '## What SSD streaming and memory placement look like\n\n'+io_scope+'\n',1)
    path.write_text(text)
    print('Prepared side-by-side tuning history, retaining the original benchmark and demo. No Git push or release was performed.')

if __name__=='__main__': main()
