#!/usr/bin/env python3
"""Prepare the parent README only after the predeclared gain and evidence checks pass."""
import hashlib
import json
from pathlib import Path
import re

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parent

def main():
    summary=json.loads((STUDY/'results/summary.json').read_text())
    decision=summary['decision']
    if not decision.get('promote') or len(summary['pairs'])!=3 or not summary['measurement_checks_passed']:
        raise SystemExit('No promotion: the predeclared paired gain and correctness/resource gates did not pass.')
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
    block=['<!-- TUNING ENHANCE 1 START -->','','## Tuning Enhance 1 results','',
        f'**32 GiB GPU expert tier: {t:+.1%} paired median completion throughput; {l:+.1%} reduction in first-output latency.**',
        'Three fresh paired trials compare 16 versus 32 GiB on the same Thor, with the RAM expert cache held at 32 decimal GB and the conservative CUDA workaround retained.','',
        '![Fresh repeated throughput and first-output comparison](Tunning-Enhance-1/assets/performance.png)','',
        '| Measured metric | 16 GiB CUDA tier | 32 GiB CUDA tier |',
        '|---|---:|---:|',
        f"| **Median batch completion throughput** | **{a['median_batch_completion_tokens_per_second']:.3f} tok/s** | **{b['median_batch_completion_tokens_per_second']:.3f} tok/s** |",
        f"| **Median server TTFT-proxy p50** | **{a['median_of_server_first_output_p50_seconds']:.2f} s** | **{b['median_of_server_first_output_p50_seconds']:.2f} s** |",
        f"| Independent server starts / HTTP requests | {a['independent_server_starts']} / {a['http_requests']} | {b['independent_server_starts']} / {b['http_requests']} |",
        f"| Known-answer checks | {a['quality_passed']}/{a['quality_total']} | {b['quality_passed']}/{b['quality_total']} |",
        f"| Median model/server startup | {a['median_startup_seconds']:.2f} s | {b['median_startup_seconds']:.2f} s |",'',
        'Throughput includes prefill and HTTP overhead. TTFT is time to first nonempty streamed output, not exact first-token timing. The paired percentages and profile medians summarize different quantities; individual observations and the fixed promotion criterion are linked below. These are measurements on this machine/workload, not a general speedup guarantee.','',
        '[Full tuning results](Tunning-Enhance-1/RESULTS.md) · [Paired gains and memory costs](Tunning-Enhance-1/README.md#repeatability-and-cost) · [Method](Tunning-Enhance-1/METHODOLOGY.md) · [All tuning files](Tunning-Enhance-1/INDEX.md)','',
        '### New 30-second Rust demo','',
        '![Real 32 GiB terminal session with accelerated source clock](Tunning-Enhance-1/assets/rust-fibonacci-30s.gif)','',
        f"The same Rust Fibonacci question, recorded with the 32 GiB configuration: **{demo['source_duration_seconds']:.1f} real seconds at {demo['acceleration_factor']:.2f}×**, plus a three-second final hold, for exactly 30 seconds of playback. This single demo is separate from the paired benchmark. [Source and timing](Tunning-Enhance-1/results/rust-fibonacci-demo.json).",'',
        'Try the tuned helper after the [normal model setup](docs/REPRODUCE.md):','',
        '```bash','python3 Tunning-Enhance-1/scripts/chat.py --max-tokens 512','```','',
        '<!-- TUNING ENHANCE 1 END -->']
    path=ROOT/'README.md'; text=path.read_text(); replacement='\n'.join(block)
    if '<!-- TUNING ENHANCE 1 START -->' in text:
        text=re.sub(r'<!-- TUNING ENHANCE 1 START -->.*?<!-- TUNING ENHANCE 1 END -->',replacement,text,flags=re.S)
    else:
        hero='![CPU versus conservative CUDA: completion tokens per second and TTFT proxy](docs/assets/results-overview.png)'
        if hero not in text: raise SystemExit('Expected baseline README hero not found; review before changing layout.')
        text=text.replace(hero,replacement+'\n\n## Original baseline snapshot\n\n'+hero,1)
    text=text.replace('[Results](#benchmark-results)','[Latest results](#tuning-enhance-1-results)',1)
    text=text.replace('[30-second demo](#watch-glm-write-rust-in-30-seconds)','[30-second demo](#new-30-second-rust-demo)',1)
    path.write_text(text)
    print('Prepared the parent README from verified tuning evidence. No Git push or release was performed.')

if __name__=='__main__': main()
