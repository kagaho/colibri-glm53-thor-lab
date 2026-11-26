#!/usr/bin/env python3
"""Render the README's cross-study comparison without pooling historical and paired trials."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime/matplotlib'))
SOURCES = ('results/summary.json', 'Tunning-Enhance-1/results/summary.json')


def load_rows():
    original, followup = [json.loads((ROOT / p).read_text()) for p in SOURCES]
    rows = []
    profiles = [('cpu-32', 'CPU'), ('cuda-conservative-16-32', 'CUDA · 16 GiB')]
    for profile, label in profiles:
        run = next(r for r in original['runs'] if r['profile'] == profile and r['status'] == 'complete')
        if run.get('numerical_failure_detected') or not run['quality_total'] or run['quality_passed'] != run['quality_total']:
            raise ValueError('Historical comparison requires valid output and passed answer checks')
        rows.append(dict(group='Original · Nov 25 UTC', label=label, starts=1,
            throughput=run['http_summary']['successful_completion_tokens_per_second'],
            first_output=run['http_summary']['successful_first_output_seconds']['p50'],
            available=run['minimum_global_memavailable_bytes']/2**30))
    if not followup['measurement_checks_passed']:
        raise ValueError('Follow-up measurement checks must pass')
    for profile, label in profiles + [('cuda-conservative-32-32', 'CUDA · 32 GiB')]:
        data = followup['aggregate'][profile]
        if not data['quality_total'] or data['quality_passed'] != data['quality_total']:
            raise ValueError('Follow-up answer checks must pass')
        rows.append(dict(group='Follow-up · Nov 26 UTC', label=label,
            starts=data['independent_server_starts'],
            throughput=data['median_batch_completion_tokens_per_second'],
            first_output=data['median_of_server_first_output_p50_seconds'],
            available=data['minimum_global_available_gib_across_starts']))
    return rows


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'text.color': '#172a3b', 'axes.labelcolor': '#526674', 'xtick.color': '#526674',
        'ytick.color': '#172a3b', 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.edgecolor': '#c9d5dd',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'svg.fonttype': 'none', 'svg.hashsalt': 'colibri-cross-study'})
    rows = load_rows()
    fig, axes = plt.subplots(1, 3, figsize=(14, 7.0))
    fig.subplots_adjust(left=.18, right=.96, top=.75, bottom=.23, wspace=.28)
    fig.text(.035, .955, 'More GPU-resident experts: speed versus shared-memory headroom',
        fontsize=18, weight='bold', va='top')
    fig.text(.035, .90, 'GLM-5.3-Flash native int4 · Thor · 32 GB RAM expert cache · CUDA uses COLI_CUDA_TC_GEMM=0',
        fontsize=11, color='#526674', va='top')
    colors = ['#9aafc5', '#8dbeb8', '#456d99', '#008778', '#a060c5']
    positions = [0, 1, 2.65, 3.65, 4.65]
    fields = [('throughput', 'Completion throughput', 'tok/s · higher is better', .66, '.3f'),
              ('first_output', 'First-output latency', 'seconds · lower is better', 59, '.2f'),
              ('available', 'Available shared memory', 'GiB · lowest observed', 95, '.2f')]
    for ax, (field, title, xlabel, limit, fmt) in zip(axes, fields):
        ax.axhspan(-.5, 1.5, color='#f3f6f9', zorder=0)
        for y, row, color in zip(positions, rows, colors):
            value = row[field]
            ax.barh(y, value, height=.48, color=color, zorder=3)
            ax.text(value+limit*.025, y, format(value, fmt), va='center', fontsize=11, weight='bold', color='#526674' if y < 2 else color)
        ax.set(xlim=(0, limit), ylim=(5.15, -.55), xlabel=xlabel)
        ax.set_title(title, loc='left', pad=33, fontsize=12, weight='bold')
        ax.set_yticks(positions)
        ax.set_yticklabels([r['label'] for r in rows] if ax is axes[0] else [])
        ax.tick_params(axis='y', length=0, pad=9)
        ax.xaxis.grid(True, color='#e7edf1', zorder=1)
        ax.set_axisbelow(True)
    axes[0].text(0, 1.06, 'Original · 1 start each', transform=axes[0].transAxes, fontsize=9, color='#526674')
    axes[0].text(-.54, .43, 'Follow-up', transform=axes[0].transAxes, fontsize=10, weight='bold', color='#172a3b')
    fig.text(.035, .14, 'Original: one start per profile. Follow-up: one CPU start; CUDA speed/latency are medians across three fresh starts.',
        fontsize=10, color='#526674')
    fig.text(.035, .10, 'Throughput includes prefill/HTTP overhead. Latency is the first nonempty streamed-output proxy, not exact token timing.',
        fontsize=10, color='#526674')
    fig.text(.035, .06, 'Memory is lowest global available RAM, not an allocation or median. Default CUDA is invalid and excluded; no historical-pair speedup claim.',
        fontsize=9.5, color='#526674')
    folder = ROOT / 'docs/assets'
    stem = folder / 'benchmark-comparison'
    svg = stem.with_suffix('.svg')
    fig.savefig(svg, metadata={'Title': 'Historical and follow-up benchmark comparison',
        'Description': 'Separate studies; single historical starts and follow-up per-start medians. See benchmark-comparison.json.',
        'Creator': 'scripts/render_comparison.py', 'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(stem.with_suffix('.png'), dpi=150, metadata={'Software': 'scripts/render_comparison.py'})
    plt.close(fig)
    provenance = dict(renderer='scripts/render_comparison.py',
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matplotlib_version=matplotlib.__version__,
        sources=[dict(path=p, sha256=hashlib.sha256((ROOT / p).read_bytes()).hexdigest()) for p in SOURCES],
        figures=['benchmark-comparison.png', 'benchmark-comparison.svg'],
        rows=rows,
        notes=['Historical and follow-up groups are not pooled or treated as paired trials.',
            'CUDA throughput and latency follow-up bars are per-start medians; CPU has one follow-up start.',
            'Latency is the median of server first-output p50 values, not exact first-token timing.',
            'Available memory is the minimum across all starts, not an allocation or per-start median.',
            'Only complete quality-checked profiles appear; default CUDA is invalid and excluded.'])
    stem.with_suffix('.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('Rendered cross-study comparison as PNG/SVG with public source hashes and plotted values.')


if __name__ == '__main__':
    main()
