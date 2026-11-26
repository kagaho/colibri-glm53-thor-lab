#!/usr/bin/env python3
"""Render reproducible tuning comparisons and a compact study workflow."""
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parent
os.environ.setdefault('MPLCONFIGDIR',str(STUDY/'runtime/matplotlib'))
COLORS={'cpu-32':'#8793a1','cuda-conservative-16-32':'#456d99','cuda-conservative-32-32':'#008778'}
LABELS={'cpu-32':'CPU reference','cuda-conservative-16-32':'16 GiB tier','cuda-conservative-32-32':'32 GiB tier'}

def save(fig,name,source_hash):
    import matplotlib.pyplot as plt
    folder=STUDY/'assets'; folder.mkdir(exist_ok=True)
    path=folder/(name+'.svg')
    fig.savefig(path,metadata={'Date':None,'Creator':'Tunning-Enhance-1/scripts/render_results.py',
        'Description':'Source: results/summary.json SHA256 '+source_hash})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(folder/(name+'.png'),dpi=150,metadata={'Software':'Tunning-Enhance-1/scripts/render_results.py'})
    plt.close(fig)

def good_runs(summary):
    return [r for r in summary['runs'] if r['status']=='complete' and not r['numerical_failure_detected']
            and r.get('quality_total') and r['quality_passed']==r['quality_total']]

def heading(fig,title,subtitle):
    fig.text(.06,.95,title,fontsize=18,weight='bold',va='top')
    fig.text(.06,.885,subtitle,fontsize=10,color='#526674',va='top')

def performance(summary,runs,sha):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5.6))
    fig.subplots_adjust(left=.17,right=.96,top=.70,bottom=.23,wspace=.45)
    heading(fig,'Does a larger GPU expert tier help?',
            'Native int4 · same 32 GB RAM expert cache · tensor-core GEMMs disabled · independent server starts')
    names=[n for n in LABELS if any(r['profile']==n for r in runs)]
    for y,name in enumerate(names):
        selected=[r for r in runs if r['profile']==name]
        rates=[r['http_summary']['successful_completion_tokens_per_second'] for r in selected]
        first=[r['http_summary']['successful_first_output_seconds']['p50'] for r in selected]
        for ax,values,format_ in [(axes[0],rates,'.3f'),(axes[1],first,'.1f')]:
            median=statistics.median(values)
            ax.barh(y,median,height=.45,color=COLORS[name],alpha=.78)
            ax.scatter(values,[y+(i-(len(values)-1)/2)*.10 for i in range(len(values))],color=COLORS[name],edgecolor='white',s=55,zorder=3)
            ax.text(median+max(values)*.025,y,format(median,format_),va='center',weight='bold',color=COLORS[name])
    for ax in axes:
        ax.set_yticks(range(len(names)),[LABELS[n]+f" (n={sum(r['profile']==n for r in runs)})" for n in names])
        ax.invert_yaxis(); ax.xaxis.grid(True,color='#e7edf1'); ax.set_axisbelow(True)
        ax.set_xlim(0,ax.get_xlim()[1]*1.17)
    axes[0].set_title('Completion throughput',loc='left',fontsize=12,pad=12)
    axes[0].set_xlabel('Tokens/s over the whole HTTP batch')
    axes[1].set_title('TTFT proxy: first nonempty output',loc='left',fontsize=12,pad=12)
    axes[1].set_xlabel('Median of server-level p50 values · seconds')
    fig.text(.06,.11,'Dots = independent server starts; bars = medians. Each start contains four correlated HTTP requests.',fontsize=10)
    fig.text(.06,.055,'Throughput includes prefill and HTTP overhead; TTFT is a proxy. One machine, uncontrolled OS cache, no significance claim.',fontsize=9,color='#526674')
    save(fig,'performance',sha)

def paired(summary,sha):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5.5))
    fig.subplots_adjust(left=.12,right=.95,top=.72,bottom=.22)
    heading(fig,'Repeatability before promotion','Each pair compares one fresh 16 GiB start with one fresh 32 GiB start; pair order alternates')
    for index,pair in enumerate(summary['pairs']):
        for offset,key,label,color in [(-.16,'throughput_gain_fraction','Throughput improvement','#008778'),
                                        (.16,'ttft_reduction_fraction','TTFT-proxy reduction','#456d99')]:
            value=100*pair[key]
            ax.bar(index+offset,value,width=.28,color=color,label=label if index==0 else None)
            ax.text(index+offset,value+(2 if value>=0 else -2),f'{value:+.1f}%',ha='center',va='bottom' if value>=0 else 'top',fontsize=10)
    ax.axhline(20,color='#b13d3d',ls='--',lw=1.5,label='Predeclared considerable-gain target: 20%')
    ax.axhline(0,color='#8793a1',lw=.8)
    ax.set_xticks(range(len(summary['pairs'])),[f"Pair {p['pair']}" for p in summary['pairs']])
    ax.set_ylabel('Improvement (%) · positive is better')
    low,high=ax.get_ylim(); ax.set_ylim(min(low,-5),max(high*1.2,35))
    ax.yaxis.grid(True,color='#e7edf1'); ax.set_axisbelow(True)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.17),ncol=2,frameon=False,fontsize=9)
    fig.text(.06,.045,'The winning metric must improve in all three pairs; other metric may not regress by more than 5% at the paired median.',fontsize=9,color='#526674')
    save(fig,'paired-gains',sha)

def phases(runs,sha):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    fig.subplots_adjust(left=.15,right=.95,top=.73,bottom=.25,wspace=.43)
    heading(fig,'Compute and storage costs','Reported phase timers and owned-process read counters; neither is a causal breakdown of latency')
    fields=[('expert_matmul_s','Expert compute'),('expert_disk_s','Expert disk'),('attention_s','Attention'),('lm_head_s','Output head')]
    for index,name in enumerate(LABELS):
        selected=[r for r in runs if r['profile']==name]
        if not selected: continue
        values=[statistics.median(statistics.mean(t[key] for t in r['engine_profile']['turns']) for r in selected) for key,_ in fields]
        ys=[y+(index-1)*.24 for y in range(len(fields))]
        axes[0].barh(ys,values,height=.21,color=COLORS[name],label=LABELS[name])
        for y,value in zip(ys,values): axes[0].text(value+.7,y,f'{value:.1f}',fontsize=9,va='center',color=COLORS[name])
        reads=[r['measurement_read_bytes']/1e9 for r in selected]
        axes[1].barh(index,statistics.median(reads),height=.45,color=COLORS[name])
        axes[1].scatter(reads,[index]*len(reads),s=30,color=COLORS[name],edgecolor='white',zorder=3)
        axes[1].text(statistics.median(reads)+6,index,f'{statistics.median(reads):.1f}',va='center',color=COLORS[name])
    axes[0].set_yticks(range(len(fields)),[label for _,label in fields]); axes[0].invert_yaxis()
    axes[0].set_xlabel('Median of per-start mean phase seconds')
    axes[0].set_xlim(0,axes[0].get_xlim()[1]*1.14)
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,-.20),ncol=2,frameon=False,fontsize=9)
    axes[1].set_yticks(range(3),list(LABELS.values())); axes[1].invert_yaxis()
    axes[1].set_xlabel('Median decimal GB read during four HTTP requests')
    axes[1].set_xlim(0,axes[1].get_xlim()[1]*1.18)
    for ax in axes: ax.xaxis.grid(True,color='#e7edf1'); ax.set_axisbelow(True)
    fig.text(.06,.06,'Phase timers may overlap and must not be summed. Reads exclude loading and separate quality checks; repeated reads can exceed file size.',fontsize=9,color='#526674')
    save(fig,'phases-and-io',sha)

def resources(runs,sha):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    fig.subplots_adjust(left=.15,right=.96,top=.74,bottom=.22,wspace=.40)
    heading(fig,'Verify actual placement and memory headroom','Thor CPU and GPU allocations share one physical memory pool; a requested budget is not measured allocation')
    selected=[r for r in runs if r['profile']!='cpu-32']
    labels=[]
    counts={}
    for i,run in enumerate(selected):
        name=run['profile']; counts[name]=counts.get(name,0)+1
        labels.append(LABELS[name]+f' #{counts[name]}')
        tier=run['observed_device_expert_tier'] or {}
        value=tier.get('reported_budget_gib',0)
        axes[0].barh(i,run['profile_settings']['gpu_tier_gib'],height=.62,color='#e6edf1')
        axes[0].barh(i,value,height=.32,color=COLORS[name])
        axes[0].text(value+.5,i,f'{value:.2f}',va='center',fontsize=9)
        available=run['minimum_global_memavailable_bytes']/2**30
        axes[1].barh(i,available,height=.45,color=COLORS[name])
        axes[1].text(available+.6,i,f'{available:.1f}',va='center',fontsize=9)
    axes[1].axvline(8,color='#b13d3d',ls='--',label='Memory stop threshold: 8 GiB')
    for ax in axes:
        ax.set_yticks(range(len(labels)),labels); ax.invert_yaxis()
        ax.xaxis.grid(True,color='#e7edf1'); ax.set_axisbelow(True)
        ax.set_xlim(0,ax.get_xlim()[1]*1.15)
    axes[0].set_xlabel('GiB · colored: reported, pale: requested tier')
    axes[1].set_xlabel('Minimum global MemAvailable during the run · GiB')
    axes[1].legend(loc='upper left',bbox_to_anchor=(0,-.18),frameon=False,fontsize=9)
    fig.text(.06,.055,'Startup tier values are rounded capacities. Global available memory is shown instead of adding RSS, which can double-count shared pages.',fontsize=9,color='#526674')
    save(fig,'placement-and-memory',sha)

def workflow(sha):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(10,3.8)); ax.axis('off'); ax.set(xlim=(0,1),ylim=(0,1))
    fig.subplots_adjust(left=.02,right=.98,bottom=.03,top=.98)
    def box(x,y,text,color='#edf2f8',w=.27):
        ax.add_patch(FancyBboxPatch((x-w/2,y-.095),w,.19,boxstyle='round,pad=.01,rounding_size=.02',facecolor=color,edgecolor='#b8cad6'))
        ax.text(x,y,text,ha='center',va='center',fontsize=11)
    def arrow(a,b): ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':'#526674','lw':1.5})
    box(.17,.82,'Repeat integrity, kernel\nand numerical checks')
    box(.52,.82,'CPU reference + 3\nalternating 16/32 GiB pairs')
    box(.84,.82,'Check quality, placement\nand memory headroom')
    arrow((.31,.82),(.375,.82)); arrow((.66,.82),(.695,.82))
    box(.52,.47,'At least 20% repeatable gain?\nNo material regression',color='#fff4d9',w=.42)
    arrow((.84,.71),(.73,.48))
    box(.21,.13,'Yes: update main README\nand publish a new release',color='#edf6f6',w=.36)
    box(.79,.13,'No: retain study evidence\nwithout claiming a speedup',color='#fbeded',w=.36)
    arrow((.37,.37),(.21,.24)); arrow((.67,.37),(.79,.24))
    save(fig,'study-workflow',sha)

def index_files():
    prefix=STUDY.name+'/'
    files=[p[len(prefix):] for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if p.startswith(prefix)]
    files=sorted(set(files)|{'INDEX.md'})
    descriptions={'README.md':'Study overview, findings, charts and new accelerated GIF.','RESULTS.md':'Generated per-profile and paired results and decision.',
        'METHODOLOGY.md':'Fixed controls, paired ordering, definitions and promotion rule.','REPRODUCE.md':'Commands to run checks, benchmarks, numeric probes and demo.',
        'LAB_NOTES.md':'Recorded actions, observations, failures and decisions.','INDEX.md':'Every published file in this tuning folder.',
        'configs/study.json':'Predeclared order, repetition counts and considerable-gain criterion.',
        'configs/profiles.json':'CPU, conservative 16 GiB, and conservative 32 GiB configurations.',
        'results/summary.json':'Curated attempts, per-request metrics, paired deltas and decision gates.',
        'results/numerical-diagnostics.json':'Repeated finite-logit and known-continuation probes on all three profiles.'}
    lines=['# Tunning-Enhance-1 file index','',f'All {len(files)} published study files. [Overview](README.md) · [Results](RESULTS.md) · [Parent repository](../README.md)','', '| File | Purpose |','|---|---|']
    for file in files:
        desc=descriptions.get(file)
        if not desc and file.endswith('.py'):
            desc=(ast.get_docstring(ast.parse((STUDY/file).read_text())) or 'Study tool.').splitlines()[0]
        if not desc: desc='Reproducible visualization.' if file.startswith('assets/') else 'Versioned experiment input or measured validation evidence.'
        lines.append(f'| [{file}]({file}) | {desc.replace("|","/")} |')
    lines+=['','Ignored local evidence: `results/raw/`, `runtime/`, and `logs/`. Weights and engine source are reused from the parent workspace.']
    (STUDY/'INDEX.md').write_text('\n'.join(lines)+'\n')

def main():
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'figure.facecolor':'white','axes.facecolor':'white',
        'text.color':'#172a3b','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'tunning-enhance-1'})
    source=STUDY/'results/summary.json'; summary=json.loads(source.read_text()); sha=hashlib.sha256(source.read_bytes()).hexdigest()
    runs=good_runs(summary)
    if not runs: raise SystemExit('No completed quality-checked observations to plot')
    performance(summary,runs,sha); paired(summary,sha); phases(runs,sha); resources(runs,sha); workflow(sha)
    manifest={'source':'results/summary.json','source_sha256':sha,'renderer':'scripts/render_results.py',
        'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'matplotlib_version':matplotlib.__version__,
        'figures':[n+ext for n in ('performance','paired-gains','phases-and-io','placement-and-memory','study-workflow') for ext in ('.png','.svg')]}
    (STUDY/'assets/figures.json').write_text(json.dumps(manifest,indent=2)+'\n')
    index_files()
    print('Rendered four tuning charts and study workflow; updated provenance and folder index.')

if __name__=='__main__': main()
