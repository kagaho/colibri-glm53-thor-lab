#!/usr/bin/env python3
"""Render this study's genuine 32 GiB terminal capture as an exact 30-second GIF."""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parent
spec=importlib.util.spec_from_file_location('baseline_demo_renderer',ROOT/'scripts/render_demo.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT=STUDY

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curate',action='store_true')
    args=parser.parse_args()
    state=None
    if args.curate:
        state,cast=module.curated_capture()
    else:
        cast=STUDY/'results/rust-fibonacci.cast'
    meta=module.render(cast,STUDY/'assets/rust-fibonacci-30s.gif')
    meta.update(renderer='../scripts/render_demo.py',wrapper='scripts/render_demo.py',
                wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    destination=STUDY/'results/rust-fibonacci-demo.json'
    if state:
        terminal=(STUDY/'runtime/rust-fibonacci-demo/terminal.txt').read_text()
        placement=re.search(r'tier glm53: on,.*?budget ([\d.]+) GiB = (\d+) experts',terminal)
        if not placement or float(placement[1])<31.98:
            raise RuntimeError('Demo did not show the requested 32 GiB CUDA expert tier')
        cfg=json.loads((STUDY/'configs/experiment.json').read_text())
        meta.update(prompt=state['prompt'],command=state['command'],started_utc=state['started_utc'],
            chat_statistics=state['chat_statistics'],containers_unchanged=state['containers_unchanged'],
            capture_script='../scripts/record_demo.py',capture_script_sha256=hashlib.sha256((ROOT/'scripts/record_demo.py').read_bytes()).hexdigest(),
            capture_wrapper='scripts/record_demo.py',profile='Conservative CUDA; 32 GiB GPU expert tier; 32 GB RAM expert cache; COLI_CUDA_TC_GEMM=0',
            observed_startup_budget_gib=float(placement[1]),observed_expert_capacity=int(placement[2]),
            engine_commit=cfg['engine_commit'],model_revision=cfg['model_revision'])
    elif destination.exists():
        meta=dict(json.loads(destination.read_text()),**meta)
    destination.write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))

if __name__=='__main__':
    main()
