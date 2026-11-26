#!/usr/bin/env python3
"""Wait for all paired runs, then repeat numeric checks and record/render the new demo."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parent

def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except OSError:
        return False

def main():
    def state(phase,**extra):
        data={'phase':phase,'pid':os.getpid(),'updated_utc':datetime.now(timezone.utc).isoformat(),**extra}
        (STUDY/'runtime/finalization-state.json').write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps(data),flush=True)
    state('waiting_for_paired_runs')
    try:
        while True:
            campaign=json.loads((STUDY/'runtime/campaign-state.json').read_text())
            if campaign['phase']=='failed':
                raise RuntimeError('Paired-run driver failed; finish did not replace its attempt')
            if campaign['phase']=='complete' and not alive(campaign['pid']):
                break
            time.sleep(10)
        commands=[('numerical_probes','check_numerics.py',[]),
                  ('recording_32gib_demo','record_demo.py',[]),
                  ('rendering_30s_gif','render_demo.py',['--curate']),
                  ('summarizing','summarize.py',[]),
                  ('rendering_charts','render_results.py',[])]
        for phase,script,args in commands:
            state(phase)
            with (STUDY/'logs'/f'{phase}.log').open('a') as output:
                subprocess.run([sys.executable,str(STUDY/'scripts'/script),*args],cwd=ROOT,
                    stdout=output,stderr=subprocess.STDOUT,check=True)
        state('complete')
    except BaseException as error:
        state('failed',error=f'{type(error).__name__}: {error}')
        raise

if __name__=='__main__': main()
