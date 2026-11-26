#!/usr/bin/env python3
"""Repeat validation, one CPU reference, and three alternating 16/32 GiB pairs."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parent
sys.path.insert(0,str(ROOT/'scripts'))
from run_experiment import container_snapshot, write_json
from validate_model import validate


def main():
    os.chdir(ROOT)
    lease=(ROOT/'runtime/campaign.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    baseline=container_snapshot()
    plan=json.loads((STUDY/'configs/study.json').read_text())
    completed=[]
    existing=STUDY/'runtime/campaign-state.json'
    if existing.exists():
        previous=json.loads(existing.read_text())
        completed=previous.get('completed',[])
    child=None
    def state(phase,**extra):
        record={'phase':phase,'pid':os.getpid(),'study_dir':STUDY.name,
                'updated_utc':datetime.now(timezone.utc).isoformat(),'completed':completed,
                'planned_runs':len(plan['run_order']),**extra}
        write_json(STUDY/'runtime/campaign-state.json',record)
        write_json(ROOT/'runtime/campaign-state.json',record)
        print(json.dumps(record),flush=True)
    def interrupt(signum,frame):
        raise KeyboardInterrupt('Tuning campaign interrupted')
    signal.signal(signal.SIGTERM,interrupt)
    def run(command,log):
        nonlocal child
        with (STUDY/'logs'/log).open('a') as output:
            child=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            result=child.wait()
            child=None
        if result:
            raise RuntimeError(f'{log} failed; local evidence retained')
    try:
        state('preflight')
        write_json(STUDY/'results/model-validation.json',validate())
        run([sys.executable,'scripts/capture_environment.py','--output',str(STUDY/'results/environment.json')],'environment.log')
        run(['make','-C','vendor/colibri/c','cuda-glm53c-check','CUDA=1','CUDA_ARCH=sm_110'],'cuda-check.log')
        run([sys.executable,'scripts/record_build.py','--log',str(STUDY/'logs/cuda-check.log'),'--output-dir',str(STUDY/'results')],'build-record.log')
        run([sys.executable,'scripts/check_native_expert.py','--output',str(STUDY/'results/native-expert-check.json')],'native-expert.log')
        run([sys.executable,'-m','unittest','discover','-s','tests','-v'],'unit-tests.log')
        (STUDY/'results/unit-tests.txt').write_text((STUDY/'logs/unit-tests.log').read_text().replace(str(ROOT),'<REPO>'))
        for index,profile in enumerate(plan['run_order']):
            if index<len(completed):
                if completed[index]['profile']!=profile or completed[index]['status']!='complete':
                    raise RuntimeError('Resume state differs from planned completed prefix')
                continue
            state('benchmarking',index=index,profile=profile)
            run([sys.executable,'scripts/run_experiment.py','--study-dir',STUDY.name,
                 '--profile',profile,'--max-tokens',str(plan['max_completion_tokens']),
                 '--repeats',str(plan['http_repeats']),'--quality'],f'run-{index+1}-{profile}.log')
            latest=json.loads((ROOT/'runtime/last-run.json').read_text())
            path=Path(latest['run'])
            if not path.is_relative_to(STUDY/'results/raw'):
                raise RuntimeError('Runner wrote outside the isolated study')
            data=json.loads((path/'run.json').read_text())
            completed.append({'index':index,'profile':profile,'run_id':path.name,'status':data['status']})
            state('run_complete',index=index,profile=profile)
            summary=STUDY/'scripts/summarize.py'
            if summary.exists():
                run([sys.executable,str(summary)],'summarize.log')
        state('complete',containers_unchanged=baseline==container_snapshot())
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state('failed',error=f'{type(error).__name__}: {error}',containers_unchanged=baseline==container_snapshot())
        raise


if __name__=='__main__':
    main()
