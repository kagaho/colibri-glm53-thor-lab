#!/usr/bin/env python3
"""Repeat CPU/16/32 GiB full-model logit-finiteness and known-continuation probes."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parent
sys.path.insert(0,str(ROOT/'scripts'))
from run_experiment import container_snapshot,meminfo,owned_stop,write_json
from summarize import parse_device_tier

PROMPTS={'short':('The capital of France is','Paris'),
    'long':('The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is','Lisbon')}

def main():
    import numpy as np
    os.chdir(ROOT)
    lease=(ROOT/'runtime/campaign.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    baseline=container_snapshot()
    cfg=json.loads((STUDY/'configs/experiment.json').read_text())
    profiles=json.loads((STUDY/'configs/profiles.json').read_text())
    raw=STUDY/'results/raw'/('numerical-probes-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    raw.mkdir(parents=True)
    write_json(STUDY/'results/numerical-diagnostics.json',{'status':'running','cases':[]})
    cases=[]
    differences=[]
    success=False
    def interrupt(signum,frame):
        raise KeyboardInterrupt('Numerical probes interrupted')
    signal.signal(signal.SIGTERM,interrupt)
    try:
        for length,(prompt,expected) in PROMPTS.items():
            reference=None
            for name,profile in profiles.items():
                state={'phase':'numerical_probes','pid':os.getpid(),'study_dir':STUDY.name,
                    'case':length+'-'+name,'updated_utc':datetime.now(timezone.utc).isoformat()}
                write_json(ROOT/'runtime/campaign-state.json',state)
                write_json(STUDY/'runtime/numeric-state.json',state)
                if meminfo()['MemAvailable']<80*2**30:
                    raise RuntimeError('Insufficient available RAM before the isolated numeric probe')
                stem=raw/(length+'-'+name)
                env={k:v for k,v in os.environ.items() if not k.startswith(('COLI_','GLM53_','G53_','CUDA_','OMP_')) and k not in ('SERVE','DUMP')}
                env.update(COLI_CUDA='0' if profile['gpu']=='none' else '1',COLI_VULKAN='0',
                    COLI_CUDA_TC_GEMM='0',COLI_VK_TIER_GB=str(profile['gpu_tier_gib']),COLI_VK_TIER_HOST='0',
                    GLM53_BITS='4',GLM53_EXPERT_GB='32',GLM53_MAXT='8192',G53_DIRECT='0',
                    OMP_NUM_THREADS='14',USAGE_SAVE='0',COLI_PREFIX_STORE='0',COLI_USAGE=str(stem)+'.usage',DUMP=str(stem)+'.f32')
                command=[str(ROOT/'vendor/colibri/c/glm53'),'--model',str(ROOT/cfg['model_directory']),
                    '--prompt',prompt,'--greedy','1']
                with Path(str(stem)+'.log').open('w') as log, Path(str(stem)+'.stdout').open('w') as output:
                    proc=subprocess.Popen(command,env=env,cwd=ROOT,stdout=output,stderr=log,start_new_session=True)
                    started=time.monotonic()
                    low=0
                    try:
                        while proc.poll() is None:
                            low=low+1 if meminfo()['MemAvailable']<8*2**30 else 0
                            if low>=3 or time.monotonic()-started>900:
                                raise RuntimeError('Numeric probe memory guard or timeout')
                            time.sleep(1)
                        if proc.returncode:
                            raise RuntimeError('Numeric probe process failed')
                    finally:
                        owned_stop(proc)
                values=np.fromfile(str(stem)+'.f32',dtype='<f4')
                stdout=Path(str(stem)+'.stdout').read_text(errors='replace')
                text=stdout+'\n'+Path(str(stem)+'.log').read_text(errors='replace')
                after=stdout.split('teacher_forcing',1)[1].split('\n',1)[1]
                continuation=after.split('[CUDA]',1)[0].split('\ndecode',1)[0].strip()
                nonfinite=int((~np.isfinite(values)).sum())
                tier=parse_device_tier(text,profile['gpu_tier_gib'])
                passed=bool(values.size and not nonfinite and continuation==expected and
                    '[router] non-finite logits' not in text and
                    (profile['gpu']=='none' or (tier and not tier['clamped_below_requested'])))
                cases.append({'prompt_kind':length,'prompt':prompt,'expected':expected,'profile':name,
                    'logit_values':int(values.size),'nonfinite_logits':nonfinite,'continuation':continuation,
                    'observed_device_expert_tier':tier,'passed':passed})
                if reference is None:
                    reference=values
                elif values.shape==reference.shape:
                    differences.append({'prompt_kind':length,'profile':name,
                        'cpu_cuda_max_absolute_logit_difference':float(np.max(np.abs(reference-values)))})
                result={'status':'passed' if len(cases)==6 and all(c['passed'] for c in cases) else 'partial',
                    'cases':cases,'differences':differences,'engine_commit':cfg['engine_commit'],
                    'model_revision':cfg['model_revision'],
                    'scope':'Two raw prompts and one greedy continuation per profile; finite logits plus known continuation, not full-model numeric equivalence',
                    'historical_negative_control':'Default tensor-core CUDA previously failed; not enabled in this tuning study'}
                write_json(STUDY/'results/numerical-diagnostics.json',result)
                if not passed:
                    raise RuntimeError('A numeric continuation, finiteness, or placement check failed')
        result['containers_unchanged']=baseline==container_snapshot()
        if not result['containers_unchanged']:
            result['status']='failed'
        write_json(STUDY/'results/numerical-diagnostics.json',result)
        success=result['status']=='passed'
        print(json.dumps(result),flush=True)
    finally:
        write_json(ROOT/'runtime/campaign-state.json',{'phase':'complete' if success else 'failed','pid':os.getpid(),'study_dir':STUDY.name})

if __name__=='__main__':
    main()
