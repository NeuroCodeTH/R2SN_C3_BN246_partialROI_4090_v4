#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4090/Linux patient-level concurrent BN246 partial-ROI runner (no Slurm)."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

DEFAULT_ROOT='/data02/tanghui/C3_R2SN'
DEFAULT_TAG='lesionaware_BN246_remainingROI_v4'


def read_manifest(path):
    with open(path,newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    required={'id','code','t1','lesion'}
    if rows and not required.issubset(rows[0]):
        raise ValueError('manifest missing '+', '.join(sorted(required-set(rows[0]))))
    for row in rows:
        if not all(row.get(k) for k in required):
            raise ValueError('manifest contains empty ID/T1/lesion field')
    return rows


def command(args,row):
    c=[sys.executable,'-u',str(Path(__file__).with_name('run_one.py')),
       '--id',row['id'],'--code',row['code'],'--t1',row['t1'],
       '--lesion',row['lesion'],'--out-root',args.out_root,
       '--nroi','246','--tag',args.tag,
       '--margin-mm',str(args.margin_mm),'--min-voxels',str(args.min_voxels),
       '--registration-mask',args.registration_mask]
    for k in ('atlas','target','params','picked','tmp_root'):
        value=getattr(args,k)
        if value:c += ['--'+k.replace('_','-'),str(value)]
    if args.keep_qc_images:c.append('--keep-qc-images')
    if args.force:c.append('--force')
    if args.reuse_registered_from:c += ['--reuse-registered-from',args.reuse_registered_from]
    return c


def prior_status(args,row):
    base=Path(args.out_root)/row['id']/args.tag
    file=base/'qc'/'run_summary.json'
    if not file.is_file() or args.force:return 'RUN',''
    try:
        m=json.loads(file.read_text(encoding='utf-8'))
    except Exception:return 'RUN',''
    if m.get('status') not in ('SUCCESS','SUCCESS_WITH_MISSING_NODES'):
        return 'RUN',''
    expected=dict(version='BN246_remaining_roi_zero_v4.1',id=row['id'],
                  t1=row['t1'],lesion=row['lesion'],nroi=246,tag=args.tag,
                  registration_mask=args.registration_mask,
                  margin_mm=args.margin_mm,min_voxels=args.min_voxels)
    mismatch=[k for k,v in expected.items() if str(m.get(k))!=str(v)]
    if mismatch:return 'PARAM_MISMATCH',','.join(mismatch)
    outputs=[base/'network'/f'{row["id"]}_network.npy',
             base/'network'/f'{row["id"]}_valid_roi_mask.npy',
             base/'radiomics'/'ROI_features_246.csv',
             base/'radiomics'/'ROI_node_metrics.csv']
    if all(p.is_file() and p.stat().st_size>0 for p in outputs):
        return 'SKIP_SUCCESS','Outputs exist; to recompute explicitly use --force'
    return 'RUN','incomplete previous output'


def run_subject(args,row,log_dir,lock_dir):
    sid=row['id']
    logpath=log_dir/f'{sid}_{args.tag}.log'
    lockpath=lock_dir/f'{sid}_{args.tag}.lock'
    start=time.monotonic()
    with lockpath.open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(id=sid,code=row['code'],state='LOCKED',returncode='',seconds=0,
                        log=str(logpath),detail='same ID/tag currently running')
        decision,detail=prior_status(args,row)
        if decision!='RUN':
            return dict(id=sid,code=row['code'],state=decision,returncode='',seconds=0,
                        log=str(logpath),detail=detail)
        env=os.environ.copy()
        for k in ('OMP_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','OPENBLAS_NUM_THREADS',
                  'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
            env[k]=str(args.threads_per_worker)
        cmd=command(args,row)
        with logpath.open('a',encoding='utf-8') as log:
            log.write(f'\n===== START {time.strftime("%F %T")} =====\n')
            log.write(' '.join(cmd)+'\n');log.flush()
            try:
                rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,
                                  check=False).returncode
            except Exception as e:
                rc=-1;log.write(f'Launcher failed: {type(e).__name__} {e}\n')
        state={0:'SUCCESS',2:'SUCCESS_WITH_MISSING_NODES',1:'FAILED'}.get(rc,'EXIT_'+str(rc))
        sfile=Path(args.out_root)/sid/args.tag/'qc'/'run_summary.json'
        if sfile.is_file():
            try:
                m=json.loads(sfile.read_text(encoding='utf-8'))
                state=m.get('status',state)
                detail=m.get('error','') or f'valid={m.get("n_valid", "?")},zero={m.get("n_zero", "?")},missing={m.get("n_other_missing", "?")}'
            except Exception:pass
        return dict(id=sid,code=row['code'],state=state,returncode=rc,
                    seconds=round(time.monotonic()-start,1),log=str(logpath),detail=detail)


def write_progress(path,rows):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=['id','code','state','returncode','seconds','log','detail'])
        wr.writeheader();wr.writerows(rows)
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='/data02/tanghui/C3_R2SN/scripts/C3_MNI_manifest.csv')
    p.add_argument('--out-root',default=DEFAULT_ROOT)
    p.add_argument('--tag',default=DEFAULT_TAG)
    p.add_argument('--nroi',type=int,choices=[246],default=246)
    p.add_argument('--margin-mm',type=float,default=0.0)
    p.add_argument('--min-voxels',type=int,default=200)
    p.add_argument('--registration-mask',choices=['fixed','none'],default='fixed')
    for k in ('atlas','target','params','picked','tmp_root'):
        p.add_argument('--'+k.replace('_','-'))
    p.add_argument('--keep-qc-images',action='store_true')
    p.add_argument('--reuse-registered-from',default='')
    p.add_argument('--force',action='store_true')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--id',default='')
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--threads-per-worker',type=int,default=2)
    p.add_argument('--progress-file',default='')
    args=p.parse_args()
    if args.workers<1 or args.threads_per_worker<1 or args.limit<0 or args.start<0:
        p.error('workers, threads must be >=1 and start/limit must be >=0')
    maxcpu=os.cpu_count() or 1
    if args.workers*args.threads_per_worker>maxcpu:
        print('WARNING: worker*threads > reported CPU count',file=sys.stderr)
    subjects=read_manifest(args.manifest)
    if args.id:
        subjects=[r for r in subjects if r['id']==args.id]
    else:
        subjects=subjects[args.start:]
        if args.limit:subjects=subjects[:args.limit]
    if not subjects:raise SystemExit('No subjects matching selection')
    if len(set(r['id'] for r in subjects))!=len(subjects):
        raise SystemExit('Duplicate IDs in manifest selection')
    root=Path(args.out_root);root.mkdir(parents=True,exist_ok=True)
    logs=root/'lesionaware_logs';locks=root/'.r2sn_v4_locks'
    logs.mkdir(exist_ok=True);locks.mkdir(exist_ok=True)
    prog=Path(args.progress_file) if args.progress_file else root/f'{args.tag}_parallel_progress.csv'
    total=len(subjects);ctr=Counter();out=[];start=time.monotonic()
    print(f'BN246 remaining ROI: patients={total}, workers={args.workers}, threads/worker={args.threads_per_worker}',flush=True)
    itr=iter(subjects)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={}
        for _ in range(min(args.workers,total)):
            row=next(itr,None)
            if row:pending[pool.submit(run_subject,args,row,logs,locks)]=row['id']
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                sid=pending.pop(future)
                try:row=future.result()
                except Exception as e:
                    row=dict(id=sid,code='',state='LAUNCHER_FAILED',returncode='',seconds=0,
                             log='',detail=f'{type(e).__name__} {e}')
                out.append(row);ctr[row['state']]+=1
                print(f'[{len(out)}/{total}] {sid}: {row["state"]} | done={len(out)}, remaining={total-len(out)} | elapsed={int(time.monotonic()-start)}s | {row["detail"]}',flush=True)
                write_progress(prog,out)
                row=next(itr,None)
                if row:pending[pool.submit(run_subject,args,row,logs,locks)]=row['id']
    print('COUNTS',dict(ctr),flush=True)
    print('PROGRESS_CSV',prog,flush=True)

if __name__=='__main__':main()
