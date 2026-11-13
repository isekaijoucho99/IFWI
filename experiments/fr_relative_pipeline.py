"""Own one bounded preflight/timing/500-update queue, with no automatic retry."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time,traceback,uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def utc(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path,payload):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def validate_receipts(directory,stage,signature,returncode,launcher_pid):
    directory=Path(directory)
    launch=read(directory/'control/LAUNCH.json');end=read(directory/'control/EXIT.json');result=read(directory/'result.json')
    if (returncode!=0 or launch['run_id']!=end['run_id'] or launch['pid']!=end['pid']
        or launch['parent_pid']!=launcher_pid or end['status']!='COMPLETED' or end['exit_code']!=0
        or result.get('status')!='COMPLETED' or result.get('stage')!=stage
        or result.get('signature')!=signature or result['cost_totals']!=end['cost_totals']):
        raise ValueError('Child process/result/identity receipt mismatch: '+stage)
    return launch,end,result


def estimate(root):
    root=Path(root);timing=read(root/'timing/result.json');cases=timing['cases']
    medians={c['arm']:c['wall_seconds_median'] for c in cases}
    training=1000*sum(medians.values());warm=4*sum(medians.values())
    events=[json.loads(line) for line in (root/'preflight/ledger.jsonl').read_text().splitlines()]
    rows=[r for r in events if r.get('forward_fields_complete')==13]
    forward=4*rows[0]['wall_seconds'] if rows and rows[0].get('wall_seconds',0)>0 else None
    evaluation=None if forward is None else 61*forward
    lower=training+warm+(evaluation or 0.)
    report=dict(utc=utc(),median_update_seconds_by_arm=medians,training_seconds=training,warmup_seconds=warm,
        full52_forward_seconds_estimate=forward,evaluation_seconds_estimate=evaluation,
        total_hours_base=lower/3600,total_hours_with25percent_allowance=1.25*lower/3600,
        basis='six actual 10-update synchronized timing runs; evaluation extrapolated from actual13-field forward',
        limitations='changing velocity, desktop contention, construction, checkpoints and analysis may change wall time; this is not an equal-accuracy speedup or a completion guarantee')
    write_once(root/'WALL_ESTIMATE_FROM_TIMING.json',report)
    return report


def run_analysis(root,dense_data):
    """One CPU-only acceptance of already completed work; never retry failures."""
    command=[sys.executable,'-B','-m','experiments.fr_relative_analysis','--pipeline-root',str(root),
        '--dense-data',str(dense_data),'--source-root',str(ROOT)]
    with (root/'analysis.stdout.log').open('x',encoding='utf-8') as stdout,(root/'analysis.stderr.log').open('x',encoding='utf-8') as stderr:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        write_once(root/'ANALYSIS_PROCESS_LAUNCH.json',dict(launcher_pid=child.pid,command=command,utc=utc(),
            physical_F=0,physical_B=0,Adam_updates=0,automatic_restart=False))
        returncode=child.wait()
    write_once(root/'ANALYSIS_PROCESS_EXIT.json',dict(launcher_pid=child.pid,returncode=returncode,utc=utc()))
    return returncode


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','plan','dense-data','dataset-index','reference-config','initial-checkpoint'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--commit',required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    if not args.execute:
        print('PREPARED_NOT_RUN: preflight then timing then exactly500, no automatic1000');return 0
    from experiments.fr_relative_suite import verify_plan,release_guard,budget,assert_cost
    plan=read(args.plan);verify_plan(plan);release_guard()
    root=args.output.resolve(strict=True)
    for name in ('preflight','timing','formal','PIPELINE_LAUNCH.json','PIPELINE_EXIT.json'):
        if (root/name).exists(): raise FileExistsError('Existing task evidence; no duplicate: '+name)
    fd=os.open(root/'PIPELINE_OWNER.lock',os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
    identity=dict(run_id=args.run_id,pid=os.getpid(),parent_pid=os.getppid(),utc=utc(),
        commit=args.commit,code_directory=str(ROOT),automatic_restart=False,target_updates=500,
        pipeline_sha256=sha(__file__),plan_signature=plan['signature'])
    os.write(fd,json.dumps(identity).encode());os.fsync(fd)
    write_once(root/'PIPELINE_LAUNCH.json',identity)
    def event(kind,**details):
        with (root/'PIPELINE_EVENTS.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(run_id=args.run_id,utc=utc(),event=kind,**details),allow_nan=False)+'\n')
    def stopping():
        request_path=root/'STOP_REQUEST.json'
        if not request_path.exists(): return False
        request=read(request_path)
        if request.get('run_id')!=args.run_id or request.get('action')!='stop': raise ValueError('Invalid pipeline STOP_REQUEST')
        return True
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    common=['--plan',str(args.plan),'--dense-data',str(args.dense_data),'--dataset-index',str(args.dataset_index),
        '--reference-config',str(args.reference_config),'--initial-checkpoint',str(args.initial_checkpoint),'--execute']
    completed=[];active=None;child=None;terminal=None;tick=time.perf_counter()
    try:
        event('pipeline_started',identity=identity)
        for index,stage in enumerate(('preflight','timing','formal'),1):
            if stopping(): terminal=dict(status='STOPPED_BY_USER',exit_code=130,between_phases=True);break
            verify_plan(plan)
            directory=root/stage
            command=[sys.executable,'-B','-m','experiments.fr_relative_suite',*common,'--stage',stage,'--output',str(directory)]
            if stage in ('timing','formal'): command+=['--preflight',str(root/'preflight')]
            if stage=='formal': command+=['--timing',str(root/'timing'),'--target','500']
            started=time.perf_counter();forwarded=False;last_seen=None
            with (root/(stage+'.stdout.log')).open('x',encoding='utf-8') as stdout,(root/(stage+'.stderr.log')).open('x',encoding='utf-8') as stderr:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                active=dict(stage=stage,directory=stage,launcher_pid=child.pid,command=command,launched_utc=utc())
                write_once(root/f'phase{index}_launch.json',active);event('phase_launched',**active)
                while child.poll() is None:
                    launch_path=directory/'control/LAUNCH.json'
                    if launch_path.exists() and last_seen is None:
                        launch=read(launch_path)
                        if launch['parent_pid']!=child.pid: raise ValueError('Actual controller parent identity mismatch')
                        write_once(root/f'phase{index}_controller.json',launch);last_seen=launch
                    if stopping() and not forwarded and last_seen is not None:
                        # Create only our exact child's request. No PID lookup/kill.
                        write_once(directory/'STOP_REQUEST.json',dict(run_id=last_seen['run_id'],action='stop'))
                        forwarded=True;event('stop_forwarded',stage=stage,child_run_id=last_seen['run_id'])
                    time.sleep(1.)
                rc=child.returncode
            write_once(root/f'phase{index}_exit_handle.json',dict(stage=stage,launcher_pid=child.pid,returncode=rc,utc=utc()))
            if rc==130 and forwarded:
                end=read(directory/'control/EXIT.json')
                if end['run_id']!=last_seen['run_id'] or end['status']!='STOPPED_BY_USER': raise ValueError('Stop receipt mismatch')
                terminal=dict(status='STOPPED_BY_USER',exit_code=130,active_phase=active,receipt=end);break
            launch,end,result=validate_receipts(directory,stage,plan['signature'],rc,child.pid)
            assert_cost(result['cost_totals'],budget(stage))
            row=dict(stage=stage,directory=stage,receipt=end,phase_wall_seconds=time.perf_counter()-started,
                launcher_pid=child.pid,controller_pid=launch['pid'],result_sha256=sha(directory/'result.json'))
            write_once(root/f'phase{index}_verified.json',row);completed.append(row)
            event('phase_verified',stage=stage,cost_totals=end['cost_totals']);active=None;child=None
            print(json.dumps(dict(stage=stage,status='VERIFIED',cost_totals=end['cost_totals'])),flush=True)
            if stage=='timing': event('budget_estimated',estimate=estimate(root))
            if forwarded: terminal=dict(status='STOPPED_BY_USER',exit_code=130,stop_raced_completed_phase=stage);break
        if terminal is None: terminal=dict(status='COMPLETED',exit_code=0)
    except BaseException as error:
        terminal=dict(status='FAILED',exit_code=1,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            active_phase=active,child_exit_code=None if child is None else child.poll(),
            caution='No automatic retry. A child without a verified terminal receipt may still be running; check its evidence before another launch.')
    finally:
        terminal.update(run_id=args.run_id,pid=os.getpid(),parent_pid=os.getppid(),utc=utc(),completed_phases=completed,
            pipeline_wall_seconds=time.perf_counter()-tick,automatic_restart=False)
        write_once(root/'PIPELINE_EXIT.json',terminal);event('pipeline_terminal',status=terminal['status'],exit_code=terminal['exit_code'])
        if terminal['status']=='STOPPED_BY_USER': write_once(root/'STOP_ACK.json',terminal)
        os.close(fd)
    if terminal['status']=='COMPLETED':
        return run_analysis(root,args.dense_data)
    return terminal['exit_code']


if __name__=='__main__': raise SystemExit(main())
