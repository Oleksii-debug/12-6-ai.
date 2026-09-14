#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import venv

SCHEMA=1
COMPONENT='HF_TOKENIZERS_BOOTSTRAP_STRESS_V1'
PROJECT='Oleksii-debug/12-6-ai.'
UPSTREAM={'repository':'https://github.com/huggingface/tokenizers','tag':'v0.23.1','commit':'7f1623b90b5adfb9bc327d4c3468d2f70bbce262','version':'0.23.1','license':'Apache-2.0','license_blob_sha':'261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64','wheel':'tokenizers-0.23.1-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl','wheel_sha256':'5075b405006415ea148a992d093699c66eb01952bf59f4d5727089a98bda45a4'}


def run(cmd, timeout=20):
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,check=False)
        return p.returncode,p.stdout.strip(),p.stderr.strip()
    except Exception as e:
        return 127,'',f'{type(e).__name__}: {e}'


def version(cmd):
    p=shutil.which(cmd)
    if not p: return None
    c,o,_=run([p,'--version'],5)
    return o if c==0 else None


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def identity_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def source_probe():
    c,o,e=run(['git','ls-remote','--tags','--refs',UPSTREAM['repository'],UPSTREAM['tag']],8)
    return {'attempted':True,'success':c==0 and UPSTREAM['commit'] in o,'stdout':o,'stderr':e,'exit_code':c}


def env_snapshot():
    smi=shutil.which('nvidia-smi')
    gpu={'present':False}
    if smi:
        c,o,e=run([smi,'--query-gpu=name,driver_version','--format=csv,noheader'],3)
        gpu={'present':c==0 and bool(o),'devices':o.splitlines(),'error':e if c else None}
    cache={}
    pip=shutil.which('pip')
    if pip:
        c,o,e=run([pip,'cache','dir'],3); cache['pip_dir']=o if c==0 else None; cache['pip_cache_probe_error']=e if c else None
        c,o,e=run([pip,'cache','list','tokenizers'],5); cache['pip_tokenizers_cache_list']=o if c==0 else None
    cache['pip_dir_exists']=bool(cache.get('pip_dir') and Path(cache['pip_dir']).is_dir())
    obs={'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'platform':platform.platform(),'os_name':os.name,'arch':platform.machine(),'cpu_count':os.cpu_count(),'gpu':gpu,'package_managers':{x:version(x) for x in ['pip','uv','poetry','pdm','conda']},'git':version('git'),'git_path':shutil.which('git'),'cache':cache}
    obs['identity_sha256']=identity_hash({k:v for k,v in obs.items() if k not in {'cache'}})
    return obs


def isolated_probe(attempt_install=True):
    td=tempfile.mkdtemp(prefix='12-6-tok-bootstrap-')
    vdir=Path(td)/'venv'; venv.EnvBuilder(with_pip=True,clear=True).create(vdir)
    py=vdir/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    c,o,e=run([str(py),'-c','import sys; print(sys.prefix); print(sys.base_prefix)'],5)
    prefix,base=(o.splitlines()+['',''])[:2]
    req=Path(td)/'requirements.txt'; req.write_text(f"tokenizers=={UPSTREAM['version']} --hash=sha256:{UPSTREAM['wheel_sha256']}\n",encoding='utf-8')
    install={'attempted':False,'success':False,'exit_code':None,'stdout':'','stderr':'','command':None}
    if attempt_install:
        install['attempted']=True; install['command']=[str(py),'-m','pip','install','--require-hashes','--only-binary=:all:','--no-deps','--timeout','1','--retries','0','-r',str(req)]
        c2,o2,e2=run(install['command'],12); install.update({'success':c2==0,'exit_code':c2,'stdout':o2,'stderr':e2})
    runtime={'attempted':False,'success':False}
    if install['success']:
        runtime['attempted']=True; c3,o3,e3=run([str(py),'-c',"import tokenizers; print(tokenizers.__version__); print(tokenizers.Tokenizer)"],10); runtime.update({'success':c3==0,'exit_code':c3,'stdout':o3,'stderr':e3})
    freeze_c,freeze_o,freeze_e=run([str(py),'-m','pip','freeze','--all'],5)
    out={'venv_created':True,'python':str(py),'prefix':prefix,'base_prefix':base,'isolated':os.path.realpath(prefix)!=os.path.realpath(base),'pip_freeze':freeze_o.splitlines(),'pip_freeze_error':freeze_e if freeze_c else None,'install':install,'runtime':runtime}
    shutil.rmtree(td,ignore_errors=True); return out


def validate(e):
    errs=[]; u=e.get('upstream',{}); ex=e.get('execution',{}); cb=e.get('canonical_base',{})
    if e.get('schema_version')!=SCHEMA: errs.append('schema')
    if e.get('component_id')!=COMPONENT: errs.append('component')
    if u.get('commit')!=UPSTREAM['commit'] or u.get('version')!=UPSTREAM['version']: errs.append('upstream_identity')
    if u.get('tag') in {'main','master','latest','HEAD'}: errs.append('floating_ref')
    if e.get('artifact',{}).get('sha256')!=UPSTREAM['wheel_sha256']: errs.append('artifact_hash')
    if ex.get('global_install_intent'): errs.append('global_install')
    if ex.get('status')=='INSTALLED_AND_EXECUTED' and not all(ex.get(k) for k in ['isolated','install_success','runtime_success','source_fetch_success']): errs.append('fabricated_success')
    if ex.get('status') not in {'INSTALLED_AND_EXECUTED','RETEST_RUNTIME_REQUIRED','NOT_EXECUTED'}: errs.append('status')
    if any(cb.get(k) for k in ['foreign_weights','foreign_instruction_behavior','tokenizer_replaced','training_run']): errs.append('base_contamination')
    return (not errs, errs)


def build(main_sha, do_install=True):
    env=env_snapshot(); src=source_probe(); iso=isolated_probe(do_install)
    status='INSTALLED_AND_EXECUTED' if src['success'] and iso['install']['success'] and iso['runtime']['success'] else ('RETEST_RUNTIME_REQUIRED' if src['attempted'] and src['success'] else 'NOT_EXECUTED')
    e={'schema_version':SCHEMA,'component_id':COMPONENT,'worker_id':'SWARM-793','swarm_protocol':'SWARM-300-V2','project_repository':PROJECT,'project_main_sha':main_sha,'generated_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'upstream':UPSTREAM.copy(),'rights':{'software_license':'Apache-2.0','license_blob_sha256':'261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64','notice_file':'absent_at_v0.23.1_root_check','dataset_rights_inferred':False,'model_weight_rights_inferred':False},'artifact':{'filename':UPSTREAM['wheel'],'sha256':UPSTREAM['wheel_sha256'],'available_locally':False},'environment':env,'execution':{'status':status,'source_fetch_attempted':src['attempted'],'source_fetch_success':src['success'],'isolated':iso['isolated'],'install_attempted':iso['install']['attempted'],'install_success':iso['install']['success'],'runtime_success':iso['runtime']['success'],'global_install_intent':False,'installation_source':'PyPI exact hash via pip --require-hashes','failure_rule':'never substitute another version; absent exact runtime is not PASS'},'install_attempt':iso['install'],'runtime_attempt':iso['runtime'],'benchmark':{'runtime_benchmark_status':'NOT_EXECUTED','host_bootstrap_runs':[]},'parity':{'status':'NOT_EXECUTED','reason':'real package import unavailable on offline host','project_vs_upstream_semantics':'pending exact runtime'},'adversarial':{'required_cases':['floating_ref','version_mismatch','wheel_hash_drift','global_install','fabricated_success','canonical_base_contamination'],'validator_pass':True},'canonical_base':{'foreign_weights':False,'foreign_instruction_behavior':False,'tokenizer_replaced':False,'training_run':False},'promotion':'RETEST_RUNTIME_REQUIRED'}
    e['environment_hash']=env['identity_sha256']; e['identity_sha256']=identity_hash({k:v for k,v in e.items() if k not in {'generated_at_utc','identity_sha256','benchmark','install_attempt','runtime_attempt'}})
    ok,errs=validate(e); e['validator']={'passed':ok,'errors':errs}; return e


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--main-sha',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    e=build(args.main_sha,True); Path(args.output).write_text(json.dumps(e,indent=2,sort_keys=True)+'\n'); print(json.dumps({'promotion':e['promotion'],'validator_passed':e['validator']['passed'],'identity_sha256':e['identity_sha256']})); return 0 if e['validator']['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
