"""ENV-151 deterministic capability-to-lock execution bootstrap."""
from __future__ import annotations
import argparse, hashlib, json, platform, re, subprocess, sys, venv
from pathlib import Path
from typing import Any
SCHEMA="12-6.execution-capabilities.v1"; MANIFEST_SCHEMA="12-6.execution-environment-manifest.v1"
_LOCK_LINE=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==([^\s;@/\\]+)(?: --hash=sha256:[0-9a-f]{64})+$")
_NAME=re.compile(r"[-_.]+")
class ExecutionBootstrapError(RuntimeError): pass
def _canonical_name(v:str)->str:return _NAME.sub("-",v.strip()).lower()
def _sha256_file(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _canonical_bytes(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
def _safe(root:Path,relative:str)->Path:
 p=Path(relative)
 if p.is_absolute() or ".." in p.parts: raise ExecutionBootstrapError(f"unsafe lock path: {relative}")
 q=(root/p).resolve()
 try:q.relative_to(root.resolve())
 except ValueError as e:raise ExecutionBootstrapError(f"lock path escapes repository: {relative}") from e
 return q
def _load_registry(root:Path)->dict[str,Any]:
 v=json.loads((root/"requirements/execution/capabilities.json").read_text(encoding="utf-8"))
 if v.get("schema")!=SCHEMA:raise ExecutionBootstrapError("execution capability registry schema mismatch")
 if platform.python_version()!=v["python"]["version"]:raise ExecutionBootstrapError(f"bootstrap requires CPython {v['python']['version']}, got {platform.python_version()}")
 if sys.implementation.name!=v["python"]["implementation"]:raise ExecutionBootstrapError("Python implementation mismatch")
 return v
def _validate_lock(root:Path,role:str,record:dict[str,Any])->dict[str,str]:
 path=_safe(root,record["path"])
 if not path.is_file():raise ExecutionBootstrapError(f"{role}: lock is missing: {path}")
 if _sha256_file(path)!=record["sha256"]:raise ExecutionBootstrapError(f"{role}: lock SHA-256 drift")
 packages={}
 for number,raw in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
  line=raw.strip()
  if not line or line.startswith("#"):continue
  if _LOCK_LINE.fullmatch(line) is None:raise ExecutionBootstrapError(f"{role}: non-exact/unhashed line {number}")
  name,tail=line.split("==",1); canonical=_canonical_name(name); version=tail.split()[0]
  if canonical in packages:raise ExecutionBootstrapError(f"{role}: duplicate package {canonical}")
  packages[canonical]=version
 if len(packages)!=record["package_count"]:raise ExecutionBootstrapError(f"{role}: package-count drift")
 return packages
def _command_capability(command:str,registry:dict[str,Any])->str|None:
 words=command.strip().split()
 if not words:return None
 if words[:3]==["python","-m","pytest"] or words[0]=="pytest":return "tests"
 if words[:3]==["python","-m","ruff"] or words[0]=="ruff":return "lint"
 return registry.get("dependency_executables",{}).get(Path(words[0]).name)
def resolve_plan(root:Path,capabilities:list[str],commands:list[str])->dict[str,Any]:
 registry=_load_registry(root); declared=[]
 for cap in capabilities:
  cap=cap.strip()
  if cap and cap not in declared:declared.append(cap)
 if not declared:raise ExecutionBootstrapError("at least one capability must be declared")
 cr=registry["capabilities"]
 for cap in declared:
  rec=cr.get(cap)
  if not isinstance(rec,dict):raise ExecutionBootstrapError(f"unknown capability: {cap}")
  if rec.get("status")!="available":raise ExecutionBootstrapError(f"{cap}: {rec.get('status','unavailable')} ({rec.get('reason','no exact lock')})")
  for req in rec.get("requires",[]):
   if req not in declared:raise ExecutionBootstrapError(f"{cap} requires declared capability {req}")
 for command in commands:
  needed=_command_capability(command,registry)
  if needed is not None and needed not in declared:raise ExecutionBootstrapError(f"command {command!r} requires undeclared capability {needed}")
 roles=["toolchain"]
 if "cuda" in declared:
  if "runtime" not in declared:raise ExecutionBootstrapError("cuda requires runtime")
  roles.append("cuda_runtime")
 elif "runtime" in declared or "distributed" in declared:roles.append("cpu_runtime")
 if "tokenizer" in declared:
  if "runtime" not in declared:roles.append("tokenizer_support")
  roles.append("tokenizer_overlay")
 if "transformers" in declared:
  if "runtime" not in declared:roles.append("transformers_support")
  roles.append("transformers_overlay")
 if "tests" in declared or "lint" in declared:roles.append("dev")
 seen=set(); roles=[r for r in roles if not (r in seen or seen.add(r))]; merged={}; locks=[]
 for role in roles:
  rec=registry["locks"][role]; packages=_validate_lock(root,role,rec)
  for name,version in packages.items():
   if name in merged and merged[name]!=version:raise ExecutionBootstrapError(f"lock conflict for {name}: {merged[name]} versus {version}")
   merged[name]=version
  locks.append({"role":role,"path":rec["path"],"sha256":rec["sha256"],"package_count":rec["package_count"]})
 forbidden=[n for n in merged if n.startswith("nvidia-") or n.startswith("cuda-") or n=="triton"]
 if "cuda" not in declared and forbidden:raise ExecutionBootstrapError("non-CUDA plan inherited CUDA packages: "+", ".join(sorted(forbidden)))
 imports=[]; executables=[]
 for cap in declared:imports+=cr[cap].get("imports",[]); executables+=cr[cap].get("executables",[])
 p={"schema":"12-6.execution-plan.v1","capabilities":declared,"commands":commands,"locks":locks,"imports":sorted(set(imports)),"executables":sorted(set(executables)),"package_count":len(merged),"cuda_packages_present":bool(forbidden),"d08_authority":registry["d08_authority"]}
 p["identity_sha256"]=hashlib.sha256(_canonical_bytes(p)).hexdigest();return p
def _venv_python(d:Path)->Path:return d/("Scripts/python.exe" if platform.system()=="Windows" else "bin/python")
def _run(c:list[str|Path],cwd:Path)->None:subprocess.run([str(x) for x in c],cwd=cwd,check=True)
def _probe_imports(python:Path,modules:list[str],cwd:Path)->None:
 if not modules:return
 code="import importlib\nmods="+repr(modules)+"\nmissing=[]\nfor n in mods:\n try: importlib.import_module(n)\n except Exception as e: missing.append((n,type(e).__name__,str(e)))\nassert not missing, 'missing imports: '+repr(missing)\n"
 _run([python,"-c",code],cwd)
def _probe_executables(python:Path,executables:list[str])->None:
 missing=[]
 for name in executables:
  candidates=[python.parent/name]
  if platform.system()=="Windows":candidates.append(python.parent/f"{name}.exe")
  if not any(p.is_file() for p in candidates):missing.append(name)
 if missing:raise ExecutionBootstrapError("missing declared executables in bootstrap venv: "+", ".join(sorted(missing)))
def _installed(python:Path,cwd:Path)->dict[str,str]:
 code="import importlib.metadata as m,json; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()},sort_keys=True))"
 return json.loads(subprocess.run([str(python),"-c",code],cwd=cwd,check=True,capture_output=True,text=True).stdout)
def preflight(root:Path,python:Path,plan:dict[str,Any],allow_no_gpu:bool)->dict[str,Any]:
 _probe_imports(python,plan["imports"],root);_probe_executables(python,plan["executables"])
 cuda={"software_capability_declared":"cuda" in plan["capabilities"],"hardware_visible":False,"hardware_claim":False,"no_gpu_preflight":False}
 if "cuda" in plan["capabilities"]:
  visible=subprocess.run([str(python),"-c","import torch; print(int(torch.cuda.is_available()))"],cwd=root,check=True,capture_output=True,text=True).stdout.strip()=="1"
  cuda["hardware_visible"]=visible;cuda["hardware_claim"]=visible
  if not visible:
   if not allow_no_gpu:raise ExecutionBootstrapError("CUDA software is installed but no CUDA hardware is visible")
   cuda["no_gpu_preflight"]=True
 return {"status":"PASS","cuda":cuda}
def bootstrap(root:Path,venv_dir:Path,capabilities:list[str],commands:list[str],manifest_path:Path,allow_no_gpu:bool)->dict[str,Any]:
 plan=resolve_plan(root,capabilities,commands)
 if venv_dir.exists():raise ExecutionBootstrapError(f"refusing non-fresh venv: {venv_dir}")
 venv.EnvBuilder(with_pip=True).create(venv_dir);python=_venv_python(venv_dir)
 for lock in plan["locks"]:_run([python,"-m","pip","install","--disable-pip-version-check","--require-hashes","--no-deps","-r",_safe(root,lock["path"])],root)
 proof=preflight(root,python,plan,allow_no_gpu);m={"schema":MANIFEST_SCHEMA,"plan":plan,"python":{"implementation":sys.implementation.name,"bootstrap_version":platform.python_version(),"venv_executable":str(python)},"platform":{"system":platform.system(),"machine":platform.machine()},"packages":_installed(python,root),"preflight":proof}
 m["identity_sha256"]=hashlib.sha256(_canonical_bytes(m)).hexdigest();manifest_path.parent.mkdir(parents=True,exist_ok=True);manifest_path.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n",encoding="utf-8");return m
def _csv(v:str)->list[str]:return [x.strip() for x in v.split(",") if x.strip()]
def main(argv:list[str]|None=None)->int:
 p=argparse.ArgumentParser();p.add_argument("action",choices=("plan","bootstrap","preflight"));p.add_argument("--repo-root",type=Path,default=Path("."));p.add_argument("--capabilities",required=True);p.add_argument("--command",action="append",default=[]);p.add_argument("--venv",type=Path);p.add_argument("--manifest",type=Path);p.add_argument("--allow-no-gpu",action="store_true");a=p.parse_args(argv);root=a.repo_root.resolve();caps=_csv(a.capabilities)
 if a.action=="plan":print(json.dumps(resolve_plan(root,caps,a.command),indent=2,sort_keys=True));return 0
 if a.action=="bootstrap":
  if a.venv is None or a.manifest is None:p.error("bootstrap requires --venv and --manifest")
  m=bootstrap(root,a.venv.resolve(),caps,a.command,a.manifest.resolve(),a.allow_no_gpu);print(json.dumps({"status":"PASS","identity_sha256":m["identity_sha256"],"capabilities":m["plan"]["capabilities"],"cuda":m["preflight"]["cuda"]},sort_keys=True));return 0
 if a.venv is None:p.error("preflight requires --venv")
 print(json.dumps(preflight(root,_venv_python(a.venv.resolve()),resolve_plan(root,caps,a.command),a.allow_no_gpu),indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
