"""The body of every kgr kernel. Stdlib only -- it runs on Kaggle's image.

`kgr push` prepends a KGR_CONFIG dict to this file and uploads the result as
a script kernel. On Kaggle it:

  1. reports the GPU and refuses to burn the run on a CPU or on a card the
     installed torch cannot drive,
  2. finds THIS push's code bundle among the attached inputs (by build id),
  3. unpacks it outside /kaggle/working so the code does not become output,
     and links the job's output directory into /kaggle/working,
  4. copies files from an attached previous kernel's output into that
     directory, so the job can resume,
  5. runs the configured command, streaming its output.

Before anything else it prints one `KGR_SESSION_INFO {json}` line (and writes
kgr_session.json): what this kernel knows about itself, which `kgr info` and
`kgr cancel` use to find the session. On SIGTERM it forwards the signal to the
job (best effort: a real `kgr cancel` was observed to kill the session
without delivering it, so checkpoint to disk periodically).

For a local dry run, point KGR_INPUT_ROOT / KGR_WORKING_ROOT / KGR_BUNDLE_DIR
at scratch directories and pass the config as a JSON file argument.
"""
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

BUNDLE_TAR = "kgr_bundle.tar.gz"
BUNDLE_MARKER = "kgr_bundle_root.json"
SEARCH_DEPTH = 4

# `kgr info` / `kgr cancel` read this line back out of the kernel log.
SESSION_MARK = "KGR_SESSION_INFO"
SESSION_FILE = "kgr_session.json"
SECRET_WORDS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "COOKIE",
                "AUTH", "KEY", "PRIVATE")
ID_NAME_HINTS = ("SESSION", "KERNEL", "RUN", "CONTAINER", "JOB", "POD",
                 "INSTANCE", "NOTEBOOK", "VERSION")
GRACE_SECONDS = 25   # after SIGTERM the job gets this long to save and exit


def log(*args, **kwargs):
    print(*args, flush=True, **kwargs)


def extract(tar_path, dest):
    with tarfile.open(tar_path) as tf:
        # Python 3.12+ warns (and 3.14 changes behaviour) without an explicit
        # filter; the bundle is our own upload, "data" keeps it portable.
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, filter="data")
        else:
            tf.extractall(dest)


def read_text(path):
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


def session_info(cfg):
    """Everything this kernel can say about itself.

    The Kaggle CLI has no call that returns the id of a running session, but
    the API can cancel one by id -- so the kernel reports whatever identifies
    it (KAGGLE_* environment, hostname, container id) and `kgr` reads it back
    from the log. Secrets are redacted: the log is also what the owner sees.
    """
    env, candidates = {}, {}
    for k, v in sorted(os.environ.items()):
        up = k.upper()
        if not up.startswith(("KAGGLE", "KGR_")):
            continue
        env[k] = "<redacted>" if any(w in up for w in SECRET_WORDS) else v[:300]
        if v.strip().isdigit() and any(h in up for h in ID_NAME_HINTS):
            candidates[k] = int(v)
    containers = set()
    for src in ("/proc/self/cgroup", "/proc/self/mountinfo", "/etc/hostname"):
        containers.update(re.findall(r"\b[0-9a-f]{64}\b", read_text(src)))
    return {
        "run": cfg["run"], "project": cfg["project"], "build_id": cfg["build_id"],
        "hostname": socket.gethostname(), "pid": os.getpid(),
        "containers": sorted(containers), "id_candidates": candidates,
        "env": env,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
    }


def announce_session(cfg, working):
    info = session_info(cfg)
    try:
        working.mkdir(parents=True, exist_ok=True)
        (working / SESSION_FILE).write_text(json.dumps(info, indent=2))
    except OSError:
        pass
    log(f"{SESSION_MARK} {json.dumps(info, separators=(',', ':'))}")


def gpu_report():
    """(device, capability) without initialising CUDA in this process --
    a context here would hold GPU memory for the whole run."""
    try:
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                        "--format=csv,noheader"], check=False)
    except FileNotFoundError:
        # Absent when no accelerator is attached; must never kill the run.
        log("no nvidia-smi on this image")
    probe = ("import json, torch; c = torch.cuda.is_available(); "
             "print(json.dumps({'torch': torch.__version__, 'cuda': c, "
             "'name': torch.cuda.get_device_name(0) if c else None, "
             "'cap': list(torch.cuda.get_device_capability(0)) if c else None}))")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if r.returncode != 0:
        has_smi = shutil.which("nvidia-smi") is not None
        log("torch not importable; judging the device by nvidia-smi alone")
        return ("cuda" if has_smi else "cpu"), None
    info = json.loads(r.stdout.strip().splitlines()[-1])
    log(f"torch {info['torch']}  cuda={info['cuda']}  {info['name'] or 'CPU ONLY'}")
    return ("cuda" if info["cuda"] else "cpu"), (tuple(info["cap"]) if info["cap"] else None)


def dump_inputs(input_root):
    # Kaggle has mounted the same dataset at different paths across otherwise
    # identical pushes, so always print the real layout.
    log(f"--- {input_root} ---")
    for p in sorted(input_root.glob("*")):
        kids = sorted(x.name for x in p.iterdir())[:8] if p.is_dir() else []
        log(f"  {p}  {'-> ' + ', '.join(kids) if kids else ''}")


def globbed(root, pattern):
    for depth in range(1, SEARCH_DEPTH + 1):
        yield from root.glob("*/" * depth + pattern)


def find_bundle(input_root, build_id):
    """('tar', path) or ('tree', dir) for the bundle with this build id."""
    seen = []
    # Kaggle usually expands an uploaded archive, but not always: take either.
    for tar in globbed(input_root, BUNDLE_TAR):
        if not tar.is_file():
            continue
        try:
            with tarfile.open(tar) as tf:
                meta = json.load(tf.extractfile(BUNDLE_MARKER))
        except Exception as e:  # noqa: BLE001 -- report and keep looking
            seen.append(f"{tar} (unreadable: {e})")
            continue
        if meta.get("build_id") == build_id:
            return "tar", tar
        seen.append(f"{tar} (build {str(meta.get('build_id'))[:8]})")
    for marker in globbed(input_root, BUNDLE_MARKER):
        try:
            meta = json.loads(marker.read_text())
        except Exception as e:  # noqa: BLE001
            seen.append(f"{marker} (unreadable: {e})")
            continue
        if meta.get("build_id") == build_id:
            return "tree", marker.parent
        seen.append(f"{marker.parent} (build {str(meta.get('build_id'))[:8]})")
    return None, seen


def forward_termination(proc, flag):
    """If this process is signalled, pass SIGTERM on to the job so it can save,
    and kill it if it does not exit within GRACE_SECONDS. Best effort: on
    Kaggle, `kgr cancel` ended the session without any signal reaching here
    (verified on a live kernel); files already on disk did survive."""
    if not hasattr(signal, "SIGTERM") or not hasattr(os, "killpg"):
        return

    def killpg(sig):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def handler(signum, frame):
        if flag:
            return
        flag.append(signum)
        log(f"\nreceived signal {signum}: forwarding to the job, "
            f"{GRACE_SECONDS}s to save and exit")
        killpg(signal.SIGTERM)
        t = threading.Timer(GRACE_SECONDS, lambda: killpg(signal.SIGKILL))
        t.daemon = True
        t.start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except ValueError:      # not the main thread
            pass


def render(template, subs):
    for k, v in subs.items():
        template = template.replace("{" + k + "}", v)
    return template


def main(cfg):
    input_root = Path(os.environ.get("KGR_INPUT_ROOT", "/kaggle/input"))
    working = Path(os.environ.get("KGR_WORKING_ROOT", "/kaggle/working"))
    bundle_dir = Path(os.environ.get("KGR_BUNDLE_DIR", "/tmp/kgr_bundle"))
    run = cfg["run"]
    t0 = time.time()

    log("=" * 64)
    log(f"kgr  project={cfg['project']}  run={run}  build={cfg['build_id'][:8]}")
    announce_session(cfg, working)   # first, so even a run that aborts below is cancellable
    device, cap = gpu_report()
    if device == "cpu" and cfg["require_gpu"]:
        sys.exit("No GPU attached although one was requested. Kaggle silently "
                 "ignores accelerator requests from accounts without phone "
                 "verification (Settings > Phone Verification), and the GPU "
                 "quota may be used up (`kaggle quota`). Set [run] "
                 "require_gpu = false to train on CPU anyway.")
    if device == "cuda" and cap and cfg["min_cuda_capability"]:
        need = tuple(int(x) for x in cfg["min_cuda_capability"].split("."))
        if cap < need:
            sys.exit(f"GPU compute capability {cap} is below the configured "
                     f"minimum {need}: recent torch wheels ship no kernels for "
                     f"it (e.g. a P100 assigned instead of a T4). Push again "
                     f"or lower [run] min_cuda_capability.")
    dump_inputs(input_root)
    log("=" * 64)

    # -- code ----------------------------------------------------------------
    kind, where = find_bundle(input_root, cfg["build_id"])
    if kind is None:
        log("bundles found:", *(where or ["none"]), sep="\n  ")
        sys.exit("This push's code bundle is not among the inputs -- the "
                 "dataset version was probably not ready yet. Push again.")
    shutil.rmtree(bundle_dir, ignore_errors=True)
    bundle_dir.mkdir(parents=True)
    if kind == "tar":
        extract(where, bundle_dir)
    else:
        shutil.copytree(where, bundle_dir, dirs_exist_ok=True)
    log(f"code: {where}")

    # -- output directory ----------------------------------------------------
    rel_out = Path(cfg["output_dir"])
    if rel_out.is_absolute() or ".." in rel_out.parts:
        sys.exit(f"[run] output_dir must be a relative path, got {rel_out}")
    out_dir = working / rel_out
    out_dir.mkdir(parents=True, exist_ok=True)
    in_bundle = bundle_dir / rel_out
    if in_bundle.is_dir() and not in_bundle.is_symlink():
        shutil.copytree(in_bundle, out_dir, dirs_exist_ok=True)
        shutil.rmtree(in_bundle)
    in_bundle.parent.mkdir(parents=True, exist_ok=True)
    if not in_bundle.exists():
        in_bundle.symlink_to(out_dir, target_is_directory=True)

    # -- resume --------------------------------------------------------------
    # A kernel's output is attachable as the next kernel's input, so a run
    # that hit the time limit continues with every file it saved (weights,
    # optimiser, replay buffer...) instead of starting over.
    resumed = {}
    pat = cfg["resume_glob"]
    skip_root = where if kind == "tree" else None
    for p in [*globbed(input_root, f"{rel_out.as_posix()}/{pat}"), *globbed(input_root, pat)]:
        if not p.is_file() or (skip_root and skip_root in p.parents):
            continue
        prev = resumed.get(p.name)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            resumed[p.name] = p
    for name, src in sorted(resumed.items()):
        shutil.copy2(src, out_dir / name)
    if cfg["resuming"] and not resumed:
        log(f"WARNING: --resume-from was given but nothing matched {pat!r}")
    log(f"resumed files: {sorted(resumed) or 'none, fresh run'}")

    # -- run -----------------------------------------------------------------
    cpus = os.cpu_count() or 1
    subs = {
        "run": run, "extra": cfg["extra"], "device": device,
        "output_dir": str(out_dir), "cpus": str(cpus),
        "cpus_minus_1": str(max(1, cpus - 1)),
        "bundle": str(bundle_dir), "input": str(input_root),
    }
    env = {**os.environ, "KGR_RUN": run, "KGR_DEVICE": device,
           "KGR_OUTPUT_DIR": str(out_dir), "KGR_BUNDLE": str(bundle_dir),
           "KGR_INPUT": str(input_root), "KGR_RESUMED": "1" if resumed else "0",
           "PYTHONUNBUFFERED": "1"}
    workdir = bundle_dir / cfg["workdir"]

    for step in cfg["setup"]:
        step = render(step, subs)
        log(f"setup: {step}")
        if subprocess.run(["bash", "-c", step], cwd=workdir, env=env).returncode != 0:
            sys.exit(f"setup step failed: {step}")

    cmd = render(cfg["command"], subs)
    log(f"launching: {cmd}")
    p = subprocess.Popen(["bash", "-c", cmd], cwd=workdir, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, errors="replace",
                         start_new_session=True)
    terminated = []
    forward_termination(p, terminated)
    for line in p.stdout:
        print(line.rstrip("\n"), flush=True)
    rc = p.wait()
    if terminated:
        log(f"\nkernel was asked to stop (signal {terminated[0]}); the job exited rc={rc}")

    outputs = sorted(x for x in out_dir.rglob("*") if x.is_file())
    log(f"\ncommand exited rc={rc} after {(time.time() - t0) / 60:.1f} min")
    for f in outputs:
        log(f"  output: {f.relative_to(working)}  {f.stat().st_size / 1e6:.1f} MB")
    (working / "kgr_run_summary.json").write_text(json.dumps({
        "run": run, "build_id": cfg["build_id"], "rc": rc, "device": device,
        "terminated": bool(terminated),
        "minutes": round((time.time() - t0) / 60, 2),
        "resumed": sorted(resumed), "outputs": [str(f.relative_to(working)) for f in outputs],
    }, indent=2))
    # A non-zero exit marks the kernel ERROR, which `kgr wait` reports; the
    # output directory is kept either way.
    sys.exit(rc)


if __name__ == "__main__":
    _cfg = globals().get("KGR_CONFIG")
    if _cfg is None:
        if len(sys.argv) != 2:
            sys.exit("usage: kernel_runner.py CONFIG.json")
        _cfg = json.loads(Path(sys.argv[1]).read_text())
    main(_cfg)
