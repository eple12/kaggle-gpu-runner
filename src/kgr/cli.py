"""kgr -- run a training job on a Kaggle GPU kernel from the command line.

    kgr push  RUN   bundle the code, publish it as a private dataset, push a
                    script kernel that trains on it (this starts the run)
    kgr wait  RUN   poll until the kernel finishes
    kgr watch RUN   stream the kernel log live
    kgr fetch RUN   download the kernel output into the project
    kgr run   RUN   push + wait + fetch
    kgr init        write a starter kgr.toml

Everything project-specific lives in kgr.toml; see the README.
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None

CONFIG_NAME = "kgr.toml"
BUNDLE_TAR = "kgr_bundle.tar.gz"
BUNDLE_MARKER = "kgr_bundle_root.json"
KERNEL_FILE = "kgr_kernel.py"
RUNNER_SRC = Path(__file__).with_name("kernel_runner.py")

DEFAULTS: dict = {
    "kaggle": {
        "user": "",
        "project": "",
        "bin": "",
        "accelerator": "NvidiaTeslaT4",
        "internet": False,
        "private": True,
        "timeout": 0,
        "datasets": [],
        "dataset_wait": 900,
    },
    "bundle": {
        "root": ".",
        "include": [],
        "exclude": ["__pycache__", "*.pyc", ".git", ".ipynb_checkpoints",
                    ".venv", "venv", ".kgr"],
    },
    "run": {
        "workdir": ".",
        "command": "",
        "default_extra": "",
        "output_dir": "outputs",
        "resume_glob": "{run}*",
        "require_gpu": True,
        "min_cuda_capability": "7.0",
        "setup": [],
    },
    "fetch": {
        "dest": "outputs",
        "glob": "{run}*",
    },
    "hooks": {
        "pre_push": [],
    },
}

TEMPLATE = '''\
# kgr.toml -- how this project trains on Kaggle. Paths are relative to this file.

[kaggle]
user = "your-kaggle-username"      # owner slug (kaggle.com/<user>)
project = "myproj"                 # prefix for the dataset/kernel slugs
accelerator = "NvidiaTeslaT4"      # case-sensitive; "none" for a CPU kernel
internet = false                   # true needs a phone-verified account
private = true
timeout = 0                        # seconds, 0 = Kaggle's own limit (9 h GPU)
datasets = []                      # extra dataset refs to attach, e.g. ["me/big-data"]

[bundle]
root = "."
include = ["train.py", "src"]      # globs; everything the job needs to run
exclude = ["__pycache__", "*.pyc", ".git", "outputs"]

[run]
workdir = "."                      # cwd for the command, inside the bundle
# Placeholders: {run} {extra} {device} {output_dir} {cpus} {cpus_minus_1}
#               {bundle} {input}
command = "python -u train.py --run-name {run} --out {output_dir} --device {device} {extra}"
default_extra = ""                 # used when `kgr push` gets no --extra
output_dir = "outputs"             # kept as kernel output; resume files land here
resume_glob = "{run}*"             # what to carry over from --resume-from
require_gpu = true                 # abort instead of silently training on CPU
min_cuda_capability = "7.0"        # abort on a GPU your torch build cannot use
setup = []                         # shell commands run first (pip needs internet)

[fetch]
dest = "outputs"                   # local directory for fetched files
glob = "{run}*"

[hooks]
pre_push = []                      # local commands that must pass before pushing
'''


# --------------------------------------------------------------------------
# config

def die(msg: str, code: int = 1) -> "NoReturn":  # type: ignore[name-defined]
    print(f"kgr: {msg}", file=sys.stderr)
    sys.exit(code)


def find_config(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            die(f"config not found: {p}")
        return p.resolve()
    here = Path.cwd().resolve()
    for d in (here, *here.parents):
        if (d / CONFIG_NAME).is_file():
            return d / CONFIG_NAME
    die(f"no {CONFIG_NAME} in {here} or any parent (run `kgr init`)")


def load_config(path: Path) -> dict:
    if tomllib is None:
        die("Python 3.11+ is required (or `pip install tomli`)")
    with open(path, "rb") as f:
        user = tomllib.load(f)
    cfg = {}
    for section, defaults in DEFAULTS.items():
        cfg[section] = {**defaults, **user.get(section, {})}
    unknown = set(user) - set(DEFAULTS)
    if unknown:
        print(f"kgr: ignoring unknown config sections {sorted(unknown)}", file=sys.stderr)
    for key in ("user", "project"):
        if not cfg["kaggle"][key] or cfg["kaggle"][key].startswith("your-"):
            die(f"set [kaggle] {key} in {path}")
    if not cfg["run"]["command"]:
        die(f"set [run] command in {path}")
    return cfg


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def kernel_ref(cfg: dict, run: str) -> str:
    return f"{cfg['kaggle']['user']}/{slugify(cfg['kaggle']['project'] + '-' + run)}"


def dataset_ref(cfg: dict) -> str:
    return f"{cfg['kaggle']['user']}/{slugify(cfg['kaggle']['project'] + '-code')}"


def uses_gpu(cfg: dict, override: str | None = None) -> tuple[bool, str]:
    acc = override or cfg["kaggle"]["accelerator"] or "none"
    return acc.lower() not in ("none", "cpu", ""), acc


# --------------------------------------------------------------------------
# kaggle CLI

def kaggle_argv(cfg: dict) -> list[str]:
    raw = cfg["kaggle"]["bin"] or os.environ.get("KGR_KAGGLE") or "kaggle"
    argv = shlex.split(raw, posix=os.name != "nt")
    if shutil.which(argv[0]) is None and not Path(argv[0]).exists():
        die(f"kaggle CLI not found ({argv[0]}). `pip install kaggle`, or set "
            f"[kaggle] bin / $KGR_KAGGLE")
    return argv


def kaggle(cfg: dict, *args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run([*kaggle_argv(cfg), *args], text=True,
                          capture_output=capture)


def dataset_state(cfg: dict, ref: str) -> tuple[str, int | None] | None:
    """(status, version) of a dataset, or None when it does not exist."""
    r = kaggle(cfg, "datasets", "status", ref, "--format",
               "json(status,current_version_number)", capture=True)
    if r.returncode != 0:
        # Older CLIs have no --format; a missing dataset also fails here.
        r = kaggle(cfg, "datasets", "status", ref, capture=True)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return r.stdout.strip().lower(), None
    try:
        d = json.loads(r.stdout.strip())
        return str(d.get("status", "")).lower(), d.get("current_version_number")
    except json.JSONDecodeError:
        return r.stdout.strip().lower(), None


KERNEL_STATES = ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED",
                 "CANCEL", "RUNNING", "QUEUED", "NEW_SCRIPT")


def kernel_status(cfg: dict, ref: str) -> str | None:
    r = kaggle(cfg, "kernels", "status", ref, capture=True)
    text = (r.stdout + r.stderr).upper()
    for s in KERNEL_STATES:
        if s in text:
            return s
    return None


# --------------------------------------------------------------------------
# bundle

def build_bundle(cfg: dict, root: Path, dest: Path, build_id: str, run: str) -> int:
    broot = (root / cfg["bundle"]["root"]).resolve()
    excludes = cfg["bundle"]["exclude"]
    include = cfg["bundle"]["include"]
    if not include:
        die("[bundle] include is empty -- list what the job needs")

    def skipped(rel: str) -> bool:
        return any(fnmatch.fnmatch(rel, pat) or
                   any(fnmatch.fnmatch(part, pat) for part in rel.split("/"))
                   for pat in excludes)

    added: set[str] = set()
    with tarfile.open(dest, "w:gz") as tf:
        for pat in include:
            matches = sorted(broot.glob(pat))
            if not matches:
                die(f"bundle include {pat!r} matched nothing under {broot}")
            for m in matches:
                files = [m] if m.is_file() else sorted(p for p in m.rglob("*") if p.is_file())
                for p in files:
                    try:
                        rel = p.resolve().relative_to(broot).as_posix()
                    except ValueError:
                        die(f"{p} is outside the bundle root {broot}")
                    if rel in added or skipped(rel):
                        continue
                    tf.add(p, arcname=rel, recursive=False)
                    added.add(rel)
        # The marker is how the kernel tells THIS push's code apart from an
        # older dataset version or a previous run's output that also carries
        # a copy of the tree.
        marker = json.dumps({
            "project": cfg["kaggle"]["project"], "run": run, "build_id": build_id,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2).encode()
        info = tarfile.TarInfo(BUNDLE_MARKER)
        info.size = len(marker)
        info.mtime = int(time.time())
        tf.addfile(info, io.BytesIO(marker))
    return len(added)


def render_kernel(cfg: dict, run: str, build_id: str, extra: str, resuming: bool) -> str:
    r = cfg["run"]
    kcfg = {
        "project": cfg["kaggle"]["project"],
        "run": run,
        "build_id": build_id,
        "command": r["command"],
        "extra": extra,
        "workdir": r["workdir"],
        "output_dir": r["output_dir"],
        "resume_glob": r["resume_glob"].replace("{run}", run),
        "resuming": resuming,
        "require_gpu": bool(r["require_gpu"]) and uses_gpu(cfg)[0],
        "min_cuda_capability": str(r["min_cuda_capability"] or ""),
        "setup": list(r["setup"]),
    }
    head = ("# Generated by kgr -- edit kgr.toml, not this file.\n"
            "import json as _kgr_json\n"
            f"KGR_CONFIG = _kgr_json.loads({json.dumps(kcfg)!r})\n\n")
    return head + RUNNER_SRC.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# commands

def cmd_init(args) -> int:
    p = Path(args.config or CONFIG_NAME)
    if p.exists():
        die(f"{p} already exists")
    p.write_text(TEMPLATE, encoding="utf-8")
    print(f"wrote {p} -- fill in [kaggle] user/project, [bundle] include and [run] command")
    return 0


def cmd_push(cfg: dict, root: Path, args) -> str:
    run = args.run
    extra = args.extra if args.extra is not None else cfg["run"]["default_extra"]
    gpu, acc = uses_gpu(cfg, args.accelerator)
    kref = kernel_ref(cfg, run)
    dref = dataset_ref(cfg)

    if not args.skip_hooks:
        for hook in cfg["hooks"]["pre_push"]:
            print(f"=== pre-push hook: {hook}", flush=True)
            if subprocess.run(hook, shell=True, cwd=root).returncode != 0:
                die("pre-push hook failed -- not spending GPU quota on it "
                    "(--skip-hooks to override)")

    stage = Path(tempfile.mkdtemp(prefix="kgr-"))
    data, kdir = stage / "data", stage / "kernel"
    data.mkdir()
    kdir.mkdir()
    build_id = uuid.uuid4().hex

    print("=== bundling", flush=True)
    n = build_bundle(cfg, root, data / BUNDLE_TAR, build_id, run)
    print(f"  {n} files, {(data / BUNDLE_TAR).stat().st_size / 1e6:.2f} MB  "
          f"(build {build_id[:8]})")
    (data / "dataset-metadata.json").write_text(json.dumps({
        "title": f"{cfg['kaggle']['project']} code bundle",
        "id": dref,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))

    (kdir / KERNEL_FILE).write_text(
        render_kernel(cfg, run, build_id, extra, bool(args.resume_from)), encoding="utf-8")
    title = slugify(cfg["kaggle"]["project"] + "-" + run).replace("-", " ")
    if len(title) < 5:
        die(f"kernel title {title!r} is under Kaggle's 5-character minimum; "
            f"use a longer project or run name")
    meta = {
        "id": kref,
        # Kaggle derives the slug from the title; keep them in lockstep or the
        # kernel lands under a different slug than the one we poll.
        "title": title,
        "code_file": KERNEL_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": str(bool(cfg["kaggle"]["private"])).lower(),
        "enable_gpu": str(gpu).lower(),
        "enable_internet": str(bool(cfg["kaggle"]["internet"])).lower(),
        "dataset_sources": [dref, *cfg["kaggle"]["datasets"]],
        "competition_sources": [],
        "kernel_sources": [args.resume_from] if args.resume_from else [],
        "model_sources": [],
    }
    if gpu:
        meta["machine_shape"] = acc
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    if args.dry_run:
        print(f"\ndry run -- staged in {stage}, nothing sent to Kaggle")
        return kref

    print("=== publishing the code dataset", flush=True)
    before = dataset_state(cfg, dref)
    if before is None:
        rc = kaggle(cfg, "datasets", "create", "-p", str(data), "--dir-mode", "zip").returncode
    else:
        msg = f"kgr {run} {datetime.now():%Y-%m-%d %H:%M}"
        rc = kaggle(cfg, "datasets", "version", "-p", str(data), "-m", msg,
                    "--dir-mode", "zip").returncode
    if rc != 0:
        die("dataset publish failed (expired API token? see output above)")

    # Publishing is asynchronous. Push the kernel before the new version is
    # ready and Kaggle either drops the attachment or mounts the previous
    # version -- a run with no code, or with last push's code.
    print("waiting for the dataset version to be ready", end="", flush=True)
    old_version = before[1] if before else None
    deadline = time.time() + float(cfg["kaggle"]["dataset_wait"])
    state = None
    while time.time() < deadline:
        state = dataset_state(cfg, dref)
        if state and state[0] == "ready" and (
                old_version is None or state[1] is None or state[1] > old_version):
            print(f" -> ready (version {state[1]})")
            break
        print(".", end="", flush=True)
        time.sleep(10)
    else:
        die(f"\ndataset never became ready (last: {state})")

    print("=== pushing the kernel (this starts the run)", flush=True)
    push = ["kernels", "push", "-p", str(kdir)]
    if gpu:
        # Pass the accelerator on the command line as well as in the metadata:
        # metadata alone has come back as a CPU image.
        push += ["--accelerator", acc]
    if cfg["kaggle"]["timeout"]:
        push += ["-t", str(int(cfg["kaggle"]["timeout"]))]
    if kaggle(cfg, *push).returncode != 0:
        die("kernel push failed")
    shutil.rmtree(stage, ignore_errors=True)

    print(f"\nkernel: {kref}\n"
          f"  https://www.kaggle.com/code/{kref}\n"
          f"  kgr watch {run}    kgr wait {run}    kgr fetch {run}")
    return kref


def cmd_wait(cfg: dict, run: str, interval: float) -> int:
    kref = kernel_ref(cfg, run)
    last, misses = None, 0
    while True:
        st = kernel_status(cfg, kref)
        if st is None:
            misses += 1
            if misses >= 10:
                die(f"could not read the status of {kref} ten times in a row", 2)
        else:
            misses = 0
            if st != last:
                print(f"[{datetime.now():%H:%M:%S}] {kref}: {st}", flush=True)
                last = st
            if st == "COMPLETE":
                return 0
            if st == "ERROR" or st.startswith("CANCEL"):
                return 1
        time.sleep(interval)


def cmd_watch(cfg: dict, run: str) -> int:
    kref = kernel_ref(cfg, run)
    kaggle(cfg, "kernels", "status", kref)
    print("=== streaming (Ctrl-C stops watching; the run keeps going)", flush=True)
    try:
        return kaggle(cfg, "kernels", "logs", kref, "-f").returncode
    except KeyboardInterrupt:
        return 0


def cmd_fetch(cfg: dict, root: Path, run: str, dest: str | None, everything: bool) -> int:
    kref = kernel_ref(cfg, run)
    out = (root / (dest or cfg["fetch"]["dest"])).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="kgr-out-"))
    print(f"=== downloading {kref} output", flush=True)
    if kaggle(cfg, "kernels", "output", kref, "-p", str(tmp)).returncode != 0:
        die("download failed")

    out.mkdir(parents=True, exist_ok=True)
    copied = []
    if everything:
        target = out / run
        shutil.copytree(tmp, target, dirs_exist_ok=True)
        copied.append(target)
    else:
        pattern = cfg["fetch"]["glob"].replace("{run}", run)
        for f in sorted(p for p in tmp.rglob(pattern) if p.is_file()):
            if f.name == KERNEL_FILE or f.name == BUNDLE_MARKER:
                continue
            target = out / f.name
            if target in copied:
                print(f"  warning: several {f.name} in the output, keeping the last")
            shutil.copy2(f, target)
            copied.append(target)
        for f in tmp.rglob("*.log"):
            target = out / f"{run}.kaggle.log"
            shutil.copy2(f, target)
            copied.append(target)

    for c in copied:
        size = c.stat().st_size / 1e6 if c.is_file() else 0
        print(f"  -> {c}  {size:.1f} MB" if c.is_file() else f"  -> {c}/")
    if not any(c.name != f"{run}.kaggle.log" for c in copied):
        listing = [str(p.relative_to(tmp)) for p in tmp.rglob("*") if p.is_file()][:30]
        print(f"nothing matched {cfg['fetch']['glob']!r}; output held: {listing}")
        shutil.rmtree(tmp, ignore_errors=True)
        return 1
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kgr", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", help=f"path to {CONFIG_NAME} (default: search upward)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="write a starter kgr.toml")

    def with_run(name, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("run", help="run name, e.g. resnet_v3 (also names the kernel)")
        return p

    for name, help_ in (("push", "bundle + push a kernel (starts the run)"),
                        ("run", "push, wait for completion, fetch")):
        p = with_run(name, help_)
        p.add_argument("--extra", help="replaces [run] default_extra for this push")
        p.add_argument("--resume-from", metavar="USER/KERNEL",
                       help="attach a previous kernel's output and carry its files over")
        p.add_argument("--accelerator", help="override [kaggle] accelerator")
        p.add_argument("--skip-hooks", action="store_true", help="skip [hooks] pre_push")
        p.add_argument("--dry-run", action="store_true",
                       help="stage the bundle and metadata locally, send nothing")
        if name == "run":
            p.add_argument("--interval", type=float, default=120)
            p.add_argument("--dest")
            p.add_argument("--all", action="store_true")

    p = with_run("wait", "poll until the kernel finishes (exit 0 = COMPLETE)")
    p.add_argument("--interval", type=float, default=120, help="seconds between polls")
    with_run("watch", "stream the kernel log")
    with_run("status", "print the kernel status")
    p = with_run("fetch", "download the kernel output")
    p.add_argument("--dest", help="override [fetch] dest")
    p.add_argument("--all", action="store_true", help="copy the whole output tree to <dest>/<run>/")
    with_run("ref", "print the kernel ref for a run name")

    args = ap.parse_args(argv)
    if args.cmd == "init":
        return cmd_init(args)

    cfg_path = find_config(args.config)
    cfg = load_config(cfg_path)
    root = cfg_path.parent

    if args.cmd == "push":
        cmd_push(cfg, root, args)
        return 0
    if args.cmd == "run":
        cmd_push(cfg, root, args)
        if args.dry_run:
            return 0
        rc = cmd_wait(cfg, args.run, args.interval)
        # Fetch even after an ERROR: whatever the job saved before it died
        # (or before the 9 h wall) is still in the output.
        frc = cmd_fetch(cfg, root, args.run, args.dest, args.all)
        return rc or frc
    if args.cmd == "wait":
        return cmd_wait(cfg, args.run, args.interval)
    if args.cmd == "watch":
        return cmd_watch(cfg, args.run)
    if args.cmd == "status":
        return kaggle(cfg, "kernels", "status", kernel_ref(cfg, args.run)).returncode
    if args.cmd == "fetch":
        return cmd_fetch(cfg, root, args.run, args.dest, args.all)
    if args.cmd == "ref":
        print(kernel_ref(cfg, args.run))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
