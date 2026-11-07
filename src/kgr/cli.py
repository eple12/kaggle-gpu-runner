"""kgr -- run a job on a Kaggle GPU kernel from the command line.

  run it
    kgr push   RUN   bundle the code, publish it as a private dataset, push a
                     script kernel that runs it (this starts the run)
    kgr wait   RUN   poll until the kernel finishes
    kgr watch  RUN   stream the kernel log live
    kgr fetch  RUN   download the kernel output into the project
    kgr run    RUN   push + wait + fetch
    kgr exec   FILE  run one script on Kaggle, no kgr.toml needed

  manage it
    kgr status RUN   kernel status        kgr runs      this project's runs
    kgr info   RUN   session id, container, URL of a run
    kgr cancel RUN   stop a running kernel (a hard stop: see the README)
    kgr logs   RUN   print or save the kernel log
    kgr quota        weekly GPU/TPU quota left

  competitions
    kgr data COMP    download the competition data
    kgr submit COMP  submit a file, or a kernel's output (code competitions)
    kgr submissions COMP

  setup
    kgr init         write a starter kgr.toml (--global: your Kaggle username)
    kgr doctor       check the kaggle CLI, credentials, quota and config
    kgr probe        run a throwaway CPU kernel and show what it reports

Everything project-specific lives in kgr.toml; see the README.
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None

from kgr import session as sess

CONFIG_NAME = "kgr.toml"
USER_CONFIG_NAME = ".kgr.toml"
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
        "competitions": [],
        "kernels": [],
        "models": [],
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
    "cancel": {
        "session_env": "",
        "wait": 90,
    },
    "hooks": {
        "pre_push": [],
    },
}

ACCELERATORS = {
    "t4": "NvidiaTeslaT4",
    "p100": "NvidiaTeslaP100",
    "l4": "NvidiaL4",
    "cpu": "none",
    "none": "none",
}

TEMPLATE = '''\
# kgr.toml -- how this project runs on Kaggle. Paths are relative to this file.

[kaggle]
user = "{user}"      # owner slug (kaggle.com/<user>)
project = "{project}"                 # prefix for the dataset/kernel slugs
accelerator = "NvidiaTeslaT4"      # t4 | p100 | l4 | none (CPU); case-sensitive otherwise
internet = false                   # true needs a phone-verified account
private = true
timeout = 0                        # seconds, 0 = Kaggle's own limit (9 h GPU)
datasets = []                      # extra dataset refs to attach, e.g. ["me/big-data"]
competitions = []                  # competition data to attach, e.g. ["titanic"] (rules accepted)
kernels = []                       # other kernels whose output you read
models = []                        # model refs, e.g. ["google/gemma/pyTorch/2b/1"]

[bundle]
root = "."
include = ["train.py", "src"]      # globs; everything the job needs to run
exclude = ["__pycache__", "*.pyc", ".git", "outputs"]

[run]
workdir = "."                      # cwd for the command, inside the bundle
# Placeholders: {{run}} {{extra}} {{device}} {{output_dir}} {{cpus}} {{cpus_minus_1}}
#               {{bundle}} {{input}}
command = "python -u train.py --run-name {{run}} --out {{output_dir}} --device {{device}} {{extra}}"
default_extra = ""                 # used when `kgr push` gets no --extra
output_dir = "outputs"             # kept as kernel output; resume files land here
resume_glob = "{{run}}*"             # what to carry over from --resume-from
require_gpu = true                 # abort instead of silently training on CPU
min_cuda_capability = "7.0"        # abort on a GPU your torch build cannot use
setup = []                         # shell commands run first (pip needs internet)

[fetch]
dest = "outputs"                   # local directory for fetched files
glob = "{{run}}*"

[cancel]
session_env = ""                   # kernel env var holding the session id, if `kgr info` shows one
wait = 90                          # seconds `kgr cancel` waits for the kernel to stop

[hooks]
pre_push = []                      # local commands that must pass before pushing
'''


# --------------------------------------------------------------------------
# config

def die(msg: str, code: int = 1) -> "NoReturn":  # type: ignore[name-defined]
    print(f"kgr: {msg}", file=sys.stderr)
    sys.exit(code)


def user_config_path() -> Path:
    return Path(os.environ.get("KGR_USER_CONFIG") or Path.home() / USER_CONFIG_NAME)


def find_config(explicit: str | None, required: bool = True) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            die(f"config not found: {p}")
        return p.resolve()
    here = Path.cwd().resolve()
    for d in (here, *here.parents):
        if (d / CONFIG_NAME).is_file():
            return d / CONFIG_NAME
    if required:
        die(f"no {CONFIG_NAME} in {here} or any parent (run `kgr init`)")
    return None


def read_toml(path: Path) -> dict:
    if tomllib is None:
        die("Python 3.11+ is required (or `pip install tomli`)")
    with open(path, "rb") as f:
        return tomllib.load(f)


def merge_config(path: Path | None) -> dict:
    """DEFAULTS < ~/.kgr.toml < the project's kgr.toml, section by section."""
    layers = []
    up = user_config_path()
    if up.is_file():
        layers.append((up, read_toml(up)))
    if path is not None:
        layers.append((path, read_toml(path)))
    cfg = {section: dict(defaults) for section, defaults in DEFAULTS.items()}
    for origin, data in layers:
        for section, values in data.items():
            if section not in DEFAULTS:
                print(f"kgr: ignoring unknown config section [{section}] in {origin}",
                      file=sys.stderr)
                continue
            cfg[section].update(values)
    return cfg


def detect_username(cfg: dict) -> str:
    """The username the kaggle CLI itself is logged in as (`kaggle config view`)."""
    try:
        r = kaggle(cfg, "config", "view", capture=True)
    except SystemExit:
        return ""
    if r.returncode != 0:
        return ""
    m = re.search(r"^\s*-?\s*username:\s*(\S+)", r.stdout, re.MULTILINE)
    return m.group(1) if m else ""


def validate_config(cfg: dict, origin: object, need_command: bool = True) -> dict:
    if not cfg["kaggle"]["user"]:
        cfg["kaggle"]["user"] = os.environ.get("KGR_USER") or detect_username(cfg)
    for key in ("user", "project"):
        if not cfg["kaggle"][key] or cfg["kaggle"][key].startswith("your-"):
            hint = " (or run `kgr init --global --user NAME` once)" if key == "user" else ""
            die(f"set [kaggle] {key} in {origin}{hint}")
    if need_command and not cfg["run"]["command"]:
        die(f"set [run] command in {origin}")
    return cfg


def load_config(path: Path, need_command: bool = True) -> dict:
    return validate_config(merge_config(path), path, need_command)


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def kernel_ref(cfg: dict, run: str) -> str:
    return f"{cfg['kaggle']['user']}/{slugify(cfg['kaggle']['project'] + '-' + run)}"


def dataset_ref(cfg: dict) -> str:
    return f"{cfg['kaggle']['user']}/{slugify(cfg['kaggle']['project'] + '-code')}"


def kernel_url(ref: str) -> str:
    return f"https://www.kaggle.com/code/{ref}"


def resolve_accelerator(name: str | None) -> str | None:
    return ACCELERATORS.get(name.lower(), name) if name else name


def uses_gpu(cfg: dict, override: str | None = None) -> tuple[bool, str]:
    acc = resolve_accelerator(override) or resolve_accelerator(cfg["kaggle"]["accelerator"]) or "none"
    return acc.lower() not in ("none", "cpu", ""), acc


def project_competition(cfg: dict, given: str | None) -> str:
    comp = given or (cfg["kaggle"]["competitions"] or [""])[0]
    if not comp:
        die("name the competition (or list it under [kaggle] competitions)")
    return comp


# --------------------------------------------------------------------------
# kaggle CLI

def kaggle_argv(cfg: dict) -> list[str]:
    raw = cfg["kaggle"]["bin"] or os.environ.get("KGR_KAGGLE") or "kaggle"
    argv = shlex.split(raw, posix=os.name != "nt")
    # Windows (non-POSIX) splitting keeps the quotes around "C:\Program Files\..."
    argv = [a[1:-1] if len(a) > 1 and a[0] == a[-1] and a[0] in "\"'" else a for a in argv]
    if shutil.which(argv[0]) is None and not Path(argv[0]).exists():
        die(f"kaggle CLI not found ({argv[0]}). `pip install kaggle`, or set "
            f"[kaggle] bin / $KGR_KAGGLE")
    return argv


def child_env() -> dict:
    # The kaggle CLI prints non-ASCII (kernel logs, titles). Without this a
    # Windows console codepage (cp949, cp1252...) makes it crash or garble.
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def kaggle(cfg: dict, *args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run([*kaggle_argv(cfg), *args], text=True, encoding="utf-8",
                          errors="replace", capture_output=capture, env=child_env())


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
    # Newer CLIs print an "outdated version" warning on stdout before the
    # JSON, so take the JSON object itself rather than the whole output.
    text = r.stdout.strip()
    a, b = text.find("{"), text.rfind("}")
    try:
        d = json.loads(text[a:b + 1])
        return str(d.get("status", "")).lower(), d.get("current_version_number")
    except (json.JSONDecodeError, ValueError):
        return text.lower(), None


KERNEL_STATES = ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED",
                 "CANCEL", "RUNNING", "QUEUED", "NEW_SCRIPT")


def kernel_status(cfg: dict, ref: str) -> str | None:
    r = kaggle(cfg, "kernels", "status", ref, capture=True)
    text = (r.stdout + r.stderr).upper()
    for s in KERNEL_STATES:
        if s in text:
            return s
    return None


def is_finished(status: str | None) -> bool:
    # CANCEL_REQUESTED is only the request landing; the session is still
    # tearing down. CANCEL_ACKNOWLEDGED is the end.
    return bool(status) and (status in ("COMPLETE", "ERROR") or
                             (status.startswith("CANCEL") and status != "CANCEL_REQUESTED"))


def stream_log_lines(cfg: dict, ref: str, timeout: float):
    """Lines of `kaggle kernels logs -f` until it ends or `timeout` seconds
    pass. The process is killed when the generator is closed."""
    proc = subprocess.Popen([*kaggle_argv(cfg), "kernels", "logs", ref, "-f"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=child_env())
    lines: queue.Queue = queue.Queue()

    def pump():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    deadline = time.time() + timeout
    try:
        while True:
            left = deadline - time.time()
            if left <= 0:
                return
            try:
                line = lines.get(timeout=left)
            except queue.Empty:
                return
            if line is None:
                return
            yield line
    finally:
        if proc.poll() is None:
            proc.kill()


# --------------------------------------------------------------------------
# session info: what the kernel reported about itself

def fetch_log(cfg: dict, ref: str) -> str | None:
    r = kaggle(cfg, "kernels", "logs", ref, capture=True)
    return r.stdout if r.returncode == 0 else None


def record_session(cfg: dict, root: Path, run: str, info: dict) -> dict:
    sid, source = sess.pick_session_id(info, cfg["cancel"]["session_env"])
    return sess.save_state(root, run, session=info, session_id=sid,
                           session_id_source=source or None)


def capture_session(cfg: dict, root: Path, run: str, timeout: float = 45) -> dict | None:
    """Read this run's KGR_SESSION_INFO from its log and remember it.

    The persisted log (works once a session is over) is tried first, then the
    live stream (works while it runs). Returns the saved state or None.
    """
    kref = kernel_ref(cfg, run)
    build = sess.load_state(root, run).get("build_id")
    text = fetch_log(cfg, kref)
    info = sess.find_session_info(text, build) if text else None
    if info is None and timeout > 0:
        stream = stream_log_lines(cfg, kref, timeout)
        try:
            for line in stream:
                got = sess.parse_session_line(line)
                if got and (build is None or got.get("build_id") == build):
                    info = got
                    break
        finally:
            stream.close()
    return record_session(cfg, root, run, info) if info else None


def describe_session(state: dict) -> list[str]:
    info = state.get("session") or {}
    if not info:
        return ["session:   not reported yet (the kernel prints it as its first step)"]
    out = []
    sid = state.get("session_id")
    if sid:
        out.append(f"session:   {sid}   <- {state.get('session_id_source')}")
    else:
        out.append(f"session:   id unknown; numeric ids the kernel reported: "
                   f"{sess.describe_candidates(info)}")
    if info.get("hostname"):
        out.append(f"host:      {info['hostname']}")
    for c in info.get("containers") or []:
        out.append(f"container: {c}")
    if info.get("started_at"):
        out.append(f"started:   {info['started_at']}")
    return out


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
# commands: setup

def cmd_init(args) -> int:
    if args.glob:
        p = user_config_path()
        if not args.user:
            die("--global needs --user NAME (your Kaggle username)")
        if p.exists() and not args.force:
            die(f"{p} already exists (--force to overwrite)")
        p.write_text(f'# kgr defaults for every project on this machine\n'
                     f'[kaggle]\nuser = "{args.user}"\n', encoding="utf-8")
        print(f"wrote {p} -- projects no longer need [kaggle] user")
        return 0
    p = Path(args.config or CONFIG_NAME)
    if p.exists():
        die(f"{p} already exists")
    user = args.user
    if not user and user_config_path().is_file():
        user = read_toml(user_config_path()).get("kaggle", {}).get("user")
    project = args.project or slugify(Path.cwd().name) or "myproj"
    p.write_text(TEMPLATE.format(user=user or "your-kaggle-username", project=project),
                 encoding="utf-8")
    print(f"wrote {p} -- fill in [bundle] include and [run] command"
          + ("" if user else ", and [kaggle] user"))
    return 0


def cmd_doctor(args) -> int:
    bad = 0

    def line(ok: bool | None, what: str, detail: str = ""):
        nonlocal bad
        mark = {True: "ok ", False: "!! ", None: "-- "}[ok]
        if ok is False:
            bad += 1
        print(f"[{mark}] {what}" + (f"  {detail}" if detail else ""))

    line(sys.version_info >= (3, 9), "python", sys.version.split()[0])
    line(tomllib is not None, "toml reader", "tomllib" if tomllib else "pip install tomli")

    cfg_path = find_config(args.config, required=False)
    try:
        cfg = merge_config(cfg_path)
    except SystemExit:
        cfg = None
    line(None if cfg_path is None else True, "project config",
         str(cfg_path) if cfg_path else "none in this directory (kgr init)")
    up = user_config_path()
    line(None if not up.is_file() else True, "user config",
         f"{up}" + ("" if up.is_file() else "  (absent; `kgr init --global --user NAME` sets your username once)"))

    fake = {"kaggle": {"bin": (cfg or {}).get("kaggle", {}).get("bin", "")}}
    try:
        argv = kaggle_argv(fake)
    except SystemExit:
        line(False, "kaggle CLI", "not found: pip install kaggle")
        return 1
    r = subprocess.run([*argv, "--version"], text=True, encoding="utf-8", errors="replace",
                       capture_output=True, env=child_env())
    ver = next((l for l in (r.stdout + r.stderr).splitlines() if l.strip().startswith("Kaggle CLI")), "").strip()
    line(r.returncode == 0, "kaggle CLI", ver or " ".join(argv))

    home = Path.home() / ".kaggle"
    creds = [n for n in ("kaggle.json", "access_token.txt") if (home / n).is_file()]
    creds += [v for v in ("KAGGLE_API_TOKEN", "KAGGLE_KEY") if os.environ.get(v)]
    line(bool(creds), "credentials", ", ".join(creds) if creds else "none: Kaggle > Settings > API > Create New Token")

    q = subprocess.run([*argv, "quota"], text=True, encoding="utf-8", errors="replace",
                       capture_output=True, env=child_env())
    quota = "\n".join(l for l in q.stdout.splitlines() if l.strip() and "outdated" not in l)
    line(q.returncode == 0, "API access + quota", ("\n" + quota) if q.returncode == 0 else (q.stderr.strip() or q.stdout.strip())[:200])

    try:
        import kagglesdk.kernels.types.kernels_api_service as ks  # noqa: F401
        has_cancel = hasattr(ks, "ApiCancelKernelSessionRequest")
    except ImportError:
        has_cancel = False
    line(has_cancel, "cancel support", "kaggle SDK has CancelKernelSession" if has_cancel else "pip install -U kaggle (2.x)")

    if cfg is not None:
        user = cfg["kaggle"]["user"]
        proj = cfg["kaggle"]["project"]
        if not user:
            user = detect_username(cfg)
            cfg["kaggle"]["user"] = user
        if cfg_path is None and not user:
            line(None, "username", "not set (needed for push / exec)")
        elif cfg_path is None:
            line(True, "username", f"{user} (from the kaggle CLI login)")
        else:
            ok = bool(user) and not user.startswith("your-") and bool(proj)
            line(ok, "kaggle user/project", f"user={user or '?'} project={proj or '?'}")
    print("\nall good" if not bad else f"\n{bad} problem(s)")
    return 1 if bad else 0


# --------------------------------------------------------------------------
# commands: run it

def cmd_push(cfg: dict, root: Path, args) -> str:
    run = args.run
    extra = args.extra if args.extra is not None else cfg["run"]["default_extra"]
    gpu, acc = uses_gpu(cfg, args.accelerator)
    kref = kernel_ref(cfg, run)
    dref = dataset_ref(cfg)
    if getattr(args, "resume", False) and not args.resume_from:
        args.resume_from = kref

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
        "competition_sources": list(cfg["kaggle"]["competitions"]),
        "kernel_sources": [*([args.resume_from] if args.resume_from else []),
                           *cfg["kaggle"]["kernels"]],
        "model_sources": list(cfg["kaggle"]["models"]),
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

    # A new push is a new session: forget the previous one's id.
    sess.state_path(root, run).unlink(missing_ok=True)
    sess.save_state(root, run, kernel=kref, project=cfg['kaggle']['project'], build_id=build_id,
                    pushed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    accelerator=acc if gpu else "none", resume_from=args.resume_from or None)

    print(f"\nkernel: {kref}\n"
          f"  {kernel_url(kref)}\n"
          f"  kgr watch {run}    kgr wait {run}    kgr fetch {run}    kgr cancel {run}")
    return kref


def cmd_wait(cfg: dict, root: Path, run: str, interval: float) -> int:
    kref = kernel_ref(cfg, run)
    last, misses, tries = None, 0, 0
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
            # Remember the session while it is alive, so `kgr cancel` from
            # another terminal needs no log round trip.
            if st == "RUNNING" and tries < 3 and not sess.load_state(root, run).get("session"):
                tries += 1
                got = capture_session(cfg, root, run, timeout=30)
                if got and got.get("session_id"):
                    print(f"  session {got['session_id']}  (kgr cancel {run})", flush=True)
            if st == "COMPLETE":
                return 0
            if is_finished(st):
                return 1
        time.sleep(interval)


def cmd_watch(cfg: dict, root: Path, run: str) -> int:
    kref = kernel_ref(cfg, run)
    kaggle(cfg, "kernels", "status", kref)
    print("=== streaming (Ctrl-C stops watching; the run keeps going)", flush=True)
    build = sess.load_state(root, run).get("build_id")
    proc = subprocess.Popen([*kaggle_argv(cfg), "kernels", "logs", kref, "-f"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=child_env())
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            info = sess.parse_session_line(line)
            if info and (build is None or info.get("build_id") == build):
                record_session(cfg, root, run, info)
        return proc.wait()
    except KeyboardInterrupt:
        proc.kill()
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
# commands: manage it

def cmd_status(cfg: dict, root: Path, run: str) -> int:
    kref = kernel_ref(cfg, run)
    rc = kaggle(cfg, "kernels", "status", kref).returncode
    sid = sess.load_state(root, run).get("session_id")
    if sid:
        print(f"session {sid}")
    return rc


def cmd_info(cfg: dict, root: Path, args) -> int:
    run = args.run
    kref = kernel_ref(cfg, run)
    state = sess.load_state(root, run)
    if args.refresh or not state.get("session"):
        state = capture_session(cfg, root, run, timeout=0 if args.no_stream else 45) or state
    status = kernel_status(cfg, kref)
    if args.json:
        print(json.dumps({"kernel": kref, "status": status, "url": kernel_url(kref), **state}, indent=2))
        return 0
    print(f"kernel:    {kref}")
    print(f"status:    {status or 'unknown'}")
    print(f"url:       {kernel_url(kref)}")
    for key, label in (("accelerator", "gpu"), ("pushed_at", "pushed"), ("build_id", "build")):
        if state.get(key):
            print(f"{label + ':':<10} {state[key]}")
    print("\n".join(describe_session(state)))
    if state.get("session") and not state.get("session_id"):
        print("\nNo variable was recognised as the session id. If one above is, either\n"
              f"  kgr cancel {run} --session-id <number>\n"
              "or set [cancel] session_env = \"NAME\" in kgr.toml.")
    return 0


def cmd_cancel(cfg: dict, root: Path, args) -> int:
    run = args.run
    kref = kernel_ref(cfg, run)
    status = kernel_status(cfg, kref)
    if is_finished(status) and not args.force:
        print(f"{kref} is already {status}; nothing to cancel")
        return 0

    sid, source = None, ""
    if args.session_id:
        try:
            sid, source = sess.parse_session_id(args.session_id), "--session-id"
        except ValueError as e:
            die(str(e))
    state = sess.load_state(root, run)
    if sid is None and state.get("session_id") and not args.refresh:
        sid, source = int(state["session_id"]), f"saved by kgr ({state.get('session_id_source')})"
    if sid is None:
        print("looking for the session id in the kernel log ...", flush=True)
        state = capture_session(cfg, root, run, timeout=args.look) or state
        if state.get("session_id"):
            sid, source = int(state["session_id"]), f"kernel log ({state.get('session_id_source')})"

    if sid is None:
        if state.get("session"):
            print("The kernel reported itself but none of its ids is recognisably the session:\n  "
                  + sess.describe_candidates(state["session"]))
            print(f"Pick the right one:  kgr cancel {run} --session-id <number>\n"
                  "or set [cancel] session_env in kgr.toml.  `kgr info` shows everything it reported.")
        else:
            print(f"No session report yet (kernel is {status or 'unknown'}). The kernel prints it as its "
                  "first step, so a kernel still QUEUED has nothing to cancel by id.\n"
                  "Retry in a minute, or pass --session-id.")
        print(f"Manual stop: open {kernel_url(kref)} -> Session -> Stop.")
        return 1

    print(f"cancelling session {sid} of {kref}  (id from {source})")
    if args.dry_run:
        print("dry run -- nothing sent")
        return 0
    err = sess.sdk_cancel(sid)
    if err:
        print(f"cancel failed: {err}", file=sys.stderr)
        print(f"Manual stop: open {kernel_url(kref)} -> Session -> Stop.", file=sys.stderr)
        return 1
    print("cancel requested", flush=True)
    sess.save_state(root, run, cancelled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    deadline = time.time() + float(args.wait if args.wait is not None else cfg["cancel"]["wait"])
    while time.time() < deadline:
        st = kernel_status(cfg, kref)
        if is_finished(st):
            print(f"{kref}: {st}")
            if args.fetch:
                cmd_fetch(cfg, root, run, args.dest, False)
            return 0
        time.sleep(min(5.0, max(0.05, deadline - time.time())))
    print(f"{kref} has not stopped yet (`kgr status {run}`).")
    return 0


def cmd_runs(cfg: dict, root: Path, args) -> int:
    states = sess.all_states(root)
    if not states:
        print("no runs pushed from this project yet")
        return 0
    print(f"{'run':<22} {'status':<20} {'gpu':<16} {'pushed (UTC)':<20} session")
    for s in states:
        st = "-" if args.no_status else (kernel_status(cfg, s.get("kernel") or kernel_ref(cfg, s["run"])) or "?")
        print(f"{s['run']:<22} {st:<20} {s.get('accelerator', '-'):<16} "
              f"{(s.get('pushed_at') or '-')[:19].replace('T', ' '):<20} {s.get('session_id') or '-'}")
    return 0


def cmd_logs(cfg: dict, root: Path, args) -> int:
    run = args.run
    if args.follow:
        return cmd_watch(cfg, root, run)
    text = fetch_log(cfg, kernel_ref(cfg, run))
    if text is None:
        die("could not read the log (does the run exist? is it finished? try `kgr watch`)")
    lines = sess.log_lines(text)
    if args.tail:
        lines = lines[-args.tail:]
    body = "\n".join(lines) + "\n"
    if args.save is not None:
        target = Path(args.save or f"{run}.kaggle.log")
        target.write_text(body, encoding="utf-8")
        print(f"wrote {target}  ({len(lines)} lines)")
    else:
        sys.stdout.write(body)
    return 0


# --------------------------------------------------------------------------
# commands: one-off scripts and a smoke test

def make_exec_namespace(args, run: str) -> argparse.Namespace:
    return argparse.Namespace(run=run, extra=None, resume_from=None, resume=False,
                              accelerator=None, skip_hooks=True, dry_run=args.dry_run)


def cmd_exec(args) -> int:
    cfg_path = find_config(args.config, required=False)
    cfg = merge_config(cfg_path)
    root = cfg_path.parent if cfg_path else Path.cwd().resolve()
    k = cfg["kaggle"]
    k["user"] = args.user or os.environ.get("KGR_USER") or k["user"]
    k["project"] = args.project or k["project"] or slugify(root.name) or "scratch"
    if args.gpu:
        k["accelerator"] = args.gpu
    if args.internet:
        k["internet"] = True
    k["datasets"] = [*k["datasets"], *args.dataset]
    k["competitions"] = [*k["competitions"], *args.competition]
    validate_config(cfg, cfg_path or "kgr exec", need_command=False)

    script = Path(args.script).resolve()
    if not script.is_file():
        die(f"no such script: {args.script}")
    try:
        rel = script.relative_to(root).as_posix()
    except ValueError:
        die(f"{script} is outside {root}; run kgr from a directory that contains it")
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    cfg["bundle"]["root"] = "."
    cfg["bundle"]["include"] = [rel, *args.include]
    cfg["run"]["command"] = f"python -u {shlex.quote(rel)} {{extra}}"
    cfg["run"]["default_extra"] = shlex.join(rest)
    cfg["run"]["output_dir"] = "outputs"
    run = args.name or script.stem

    ns = make_exec_namespace(args, run)
    cmd_push(cfg, root, ns)
    if args.dry_run or args.no_wait:
        return 0
    rc = cmd_wait(cfg, root, run, args.interval)
    # Everything the script left in $KGR_OUTPUT_DIR (= /kaggle/working/outputs).
    frc = cmd_fetch(cfg, root, run, args.dest or "outputs", True)
    return rc or frc


PROBE_SCRIPT = '''\
import os
print("kgr probe: kernel is alive")
print("cpus:", os.cpu_count())
'''


def cmd_probe(args) -> int:
    """A throwaway CPU kernel: shows the session report Kaggle really gives."""
    cfg = merge_config(find_config(args.config, required=False))
    cfg["kaggle"].update(user=args.user or os.environ.get("KGR_USER") or cfg["kaggle"]["user"],
                         project="kgr-probe", accelerator="none", competitions=[], datasets=[])
    validate_config(cfg, "kgr probe", need_command=False)
    root = Path(tempfile.mkdtemp(prefix="kgr-probe-"))
    (root / "probe.py").write_text(PROBE_SCRIPT, encoding="utf-8")
    cfg["bundle"].update(root=".", include=["probe.py"])
    cfg["run"].update(command="python -u probe.py", default_extra="", require_gpu=False)
    run = "probe"
    try:
        cmd_push(cfg, root, make_exec_namespace(args, run))
        if args.dry_run:
            return 0
        rc = cmd_wait(cfg, root, run, args.interval)
        state = capture_session(cfg, root, run, timeout=0) or sess.load_state(root, run)
        print("\n".join(describe_session(state)))
        if args.json and state.get("session"):
            print(json.dumps(state["session"], indent=2))
        return rc
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------
# commands: competitions

def cmd_data(cfg: dict, root: Path, args) -> int:
    comp = project_competition(cfg, args.competition)
    dest = (root / (args.dest or "data")).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if kaggle(cfg, "competitions", "download", comp, "-p", str(dest)).returncode != 0:
        die("download failed (accepted the competition rules on kaggle.com?)")
    if not args.keep_zip:
        for z in dest.glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest / comp)
            z.unlink()
            print(f"  extracted {z.name} -> {dest / comp}")
    return 0


def cmd_submit(cfg: dict, root: Path, args) -> int:
    comp = project_competition(cfg, args.competition)
    cmd = ["competitions", "submit", comp, "-m", args.message]
    if args.run:
        # Code competition: submit a file a kernel wrote.
        cmd += ["-k", kernel_ref(cfg, args.run), "-f", args.file or "submission.csv"]
        if args.version:
            cmd += ["-v", args.version]
    else:
        if not args.file:
            die("name the file to submit (or --run RUN for a code competition)")
        f = Path(args.file)
        if not f.is_file():
            die(f"no such file: {f}")
        cmd += ["-f", str(f)]
    return kaggle(cfg, *cmd).returncode


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="kgr", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", help=f"path to {CONFIG_NAME} (default: search upward)")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    p = sub.add_parser("init", help="write a starter kgr.toml")
    p.add_argument("--user", help="Kaggle username")
    p.add_argument("--project", help="project slug (default: directory name)")
    p.add_argument("--global", dest="glob", action="store_true",
                   help=f"write ~/{USER_CONFIG_NAME} with your username instead")
    p.add_argument("--force", action="store_true")

    sub.add_parser("doctor", help="check the kaggle CLI, credentials, quota and config")
    sub.add_parser("quota", help="weekly GPU/TPU quota left")

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
        p.add_argument("--resume", action="store_true",
                       help="shorthand: --resume-from this run's own previous kernel")
        p.add_argument("--accelerator", "--gpu", dest="accelerator",
                       help="override [kaggle] accelerator (t4, p100, l4, none, or the exact name)")
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

    p = with_run("info", "session id, container and URL of a run")
    p.add_argument("--json", action="store_true")
    p.add_argument("--refresh", action="store_true", help="re-read the kernel log")
    p.add_argument("--no-stream", action="store_true", help="do not attach to the live log")

    p = with_run("cancel", "stop a running kernel")
    p.add_argument("--session-id", metavar="ID", help="session id (or a URL ending in it), if kgr cannot find it")
    p.add_argument("--refresh", action="store_true", help="ignore the saved id and re-read the log")
    p.add_argument("--look", type=float, default=45, metavar="SECONDS",
                   help="how long to look for the session in the live log (default 45)")
    p.add_argument("--wait", type=float, default=None, metavar="SECONDS",
                   help="how long to wait for the kernel to stop (default [cancel] wait)")
    p.add_argument("--fetch", action="store_true", help="download whatever it saved once it stops")
    p.add_argument("--dest", help="with --fetch: override [fetch] dest")
    p.add_argument("--force", action="store_true", help="try even if the kernel looks finished")
    p.add_argument("--dry-run", action="store_true", help="find the session, send nothing")

    p = sub.add_parser("runs", help="runs pushed from this project")
    p.add_argument("--no-status", action="store_true", help="skip the live status lookup")

    p = with_run("logs", "print or save the kernel log")
    p.add_argument("-f", "--follow", action="store_true", help="stream it live (same as watch)")
    p.add_argument("--tail", type=int, metavar="N", help="only the last N lines")
    p.add_argument("--save", nargs="?", const="", metavar="FILE", help="write to FILE (default <run>.kaggle.log)")

    p = sub.add_parser("exec", help="run one script on Kaggle without a kgr.toml")
    p.add_argument("--name", help="run name (default: the script's name)")
    p.add_argument("--gpu", "--accelerator", dest="gpu", help="t4, p100, l4, none, or the exact name")
    p.add_argument("--user", help="Kaggle username (else $KGR_USER or ~/.kgr.toml)")
    p.add_argument("--project", help="slug prefix (default: directory name)")
    p.add_argument("--include", action="append", default=[], metavar="GLOB",
                   help="more files to upload next to the script (repeatable)")
    p.add_argument("--dataset", action="append", default=[], metavar="USER/NAME")
    p.add_argument("--competition", action="append", default=[], metavar="SLUG")
    p.add_argument("--internet", action="store_true")
    p.add_argument("--no-wait", action="store_true", help="push and return")
    p.add_argument("--interval", type=float, default=30)
    p.add_argument("--dest", help="where to put <run>/ with the output (default ./outputs)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("script", help="python file to run (written output goes to $KGR_OUTPUT_DIR)")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="arguments for the script")

    p = sub.add_parser("probe", help="throwaway CPU kernel: what does Kaggle report about a session?")
    p.add_argument("--user")
    p.add_argument("--interval", type=float, default=15)
    p.add_argument("--json", action="store_true", help="print the full report")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("data", help="download and unzip competition data")
    p.add_argument("competition", nargs="?", help="slug (default: first of [kaggle] competitions)")
    p.add_argument("--dest", help="directory (default ./data)")
    p.add_argument("--keep-zip", action="store_true")

    p = sub.add_parser("submit", help="submit a file (or a kernel's output) to a competition")
    p.add_argument("competition", nargs="?", help="slug (default: first of [kaggle] competitions)")
    p.add_argument("file", nargs="?", help="file to upload")
    p.add_argument("-m", "--message", required=True)
    p.add_argument("--run", help="code competition: submit this run's kernel output")
    p.add_argument("--version", help="with --run: kernel version")

    p = sub.add_parser("submissions", help="your submissions to a competition")
    p.add_argument("competition", nargs="?")

    args = ap.parse_args(argv)
    if args.cmd == "init":
        return cmd_init(args)
    if args.cmd == "doctor":
        return cmd_doctor(args)
    if args.cmd == "exec":
        return cmd_exec(args)
    if args.cmd == "probe":
        return cmd_probe(args)

    if args.cmd in ("quota", "data", "submit", "submissions"):
        # Usable in a bare competition directory: no kgr.toml, no username.
        cfg_path = find_config(args.config, required=False)
        cfg = merge_config(cfg_path)
        root = cfg_path.parent if cfg_path else Path.cwd().resolve()
        if args.cmd == "quota":
            return kaggle(cfg, "quota").returncode
        if args.cmd == "data":
            return cmd_data(cfg, root, args)
        if args.cmd == "submit":
            if args.run:
                validate_config(cfg, cfg_path or "kgr submit --run", need_command=False)
            return cmd_submit(cfg, root, args)
        return kaggle(cfg, "competitions", "submissions",
                      project_competition(cfg, args.competition)).returncode

    cfg_path = find_config(args.config, required=False)
    if cfg_path is None and args.cmd in BARE_OK and getattr(args, "run", None):
        # `kgr exec` leaves no kgr.toml, only .kgr/runs/<run>.json: enough to
        # status / info / cancel / fetch the run from the same directory.
        root = Path.cwd().resolve()
        state = sess.load_state(root, args.run)
        if state:
            cfg = merge_config(None)
            cfg["kaggle"]["project"] = state.get("project") or project_from_kernel(state, args.run) or slugify(root.name)
            validate_config(cfg, "this directory's .kgr state", need_command=False)
            return run_command(args, cfg, root)
    if cfg_path is None:
        find_config(args.config)          # dies with the usual message
    cfg = load_config(cfg_path, need_command=args.cmd in ("push", "run"))
    return run_command(args, cfg, cfg_path.parent)


def project_from_kernel(state: dict, run: str) -> str:
    slug = (state.get("kernel") or "").split("/", 1)[-1]
    suffix = "-" + slugify(run)
    return slug[:-len(suffix)] if slug.endswith(suffix) else ""


BARE_OK = ("wait", "watch", "status", "fetch", "ref", "info", "cancel", "logs")


def run_command(args, cfg: dict, root: Path) -> int:

    if args.cmd == "push":
        cmd_push(cfg, root, args)
        return 0
    if args.cmd == "run":
        cmd_push(cfg, root, args)
        if args.dry_run:
            return 0
        rc = cmd_wait(cfg, root, args.run, args.interval)
        # Fetch even after an ERROR: whatever the job saved before it died
        # (or before the 9 h wall) is still in the output.
        frc = cmd_fetch(cfg, root, args.run, args.dest, args.all)
        return rc or frc
    if args.cmd == "wait":
        return cmd_wait(cfg, root, args.run, args.interval)
    if args.cmd == "watch":
        return cmd_watch(cfg, root, args.run)
    if args.cmd == "status":
        return cmd_status(cfg, root, args.run)
    if args.cmd == "fetch":
        return cmd_fetch(cfg, root, args.run, args.dest, args.all)
    if args.cmd == "ref":
        print(kernel_ref(cfg, args.run))
        return 0
    if args.cmd == "info":
        return cmd_info(cfg, root, args)
    if args.cmd == "cancel":
        return cmd_cancel(cfg, root, args)
    if args.cmd == "runs":
        return cmd_runs(cfg, root, args)
    if args.cmd == "logs":
        return cmd_logs(cfg, root, args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
