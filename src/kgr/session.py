"""Finding, remembering and cancelling a running Kaggle kernel session.

The Kaggle CLI can push, poll and download a kernel but has no `cancel`, and
nothing it prints names the *session* of the current run. The API behind it
does have `CancelKernelSession(kernel_session_id)`. The missing piece is the
id, so the kernel reports what identifies it (see kernel_runner.session_info)
as a `KGR_SESSION_INFO {json}` log line; this module reads that line back,
keeps it under `<project>/.kgr/runs/<run>.json`, and calls the API.

Nothing here talks to Kaggle except `sdk_cancel`, which tests replace.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

MARK = "KGR_SESSION_INFO"
STATE_DIR = Path(".kgr") / "runs"

# Environment variables that are known (or likely) to hold the session id, in
# priority order. The first one present in a kernel's report wins; a project
# can name another with `[cancel] session_env`.
SESSION_ENV_NAMES = (
    "KAGGLE_KERNEL_SESSION_ID",
    "KAGGLE_SESSION_ID",
    "KAGGLE_KERNEL_RUN_ID",
    "KAGGLE_KERNEL_ID",
)

CONTAINER_NAME_ID = re.compile(r"-(\d{5,})-")

TERMINAL = ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL")


# --------------------------------------------------------------------------
# reading the kernel log

def log_lines(raw: str) -> list[str]:
    """Plain lines out of `kaggle kernels logs`, which prints either text or a
    JSON array of {stream_name, time, data} events."""
    raw = raw or ""
    stripped = raw.strip()
    if stripped.startswith("["):
        try:
            events = json.loads(stripped)
            joined = "".join(e.get("data", "") for e in events if isinstance(e, dict))
            return joined.splitlines()
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    return raw.splitlines()


def parse_session_line(line: str) -> dict | None:
    pos = line.find(MARK)
    if pos < 0:
        return None
    body = line[pos + len(MARK):].strip()
    for candidate in (body, body.replace('\\"', '"')):
        try:
            info = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(info, dict):
            return info
    return None


def find_session_info(raw_log: str, build_id: str | None = None) -> dict | None:
    """The last session report in the log (a log can hold several when a
    kernel restarts); with build_id, only a report from that push."""
    found = None
    for line in log_lines(raw_log):
        info = parse_session_line(line)
        if info and (build_id is None or info.get("build_id") == build_id):
            found = info
    return found


# --------------------------------------------------------------------------
# choosing the session id

def pick_session_id(info: dict, preferred_env: str = "") -> tuple[int | None, str]:
    """(session id, where it came from).

    Known variable names first, then the project's own `session_env`. A lone
    numeric candidate is accepted as a guess; several are ambiguous and the
    caller must be told which to use (`--session-id`).
    """
    env = info.get("env") or {}
    cands = {k: int(v) for k, v in (info.get("id_candidates") or {}).items()
             if str(v).lstrip("-").isdigit()}
    for name in ([preferred_env] if preferred_env else []) + list(SESSION_ENV_NAMES):
        raw = cands.get(name)
        if raw is None and str(env.get(name, "")).isdigit():
            raw = int(env[name])
        if raw is not None:
            return raw, f"${name}"
    # Observed on a real Batch kernel (no variable holds the id): the
    # container name embeds it, kaggle_<token>-<number>-webtier.
    m = CONTAINER_NAME_ID.search(str(env.get("KAGGLE_CONTAINER_NAME", "")))
    if m:
        return int(m.group(1)), "$KAGGLE_CONTAINER_NAME (the number in the name)"
    if len(cands) == 1:
        (name, value), = cands.items()
        return value, f"${name} (only numeric id the kernel reported -- a guess)"
    return None, ""


def describe_candidates(info: dict) -> str:
    cands = info.get("id_candidates") or {}
    if not cands:
        return "(the kernel reported no numeric ids)"
    return ", ".join(f"{k}={v}" for k, v in sorted(cands.items()))


# --------------------------------------------------------------------------
# per-run state under <project>/.kgr/runs

def state_path(root: Path, run: str) -> Path:
    return root / STATE_DIR / f"{run}.json"


def load_state(root: Path, run: str) -> dict:
    try:
        return json.loads(state_path(root, run).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(root: Path, run: str, **fields) -> dict:
    state = load_state(root, run)
    state.update({k: v for k, v in fields.items() if v is not None})
    state["run"] = run
    state["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p = state_path(root, run)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(p)
    return state


def all_states(root: Path) -> list[dict]:
    out = []
    for p in sorted((root / STATE_DIR).glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(out, key=lambda s: s.get("pushed_at") or s.get("updated_at") or "")


# --------------------------------------------------------------------------
# the one call that reaches Kaggle

def sdk_cancel(session_id: int) -> str:
    """Ask Kaggle to stop a session. Returns "" on success, else the reason.

    Goes through the `kaggle` package's own client (so it uses the same
    credentials as the CLI) because the CLI has no cancel command.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.kernels.types.kernels_api_service import ApiCancelKernelSessionRequest
    except ImportError:
        return ("this kaggle package has no CancelKernelSession; "
                "`pip install -U kaggle` (2.x)")
    try:
        api = KaggleApi()
        api.authenticate()
        req = ApiCancelKernelSessionRequest()
        req.kernel_session_id = int(session_id)
        with api.build_kaggle_client() as client:
            resp = client.kernels.kernels_api_client.cancel_kernel_session(req)
        return getattr(resp, "error_message", "") or ""
    except Exception as e:  # noqa: BLE001 -- surface any API/auth failure as text
        return f"{type(e).__name__}: {e}"


_NUM = re.compile(r"^\d+$")


def parse_session_id(text: str) -> int:
    """`--session-id` accepts the bare number or a Kaggle URL ending in it."""
    text = text.strip().rstrip("/")
    tail = text.rsplit("/", 1)[-1]
    if not _NUM.match(tail):
        raise ValueError(f"not a session id: {text!r}")
    return int(tail)
