# kaggle-gpu-runner (`kgr`)

Run a training job on a **Kaggle GPU kernel** from your terminal: bundle the
code, push it, wait, and pull the checkpoints back into your project. Nothing
here knows about your model — everything project-specific lives in one
`kgr.toml`.

```
kgr push resnet_v3          # bundle -> private dataset -> script kernel (starts the run)
kgr wait resnet_v3          # poll until COMPLETE / ERROR
kgr fetch resnet_v3         # checkpoints -> ./outputs
kgr run  resnet_v3          # all three
kgr cancel resnet_v3        # hard stop; files already on disk are kept
kgr exec train.py -- --lr 3e-4   # one script, no kgr.toml at all
```

Why Kaggle for long runs: a kernel runs **non-interactively for up to 9 h on a
GPU** (T4 ×2 / P100 / L4, ~30 GPU-hours a week), so there is no browser
session to keep alive. For interactive or session-based work see the sibling
project [colab-gpu-runner](https://github.com/eple12/colab-gpu-runner).

## Quick start (new machine, new project)

```bash
pipx install git+https://github.com/eple12/kaggle-gpu-runner   # or pip install -U ...
kgr doctor                       # kaggle CLI, token, quota, cancel support
cd my-competition
kgr exec --gpu t4 --competition titanic train.py -- --epochs 20
```

`kgr` takes your username from the `kaggle` CLI login (`kaggle config view`),
so a bare script needs no config. `kgr exec` uploads just that script (plus
`--include` globs), runs it on Kaggle, waits, and leaves everything the script
wrote to `$KGR_OUTPUT_DIR` in `./outputs/<name>/`. For a project you run
repeatedly, `kgr init` writes a `kgr.toml` (`kgr init --global --user NAME`
stores the username once for every project).

## How it works

```
your project                          Kaggle
────────────                          ──────
kgr.toml ─┐
code  ────┼─ kgr push ─► dataset  <user>/<project>-code   (new version each push)
          │            ► kernel   <user>/<project>-<run>  (script, GPU, private)
          │                 │  kernel_runner.py:
          │                 │   0. print KGR_SESSION_INFO (ids for kgr cancel)
          │                 │   1. GPU check (abort on CPU / unsupported card)
          │                 │   2. find THIS push's bundle (build id)
          │                 │   3. unpack to /tmp, link output_dir -> /kaggle/working
          │                 │   4. copy resume files from --resume-from kernel
          │                 │   5. run [run] command, stream the log
outputs/ ◄── kgr fetch ◄────┘  /kaggle/working = kernel output
```

## Install

Requirements: Python 3.9+ locally (3.11+ needs no extra packages), a Kaggle
account, and the `kaggle` CLI (installed as a dependency).

Pick one:

```bash
# 1. as a tool, available everywhere
pipx install git+https://github.com/eple12/kaggle-gpu-runner
# or: uv tool install git+https://github.com/eple12/kaggle-gpu-runner

# 2. into the project's own environment
pip install git+https://github.com/eple12/kaggle-gpu-runner

# 3. vendored, no install (pin a commit, edit freely)
git submodule add https://github.com/eple12/kaggle-gpu-runner vendor/kaggle-gpu-runner
python vendor/kaggle-gpu-runner/kgr.py --help
```

### One-time Kaggle setup

1. Kaggle → Settings → API → **Create New Token**, save it as
   `~/.kaggle/kaggle.json` (`chmod 600` it on Linux/macOS).
2. Kaggle → Settings → **Phone Verification**. Without it the GPU request is
   *silently ignored* — the kernel starts fine and trains on a CPU. Use a
   carrier number; VoIP numbers are commonly rejected.

## Integrating into a project

### 1. Make the training script resumable

`kgr` only needs a command. The contract:

| Your script should | Because |
| --- | --- |
| write checkpoints under the directory passed as `{output_dir}` | only `/kaggle/working` survives the kernel |
| on start, continue from a checkpoint already in that directory | that is how `--resume-from` crosses the 9 h wall |
| save periodically and atomically (`tmp` + `os.replace`) | a kernel killed at the limit keeps whatever is on disk |
| name its files with the run name (`{run}_best.pt`, `{run}_state.pt`) | `resume_glob` / `fetch.glob` select files by it |
| exit non-zero on failure | the kernel shows ERROR and `kgr wait` returns 1 |

If you resume RL / anything with a replay buffer or optimizer state, save
that too — reloading only the weights into an empty buffer tends to undo
the progress.

[`examples/minimal/train.py`](examples/minimal/train.py) is a 40-line
reference.

### 2. Add `kgr.toml` at the project root

```bash
kgr init          # writes a commented template
```

```toml
[kaggle]
user = "your-kaggle-username"
project = "myproj"                 # dataset: myproj-code, kernels: myproj-<run>
accelerator = "NvidiaTeslaT4"      # case-sensitive; "none" = CPU kernel
internet = false                   # true needs phone verification
datasets = []                      # large data you uploaded separately, e.g. ["me/imagenet-subset"]

[bundle]
include = ["train.py", "src", "configs", "data/small"]
exclude = ["__pycache__", "*.pyc", ".git", "outputs"]

[run]
command = "python -u train.py --run-name {run} --out {output_dir} --device {device} --workers {cpus_minus_1} {extra}"
default_extra = "--epochs 50"
output_dir = "outputs"
resume_glob = "{run}*"
setup = []                         # e.g. ["pip install -q timm"] (needs internet = true)

[fetch]
dest = "outputs"
glob = "{run}*"

[hooks]
pre_push = ["python -m pytest -q tests/smoke"]   # don't spend GPU quota on a broken edit
```

Placeholders available in `command` and `setup`:

| placeholder | value on Kaggle |
| --- | --- |
| `{run}` | the run name given to `kgr push` |
| `{extra}` | `--extra "..."`, else `default_extra` |
| `{output_dir}` | absolute path under `/kaggle/working` |
| `{device}` | `cuda` or `cpu` |
| `{cpus}`, `{cpus_minus_1}` | CPU count (e.g. for dataloader / eval workers) |
| `{bundle}` | where the code was unpacked |
| `{input}` | `/kaggle/input` (attached `datasets` are under it) |

The same values are exported as `KGR_RUN`, `KGR_DEVICE`, `KGR_OUTPUT_DIR`,
`KGR_BUNDLE`, `KGR_INPUT` and `KGR_RESUMED` (`1` if files were carried over).

Add `outputs/` (or your `fetch.dest`) to `.gitignore`.

### 3. Run

```bash
kgr push exp_01 --extra "--lr 3e-4"
kgr watch exp_01                     # live log; Ctrl-C only stops watching
kgr wait exp_01 && kgr fetch exp_01
```

Continue a run that hit the time limit — the previous kernel's output is
attached as an input and matching files are copied into `output_dir` before
the command starts:

```bash
kgr push exp_01 --resume-from your-kaggle-username/myproj-exp-01
```

Useful flags: `--dry-run` (stage bundle + metadata locally, send nothing —
inspect what would be uploaded), `--skip-hooks`, `--accelerator NvidiaL4`,
`kgr fetch --all` (whole output tree to `<dest>/<run>/`), `kgr ref <run>`.

### Managing a run

| command | what it does |
| --- | --- |
| `kgr runs` | every run pushed from this project, with live status and session id |
| `kgr info RUN` | kernel URL, status, GPU, and the session / container ids the kernel reported (`--json`, `--refresh`) |
| `kgr cancel RUN` | stop the kernel (see below); `--fetch` downloads whatever it saved |
| `kgr logs RUN` | print the log; `--tail 50`, `--save [FILE]`, `-f` to follow |
| `kgr quota` | weekly GPU / TPU hours left |
| `kgr push RUN --resume` | continue from this run's own previous output (`--resume-from` for another kernel) |
| `kgr push RUN --gpu t4` | accelerator shortcuts `t4`, `p100`, `l4`, `none` for `--accelerator` |

### Cancelling

The `kaggle` CLI cannot cancel a kernel, and nothing it prints names the
*session* of the run in progress. The API underneath can
(`CancelKernelSession(kernel_session_id)`), so `kgr` closes the gap from the
inside: the first thing every kernel does is print one line

```
KGR_SESSION_INFO {"run": ..., "hostname": ..., "containers": [...], "id_candidates": {...}, "env": {...}}
```

(also saved as `kgr_session.json` in the output). It holds the `KAGGLE_*`
environment (tokens and keys redacted), the hostname, container ids and every
numeric id-like variable. `kgr wait` / `watch` / `info` read that line back,
store it in `.kgr/runs/<run>.json`, and `kgr cancel` sends the session id to
the API, then waits until the status is `CANCEL_*`.

```bash
kgr cancel exp_01                      # id found automatically
kgr cancel exp_01 --session-id 123456  # or a URL ending in the id
kgr cancel exp_01 --fetch              # stop, then download what it saved
```

- Which variable is the session id is decided by name
  (`KAGGLE_KERNEL_SESSION_ID`, `KAGGLE_SESSION_ID`, ...). If none matches,
  `kgr` never guesses between several candidates: it lists them, and you pass
  `--session-id`, or set `[cancel] session_env = "NAME"` once. **`kgr probe`
  runs a throwaway CPU kernel and prints exactly what Kaggle reports** -- run
  it once to see which variable is yours.
- What a real Batch kernel exposes (`kgr probe`): no variable named like an id,
  but `KAGGLE_CONTAINER_NAME=kaggle_<token>-<number>-webtier`. `kgr` takes that
  number as the session id when nothing better is reported -- **verified**: the
  API cancelled the kernel with exactly that number.
- `kgr exec` leaves no `kgr.toml`, only `.kgr/runs/`; `status`, `info`, `cancel`,
  `wait`, `fetch`, `logs` work from that same directory.
- A kernel still `QUEUED` has not started, so it has printed nothing; use the
  Stop button on its page (`kgr cancel` prints the link).
- **Cancel is a hard stop.** Verified on a live kernel: the session ended
  `CANCEL_REQUESTED` -> `CANCEL_ACKNOWLEDGED` within seconds, the job got no
  SIGTERM and no summary was written, but files it had already written to the
  output directory were kept and `kgr fetch` downloads them. So checkpoint
  periodically and atomically (the contract above); do not rely on a
  last-moment save. (The runner still forwards SIGTERM/SIGINT if it ever
  receives one, best effort.)
- Needs `kaggle>=2.x` (its SDK has the cancel call); `kgr doctor` checks.

### Competitions

```bash
kgr data titanic                       # download + unzip into ./data/titanic
kgr exec --competition titanic train.py
kgr submit titanic submission.csv -m "lr 3e-4"
kgr submit titanic --run exp_01 -m "from kernel"   # code competitions: a kernel's output
kgr submissions titanic
```

Put `competitions = ["titanic"]` under `[kaggle]` to attach the data to every
run of the project (accept the rules on kaggle.com first); it then lives under
`{input}` (`/kaggle/input`) -- the runner prints the real layout each run. The
same section takes `kernels = [...]` and `models = [...]`. These commands work
in a bare directory, without a `kgr.toml`.

### Driving it from other tooling

Every command returns a meaningful exit code (`wait`: 0 COMPLETE, 1 ERROR or
cancelled, 2 status unreadable; `cancel`: 0 stopped or already finished, 1 could
not cancel), so wrapping it is plain shell:

```bash
kgr run "$RUN" --extra "$FLAGS" \
  && python evaluate.py "outputs/${RUN}_best.pt" \
  && cp "outputs/${RUN}_best.pt" models/deployed.pt
```

On Windows it runs natively (`kgr.exe` via pip, or `python kgr.py`); WSL is
not required.

## Things Kaggle does that `kgr` works around

These each cost a real run before they were handled:

- **Dataset publishing is asynchronous.** A kernel pushed before the new
  version is ready gets no code or the previous version's code. `kgr` waits
  for the version number to advance, and the kernel checks a per-push build
  id, failing loudly instead of training stale code.
- **Mount paths vary** (`/kaggle/input/<slug>/…` on one push,
  `/kaggle/input/datasets/<slug>/…` on the next), and uploaded archives are
  sometimes expanded, sometimes not. The runner searches a few levels deep for
  either form and prints the input layout every run.
- **GPU requests can silently downgrade.** Metadata `enable_gpu` alone has
  come back as a CPU image, so `--accelerator` is also passed on push; the
  value is case-sensitive (`nvidiaTeslaT4` quietly becomes a generic shape).
  With `require_gpu = true` the kernel aborts on CPU instead of burning hours.
- **An older GPU may be assigned** (P100, compute capability 6.0) that current
  torch CUDA wheels cannot run. `min_cuda_capability` stops the run up front
  with a clear message rather than a traceback from the first CUDA op.
- **`nvidia-smi` is missing** on CPU images; diagnostics never crash the run.
- **Output is only `/kaggle/working`.** The code is unpacked to `/tmp` and
  only `output_dir` is linked in, so fetched output is your artifacts, not a
  copy of the repo.
- **Probing the GPU with torch in the runner** would hold a CUDA context for
  the whole run, so it happens in a short-lived subprocess.

## Config reference

| key | default | meaning |
| --- | --- | --- |
| `kaggle.user` | — | Kaggle username |
| `kaggle.project` | — | slug prefix |
| `kaggle.bin` | `kaggle` | command used to call the CLI (also `$KGR_KAGGLE`) |
| `kaggle.accelerator` | `NvidiaTeslaT4` | `NvidiaTeslaT4`, `NvidiaTeslaP100`, `NvidiaL4`, … or `none` |
| `kaggle.internet` | `false` | kernel internet access |
| `kaggle.private` | `true` | kernel visibility (the code dataset is always private) |
| `kaggle.timeout` | `0` | run-time cap in seconds, `0` = Kaggle's limit |
| `kaggle.datasets` | `[]` | extra dataset refs to attach |
| `kaggle.dataset_wait` | `900` | seconds to wait for the code dataset |
| `bundle.root` | `.` | base directory for `include` |
| `bundle.include` | — | globs of files/dirs to upload |
| `bundle.exclude` | caches, `.git`, venvs | patterns matched against each path and path component |
| `run.workdir` | `.` | command cwd inside the bundle |
| `run.command` | — | the job |
| `run.default_extra` | `""` | `{extra}` when `--extra` is not given |
| `run.output_dir` | `outputs` | relative dir that becomes kernel output |
| `run.resume_glob` | `{run}*` | files carried over from `--resume-from` |
| `run.require_gpu` | `true` | abort when no GPU is attached |
| `run.min_cuda_capability` | `7.0` | abort below this (`""` disables) |
| `run.setup` | `[]` | shell commands run before `command` |
| `kaggle.competitions` | `[]` | competition data to attach (`titanic`) |
| `kaggle.kernels` | `[]` | other kernels whose output the job reads |
| `kaggle.models` | `[]` | model refs to attach |
| `cancel.session_env` | `""` | kernel variable that holds the session id, when it is not auto-detected |
| `cancel.wait` | `90` | seconds `kgr cancel` waits for the stop |
| `fetch.dest` | `outputs` | local destination |
| `fetch.glob` | `{run}*` | files to copy out of the downloaded output |
| `hooks.pre_push` | `[]` | local commands that must succeed before pushing |

## Windows

`kgr` runs natively. The kaggle CLI's output is decoded as UTF-8 regardless of
the console codepage (cp949 / cp1252 used to garble or crash log reading).
Settings shared by all projects go in `~/.kgr.toml`.

## Development

```bash
python -m unittest discover -s tests -v
```

The tests use `tests/fake_kaggle.py`, which implements the CLI calls `kgr`
makes and really executes pushed kernels locally (with inputs laid out like
`/kaggle/input`), so push → run → resume → fetch is covered end to end
without an account, including `cancel` (the one API call is replaced by a stub)
and the SIGTERM hand-over. The runner uses `bash` and symlinks, so run them on
Linux, macOS or WSL; on plain Windows the tests that execute a kernel are
skipped.

## License

MIT
