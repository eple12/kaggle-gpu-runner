"""End-to-end tests against a fake Kaggle that runs kernels locally.

    python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kgr import cli, session  # noqa: E402

FAKE = REPO / "tests" / "fake_kaggle.py"
TRAIN = REPO / "examples" / "minimal" / "train.py"


def _can_symlink():
    d = Path(tempfile.mkdtemp(prefix="kgr-sym-"))
    try:
        (d / "t").symlink_to(d, target_is_directory=True)
        return True
    except OSError:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


# The kernel runner links its output dir into /kaggle/working with a symlink;
# Windows only allows that in Developer Mode / as admin. Kaggle runs Linux.
needs_symlink = unittest.skipUnless(_can_symlink(), "kernel runner needs symlinks (Linux/macOS/WSL)")


def fake_bin():
    parts = [sys.executable, str(FAKE)]
    return subprocess.list2cmdline(parts) if os.name == "nt" else shlex.join(parts)


class KgrTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kgr-test-"))
        self.proj = self.tmp / "proj"
        self.proj.mkdir()
        shutil.copy(TRAIN, self.proj / "train.py")
        (self.proj / "outputs").mkdir()
        (self.proj / "outputs" / "stale_local_file.bin").write_text("must not be bundled")
        self.home = self.tmp / "kaggle_home"
        self.home.mkdir()
        self._env = {k: os.environ.get(k) for k in ("FAKE_KAGGLE_HOME", "KGR_USER_CONFIG",
                                                    "KGR_KAGGLE", "KGR_USER")}
        os.environ["FAKE_KAGGLE_HOME"] = str(self.home)
        os.environ["KGR_USER_CONFIG"] = str(self.tmp / "no-such-user-config.toml")
        os.environ.pop("KGR_KAGGLE", None)
        os.environ.pop("KGR_USER", None)
        self.cfg_path = self.proj / "kgr.toml"
        self.write_config()
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_config(self, accelerator="none", hooks=(), extra=""):
        self.cfg_path.write_text(f'''
[kaggle]
user = "tester"
project = "demo"
bin = {json.dumps(fake_bin())}
accelerator = "{accelerator}"
dataset_wait = 30
{extra}

[bundle]
include = ["train.py", "outputs"]
exclude = ["outputs"]

[run]
command = "python -u train.py --run-name {{run}} --out {{output_dir}} --device {{device}} {{extra}}"
default_extra = "--steps 300"
output_dir = "outputs"

[fetch]
dest = "fetched"

[hooks]
pre_push = {json.dumps(list(hooks))}
''', encoding="utf-8")

    def kgr(self, *argv, config=True):
        out = io.StringIO()
        full = ["-c", str(self.cfg_path), *argv] if config else list(argv)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                rc = cli.main(full)
            except SystemExit as e:
                rc = e.code
        return rc, out.getvalue()

    def fetched_state(self, run):
        return json.loads((self.proj / "fetched" / f"{run}_state.json").read_text())

    def kernel_dir(self, slug):
        return self.home / "kernels" / slug

    def set_status(self, slug, status):
        (self.kernel_dir(slug) / "status").write_text(status)

    # ------------------------------------------------------------------
    # the original behaviour

    def test_names_follow_kaggle_slug_rules(self):
        cfg = cli.load_config(self.cfg_path)
        self.assertEqual(cli.kernel_ref(cfg, "Spa_v26.k"), "tester/demo-spa-v26-k")
        self.assertEqual(cli.dataset_ref(cfg), "tester/demo-code")

    def test_dry_run_stages_bundle_and_gpu_metadata(self):
        self.write_config(accelerator="NvidiaTeslaT4")
        rc, out = self.kgr("push", "run_1", "--dry-run")
        self.assertEqual(rc, 0, out)
        stage = Path(out.rsplit("staged in ", 1)[1].split(",")[0].strip())
        meta = json.loads((stage / "kernel" / "kernel-metadata.json").read_text())
        self.assertEqual(meta["id"], "tester/demo-run-1")
        self.assertEqual(meta["title"], "demo run 1")
        self.assertEqual(meta["enable_gpu"], "true")
        self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
        with tarfile.open(stage / "data" / cli.BUNDLE_TAR) as tf:
            names = tf.getnames()
        self.assertIn("train.py", names)
        self.assertIn(cli.BUNDLE_MARKER, names)
        self.assertFalse(any("stale_local_file" in n for n in names), names)
        self.assertFalse((self.home / "kernels").exists(), "dry run must not push")
        shutil.rmtree(stage)

    @needs_symlink
    def test_run_then_resume_across_kernels(self):
        rc, out = self.kgr("run", "run_1", "--interval", "0.05")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.fetched_state("run_1")["step"], 300)
        self.assertTrue((self.proj / "fetched" / "run_1.kaggle.log").is_file())
        push_args = json.loads((self.kernel_dir("demo-run-1") / "push_args.json").read_text())
        self.assertNotIn("--accelerator", push_args, "CPU kernel must not request a GPU")

        # Second leg: attach the first kernel's output and train further.
        rc, out = self.kgr("run", "run_1", "--resume-from", "tester/demo-run-1",
                           "--extra", "--steps 600", "--interval", "0.05")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.fetched_state("run_1")["step"], 600)
        log = (self.proj / "fetched" / "run_1.kaggle.log").read_text(encoding="utf-8")
        self.assertIn("starting at step 300", log)
        self.assertIn("resumed files: ['run_1_state.json']", log)

    @needs_symlink
    def test_failing_job_reports_error_but_still_fetches(self):
        (self.proj / "train.py").write_text(
            "import sys, pathlib\n"
            "out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
            "(out / 'run_x_partial.bin').write_text('saved before dying')\n"
            "sys.exit(3)\n")
        rc, out = self.kgr("run", "run_x", "--interval", "0.05")
        self.assertEqual(rc, 1, out)
        self.assertIn("ERROR", out)
        self.assertTrue((self.proj / "fetched" / "run_x_partial.bin").is_file())

    def test_pre_push_hook_blocks_push(self):
        self.write_config(hooks=[f"{shlex.quote(Path(sys.executable).as_posix())} -c \"raise SystemExit(1)\""])
        rc, _ = self.kgr("push", "run_1")
        self.assertEqual(rc, 1)
        self.assertFalse((self.home / "kernels").exists())

    def test_runner_refuses_bundle_from_another_push(self):
        cfg = cli.load_config(self.cfg_path)
        inputs = self.tmp / "input"
        (inputs / "demo-code").mkdir(parents=True)
        cli.build_bundle(cfg, self.proj, inputs / "demo-code" / cli.BUNDLE_TAR, "old-build", "r")
        kernel = self.tmp / "kernel.py"
        kernel.write_text(cli.render_kernel(cfg, "r", "new-build", "", False))
        r = self.run_kernel(kernel, inputs)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not among the inputs", r.stdout + r.stderr)

    @needs_symlink
    def test_runner_finds_expanded_tree_and_ignores_stale_copies(self):
        cfg = cli.load_config(self.cfg_path)
        inputs = self.tmp / "input"
        for sub, build in (("demo-code/kgr_bundle", "good"), ("prev-kernel/tmp/copy", "stale")):
            d = inputs / sub
            d.mkdir(parents=True)
            tar = self.tmp / f"{build}.tar.gz"
            cli.build_bundle(cfg, self.proj, tar, build, "r")
            with tarfile.open(tar) as tf:
                tf.extractall(d, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
        kernel = self.tmp / "kernel.py"
        kernel.write_text(cli.render_kernel(cfg, "r", "good", "--steps 100", False))
        r = self.run_kernel(kernel, inputs)
        self.assertEqual(r.returncode, 0, (r.stdout or "") + (r.stderr or ""))
        self.assertIn("demo-code/kgr_bundle", r.stdout)
        summary = json.loads((self.tmp / "working" / "kgr_run_summary.json").read_text())
        self.assertEqual(summary["outputs"], ["outputs/r_state.json"])

    def run_kernel(self, kernel, inputs, **popen):
        env = {**os.environ, "KGR_INPUT_ROOT": str(inputs),
               "KGR_WORKING_ROOT": str(self.tmp / "working"),
               "KGR_BUNDLE_DIR": str(self.tmp / "bundle"),
               "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        return subprocess.run([sys.executable, str(kernel)], env=env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", **popen)

    # ------------------------------------------------------------------
    # session info and cancel

    @needs_symlink
    def test_kernel_reports_itself_without_leaking_secrets(self):
        rc, out = self.kgr("run", "run_1", "--interval", "0.05")
        self.assertEqual(rc, 0, out)
        log = (self.kernel_dir("demo-run-1") / "log.txt").read_text(encoding="utf-8")
        line = next(l for l in log.splitlines() if l.startswith(session.MARK))
        info = session.parse_session_line(line)
        self.assertEqual(info["run"], "run_1")
        self.assertEqual(info["env"]["KAGGLE_SESSION_ID"], "424242")
        self.assertEqual(info["env"]["KAGGLE_DATA_PROXY_TOKEN"], "<redacted>")
        self.assertNotIn("very-secret", log)
        self.assertEqual(info["id_candidates"]["KAGGLE_SESSION_ID"], 424242)

    def test_info_finds_and_saves_the_session_id(self):
        self.kgr("run", "run_1", "--interval", "0.05")
        rc, out = self.kgr("info", "run_1")
        self.assertEqual(rc, 0, out)
        self.assertIn("424242", out)
        self.assertIn("https://www.kaggle.com/code/tester/demo-run-1", out)
        state = session.load_state(self.proj, "run_1")
        self.assertEqual(state["session_id"], 424242)
        self.assertEqual(state["session_id_source"], "$KAGGLE_SESSION_ID")
        rc, out = self.kgr("info", "run_1", "--json")
        self.assertEqual(json.loads(out)["session_id"], 424242)

    def test_cancel_sends_the_session_id_and_waits_for_the_stop(self):
        self.kgr("run", "run_1", "--interval", "0.05")
        self.set_status("demo-run-1", "RUNNING")
        calls = []

        def fake_cancel(sid):
            calls.append(sid)
            self.set_status("demo-run-1", "CANCEL_ACKNOWLEDGED")
            return ""

        orig, session.sdk_cancel = session.sdk_cancel, fake_cancel
        try:
            rc, out = self.kgr("cancel", "run_1", "--wait", "5")
        finally:
            session.sdk_cancel = orig
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [424242])
        self.assertIn("CANCEL_ACKNOWLEDGED", out)
        self.assertIn("cancelled_at", session.load_state(self.proj, "run_1"))

    def test_cancel_accepts_an_explicit_id_or_url(self):
        self.kgr("run", "run_1", "--interval", "0.05")
        self.set_status("demo-run-1", "RUNNING")
        calls = []
        orig, session.sdk_cancel = session.sdk_cancel, lambda sid: calls.append(sid) or ""
        try:
            rc, out = self.kgr("cancel", "run_1", "--session-id",
                               "https://www.kaggle.com/code/x/y/edit/run/777", "--wait", "0")
        finally:
            session.sdk_cancel = orig
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [777])

    @needs_symlink
    def test_cancel_does_nothing_for_a_finished_kernel(self):
        self.kgr("run", "run_1", "--interval", "0.05")
        orig, session.sdk_cancel = session.sdk_cancel, lambda sid: self.fail("must not cancel")
        try:
            rc, out = self.kgr("cancel", "run_1")
        finally:
            session.sdk_cancel = orig
        self.assertEqual(rc, 0, out)
        self.assertIn("already COMPLETE", out)

    def test_cancel_reports_api_errors(self):
        self.kgr("run", "run_1", "--interval", "0.05")
        self.set_status("demo-run-1", "RUNNING")
        orig, session.sdk_cancel = session.sdk_cancel, lambda sid: "HTTPError: 403"
        try:
            rc, out = self.kgr("cancel", "run_1")
        finally:
            session.sdk_cancel = orig
        self.assertEqual(rc, 1)
        self.assertIn("403", out)
        self.assertIn("Manual stop", out)

    def test_cancel_without_a_report_says_so(self):
        self.kgr("run", "run_1", "--interval", "0.05")
        (self.kernel_dir("demo-run-1") / "log.txt").write_text("nothing useful\n")
        self.set_status("demo-run-1", "QUEUED")
        rc, out = self.kgr("cancel", "run_1", "--look", "0")
        self.assertEqual(rc, 1)
        self.assertIn("No session report yet", out)

    def test_ambiguous_ids_are_never_guessed(self):
        info = {"id_candidates": {"A_RUN_ID": 1, "B_JOB_ID": 2}, "env": {}}
        self.assertEqual(session.pick_session_id(info), (None, ""))
        self.assertEqual(session.pick_session_id(info, "B_JOB_ID"), (2, "$B_JOB_ID"))
        only = {"id_candidates": {"X_KERNEL_THING": 9}, "env": {}}
        sid, why = session.pick_session_id(only)
        self.assertEqual(sid, 9)
        self.assertIn("guess", why)
        named = {"id_candidates": {"KAGGLE_KERNEL_ID": 5, "KAGGLE_SESSION_ID": 6}, "env": {}}
        self.assertEqual(session.pick_session_id(named)[0], 6)

    def test_session_id_from_the_real_kaggle_container_name(self):
        # Captured from a real Batch kernel: no *_ID variable, only this name.
        info = {"id_candidates": {}, "env": {
            "KAGGLE_CONTAINER_NAME":
                "kaggle_JBY9rf73f9RzrRwdUZMOm6A2mKfWgvnTvAXnvk1k-366462278-webtier",
            "KAGGLE_KERNEL_RUN_TYPE": "Batch"}}
        sid, why = session.pick_session_id(info)
        self.assertEqual(sid, 366462278)
        self.assertIn("KAGGLE_CONTAINER_NAME", why)
        info["env"]["KAGGLE_SESSION_ID"] = "7"        # an explicit variable still wins
        self.assertEqual(session.pick_session_id(info)[0], 7)

    def test_cancel_requested_is_not_the_end(self):
        for st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
            self.assertTrue(cli.is_finished(st), st)
        for st in ("CANCEL_REQUESTED", "RUNNING", "QUEUED", None):
            self.assertFalse(cli.is_finished(st), st)

    def test_log_parsing_handles_json_event_logs(self):
        info = {"run": "r", "build_id": "b", "env": {"KAGGLE_SESSION_ID": "5"}}
        events = [{"stream_name": "stdout", "time": 1, "data": "hello\n"},
                  {"stream_name": "stdout", "time": 2,
                   "data": f"{session.MARK} {json.dumps(info)}\n"}]
        found = session.find_session_info(json.dumps(events))
        self.assertEqual(found["run"], "r")
        self.assertIsNone(session.find_session_info(json.dumps(events), build_id="other"))

    def test_session_id_parsing(self):
        self.assertEqual(session.parse_session_id(" 123 "), 123)
        self.assertEqual(session.parse_session_id("https://k.com/code/a/b/edit/run/99/"), 99)
        with self.assertRaises(ValueError):
            session.parse_session_id("abc")

    @unittest.skipIf(os.name == "nt", "needs POSIX signals and process groups")
    def test_sigterm_reaches_the_job_so_it_can_save(self):
        (self.proj / "train.py").write_text(
            "import signal, sys, time, pathlib\n"
            "out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
            "def bye(*a):\n"
            "    (out / 'r_saved_on_term.txt').write_text('saved')\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, bye)\n"
            "print('ready', flush=True)\n"
            "time.sleep(60)\n")
        cfg = cli.load_config(self.cfg_path)
        inputs = self.tmp / "input" / "demo-code"
        inputs.mkdir(parents=True)
        cli.build_bundle(cfg, self.proj, inputs / cli.BUNDLE_TAR, "b1", "r")
        kernel = self.tmp / "kernel.py"
        kernel.write_text(cli.render_kernel(cfg, "r", "b1", "", False))
        env = {**os.environ, "KGR_INPUT_ROOT": str(self.tmp / "input"),
               "KGR_WORKING_ROOT": str(self.tmp / "working"),
               "KGR_BUNDLE_DIR": str(self.tmp / "bundle")}
        p = subprocess.Popen([sys.executable, str(kernel)], env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            if line.startswith("ready"):
                break
        p.send_signal(signal.SIGTERM)
        p.communicate(timeout=30)
        self.assertTrue((self.tmp / "working" / "outputs" / "r_saved_on_term.txt").is_file())
        summary = json.loads((self.tmp / "working" / "kgr_run_summary.json").read_text())
        self.assertTrue(summary["terminated"])

    # ------------------------------------------------------------------
    # convenience commands

    @needs_symlink
    def test_runs_logs_and_status(self):
        self.kgr("run", "run_1", "--interval", "0.05")
        self.kgr("info", "run_1")          # remembers the session id
        rc, out = self.kgr("runs")
        self.assertEqual(rc, 0)
        self.assertIn("run_1", out)
        self.assertIn("COMPLETE", out)
        self.assertIn("424242", out)
        rc, out = self.kgr("logs", "run_1", "--tail", "3")
        self.assertEqual(rc, 0)
        self.assertEqual(len(out.splitlines()), 3)
        target = self.tmp / "saved.log"
        rc, out = self.kgr("logs", "run_1", "--save", str(target))
        self.assertIn("run_1", target.read_text(encoding="utf-8"))
        rc, out = self.kgr("status", "run_1")   # the CLI's own line goes to the real stdout
        self.assertEqual(rc, 0)
        self.assertIn("session 424242", out)

    def test_push_resets_the_remembered_session(self):
        self.kgr("run", "run_1", "--interval", "0.05")
        self.kgr("info", "run_1")
        self.assertTrue(session.load_state(self.proj, "run_1").get("session_id"))
        self.kgr("push", "run_1")
        self.assertFalse(session.load_state(self.proj, "run_1").get("session_id"))

    def test_gpu_aliases_and_extra_sources_reach_the_metadata(self):
        self.write_config(extra='competitions = ["titanic"]\nkernels = ["someone/notebook"]\n'
                                'models = ["g/m/pyTorch/2b/1"]')
        rc, out = self.kgr("push", "run_1", "--gpu", "t4", "--resume", "--dry-run")
        self.assertEqual(rc, 0, out)
        stage = Path(out.rsplit("staged in ", 1)[1].split(",")[0].strip())
        meta = json.loads((stage / "kernel" / "kernel-metadata.json").read_text())
        self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(meta["competition_sources"], ["titanic"])
        self.assertEqual(meta["kernel_sources"], ["tester/demo-run-1", "someone/notebook"])
        self.assertEqual(meta["model_sources"], ["g/m/pyTorch/2b/1"])
        shutil.rmtree(stage)

    def test_user_config_supplies_the_username(self):
        uc = self.tmp / "user.toml"
        uc.write_text('[kaggle]\nuser = "globaluser"\n')
        os.environ["KGR_USER_CONFIG"] = str(uc)
        self.cfg_path.write_text('[kaggle]\nproject = "p"\n[run]\ncommand = "true"\n')
        cfg = cli.load_config(self.cfg_path)
        self.assertEqual(cli.kernel_ref(cfg, "r1"), "globaluser/p-r1")

    def test_init_global_and_project(self):
        uc = self.tmp / "user.toml"
        os.environ["KGR_USER_CONFIG"] = str(uc)
        rc, out = self.kgr("init", "--global", "--user", "me", config=False)
        self.assertEqual(rc, 0, out)
        os.chdir(self.tmp)
        (self.tmp / "My Project").mkdir()
        os.chdir(self.tmp / "My Project")
        rc, out = self.kgr("init", config=False)
        self.assertEqual(rc, 0, out)
        text = (self.tmp / "My Project" / "kgr.toml").read_text()
        self.assertIn('user = "me"', text)
        self.assertIn('project = "my-project"', text)
        cli.read_toml(self.tmp / "My Project" / "kgr.toml")   # template must parse

    @needs_symlink
    def test_exec_runs_a_bare_script_without_kgr_toml(self):
        bare = self.tmp / "bare"
        bare.mkdir()
        (bare / "hello.py").write_text(
            "import os, sys, pathlib\n"
            "out = pathlib.Path(os.environ['KGR_OUTPUT_DIR'])\n"
            "(out / 'result.txt').write_text(' '.join(sys.argv[1:]))\n")
        os.environ["KGR_KAGGLE"] = fake_bin()
        os.chdir(bare)
        rc, out = self.kgr("exec", "--user", "me", "--gpu", "none", "--competition", "titanic",
                           "--interval", "0.05", "hello.py", "--", "--lr", "3e-4",
                           config=False)
        self.assertEqual(rc, 0, out)
        self.assertEqual((bare / "outputs" / "hello" / "outputs" / "result.txt").read_text(),
                         "--lr 3e-4")
        meta = json.loads((self.kernel_dir("bare-hello") / "meta.json").read_text())
        self.assertEqual(meta["id"], "me/bare-hello")
        self.assertEqual(meta["competition_sources"], ["titanic"])
        self.assertTrue((bare / ".kgr" / "runs" / "hello.json").is_file())
        # no kgr.toml here, yet the run can be inspected from this directory
        os.environ["KGR_USER"] = "me"
        rc, out = self.kgr("info", "hello", config=False)
        self.assertEqual(rc, 0, out)
        self.assertIn("me/bare-hello", out)
        self.assertIn("424242", out)

    def test_exec_needs_a_username(self):
        os.environ["KGR_KAGGLE"] = fake_bin()
        (self.tmp / "s.py").write_text("print(1)\n")
        os.chdir(self.tmp)
        rc, out = self.kgr("exec", "s.py", config=False)
        self.assertEqual(rc, 1)
        self.assertIn("[kaggle] user", out)

    def test_competition_data_and_submit(self):
        os.environ["KGR_KAGGLE"] = fake_bin()
        os.chdir(self.tmp)
        rc, out = self.kgr("data", "titanic", config=False)
        self.assertEqual(rc, 0, out)
        self.assertTrue((self.tmp / "data" / "titanic" / "train.csv").is_file())
        self.assertFalse(list((self.tmp / "data").glob("*.zip")))
        (self.tmp / "sub.csv").write_text("id,y\n1,0\n")
        rc, out = self.kgr("submit", "titanic", "sub.csv", "-m", "first", config=False)
        self.assertEqual(rc, 0, out)
        sent = json.loads((self.home / "submit.json").read_text())
        self.assertEqual(sent, ["competitions", "submit", "titanic", "-m", "first", "-f", "sub.csv"])

    def test_quota_needs_no_project(self):
        os.environ["KGR_KAGGLE"] = fake_bin()
        os.chdir(self.tmp)
        rc, out = self.kgr("quota", config=False)
        self.assertEqual(rc, 0)

    def test_dataset_state_ignores_a_warning_before_the_json(self):
        cfg = cli.load_config(self.cfg_path)
        noisy = ("Warning: Looks like you're using an outdated `kaggle` version\n"
                 '{"status": "ready", "current_version_number": 3}\n')
        done = subprocess.CompletedProcess([], 0, noisy, "")
        orig, cli.kaggle = cli.kaggle, lambda *a, **k: done
        try:
            self.assertEqual(cli.dataset_state(cfg, "tester/demo-code"), ("ready", 3))
        finally:
            cli.kaggle = orig

    def test_kaggle_bin_with_quoted_path_is_parsed(self):
        cfg = {"kaggle": {"bin": f'"{sys.executable}" "{FAKE}"'}}
        self.assertEqual(cli.kaggle_argv(cfg), [sys.executable, str(FAKE)])


if __name__ == "__main__":
    unittest.main()
