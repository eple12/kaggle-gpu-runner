"""End-to-end tests against a fake Kaggle that runs kernels locally.

    python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kgr import cli  # noqa: E402

FAKE = REPO / "tests" / "fake_kaggle.py"
TRAIN = REPO / "examples" / "minimal" / "train.py"


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
        os.environ["FAKE_KAGGLE_HOME"] = str(self.home)
        self.cfg_path = self.proj / "kgr.toml"
        self.write_config()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_config(self, accelerator="none", hooks=()):
        kaggle_bin = f"{shlex.quote(sys.executable)} {shlex.quote(str(FAKE))}"
        self.cfg_path.write_text(f'''
[kaggle]
user = "tester"
project = "demo"
bin = {json.dumps(kaggle_bin)}
accelerator = "{accelerator}"
dataset_wait = 30

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
''')

    def kgr(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                rc = cli.main(["-c", str(self.cfg_path), *argv])
            except SystemExit as e:
                rc = e.code
        return rc, out.getvalue()

    def fetched_state(self, run):
        return json.loads((self.proj / "fetched" / f"{run}_state.json").read_text())

    # ------------------------------------------------------------------

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

    def test_run_then_resume_across_kernels(self):
        rc, out = self.kgr("run", "run_1", "--interval", "0.05")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.fetched_state("run_1")["step"], 300)
        self.assertTrue((self.proj / "fetched" / "run_1.kaggle.log").is_file())
        push_args = json.loads((self.home / "kernels" / "demo-run-1" / "push_args.json").read_text())
        self.assertNotIn("--accelerator", push_args, "CPU kernel must not request a GPU")

        # Second leg: attach the first kernel's output and train further.
        rc, out = self.kgr("run", "run_1", "--resume-from", "tester/demo-run-1",
                           "--extra", "--steps 600", "--interval", "0.05")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.fetched_state("run_1")["step"], 600)
        log = (self.proj / "fetched" / "run_1.kaggle.log").read_text()
        self.assertIn("starting at step 300", log)
        self.assertIn("resumed files: ['run_1_state.json']", log)

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
        self.write_config(hooks=[f"{shlex.quote(sys.executable)} -c \"raise SystemExit(1)\""])
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
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("demo-code/kgr_bundle", r.stdout)
        summary = json.loads((self.tmp / "working" / "kgr_run_summary.json").read_text())
        self.assertEqual(summary["outputs"], ["outputs/r_state.json"])

    def run_kernel(self, kernel, inputs):
        env = {**os.environ, "KGR_INPUT_ROOT": str(inputs),
               "KGR_WORKING_ROOT": str(self.tmp / "working"),
               "KGR_BUNDLE_DIR": str(self.tmp / "bundle")}
        return subprocess.run([sys.executable, str(kernel)], env=env,
                              capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
