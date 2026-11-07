"""A stand-in for the `kaggle` CLI that runs kernels on this machine.

Implements just the calls kgr makes. `kernels push` really executes the
generated kernel, with the attached datasets and previous kernel outputs laid
out as an input tree, so the tests exercise the same path Kaggle would.
State lives under $FAKE_KAGGLE_HOME.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path(os.environ["FAKE_KAGGLE_HOME"])


def slug(ref):
    return ref.split("/", 1)[1]


def arg_after(args, flag):
    return args[args.index(flag) + 1]


def main(a):
    if a[:2] == ["datasets", "status"]:
        d = HOME / "datasets" / slug(a[2])
        if not d.exists():
            print("403 Client Error: Forbidden", file=sys.stderr)
            return 1
        v = int((d / "version").read_text())
        print(json.dumps({"status": "ready", "current_version_number": v}) if "--format" in a else "ready")
        return 0

    if a[:2] in (["datasets", "create"], ["datasets", "version"]):
        src = Path(arg_after(a, "-p"))
        meta = json.loads((src / "dataset-metadata.json").read_text())
        d = HOME / "datasets" / slug(meta["id"])
        if a[1] == "create" and d.exists():
            print("dataset already exists", file=sys.stderr)
            return 1
        if a[1] == "version" and not d.exists():
            print("no such dataset", file=sys.stderr)
            return 1
        version = int((d / "version").read_text()) + 1 if d.exists() else 1
        shutil.rmtree(d / "current", ignore_errors=True)
        shutil.copytree(src, d / "current")
        (d / "version").write_text(str(version))
        return 0

    if a[:2] == ["kernels", "push"]:
        src = Path(arg_after(a, "-p"))
        meta = json.loads((src / "kernel-metadata.json").read_text())
        ks = slug(meta["id"])
        k = HOME / "kernels" / ks
        inputs = HOME / "inputs" / ks
        shutil.rmtree(inputs, ignore_errors=True)
        inputs.mkdir(parents=True)
        for ref in meta["dataset_sources"]:
            shutil.copytree(HOME / "datasets" / slug(ref) / "current", inputs / slug(ref))
        for ref in meta["kernel_sources"]:
            shutil.copytree(HOME / "kernels" / slug(ref) / "output", inputs / slug(ref))
        out = k / "output"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        env = {**os.environ, "KGR_INPUT_ROOT": str(inputs), "KGR_WORKING_ROOT": str(out),
               "KGR_BUNDLE_DIR": str(HOME / "bundles" / ks)}
        # What a real Kaggle container exposes (names are the fake's own choice).
        env.update(KAGGLE_KERNEL_RUN_TYPE="Batch", KAGGLE_SESSION_ID="424242",
                   KAGGLE_DATA_PROXY_TOKEN="very-secret", KAGGLE_URL_BASE="https://www.kaggle.com",
                   PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        r = subprocess.run([sys.executable, str(src / meta["code_file"])], env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        (k / "log.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
        (out / f"{ks}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        (k / "status").write_text("COMPLETE" if r.returncode == 0 else "ERROR")
        (k / "meta.json").write_text(json.dumps(meta))
        (k / "push_args.json").write_text(json.dumps(a))
        print("Kernel version 1 successfully pushed.")
        return 0

    if a[:2] == ["kernels", "status"]:
        st = (HOME / "kernels" / slug(a[2]) / "status").read_text()
        print(f'{a[2]} has status "KernelWorkerStatus.{st}"')
        return 0

    if a[:2] == ["kernels", "output"]:
        shutil.copytree(HOME / "kernels" / slug(a[2]) / "output", arg_after(a, "-p"),
                        dirs_exist_ok=True)
        return 0

    if a[:2] == ["kernels", "logs"]:
        print((HOME / "kernels" / slug(a[2]) / "log.txt").read_text(encoding="utf-8"))
        return 0

    if a[:1] == ["quota"]:
        print("GPU: 3.00h used of 30.00h")
        return 0

    if a[:2] == ["competitions", "download"]:
        import zipfile
        dest = Path(arg_after(a, "-p"))
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest / f"{a[2]}.zip", "w") as zf:
            zf.writestr("train.csv", "a,b\n1,2\n")
        return 0

    if a[:2] == ["competitions", "submit"]:
        (HOME / "submit.json").write_text(json.dumps(a))
        return 0

    print(f"fake kaggle: unsupported {a}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
