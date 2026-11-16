"""
Checks that Lift behaves exactly the same with the current working tree as with a reference commit (default HEAD):
every observation, reward, done flag, success flag and sim state over seeded rollouts must be bit-identical.

It builds a copy of the robosuite package at the reference in a temp dir (symlinks to the working tree, with every
file under robosuite/ that differs from the reference replaced by its reference version), runs lift_record.py
against each version, and compares the results. The working tree is never modified.

Usage (from the repo root, in the env robosuite is installed in):
    MUJOCO_GL=egl python tests/ycb/check_lift_unchanged.py [--ref HEAD] [--dataset path/to/lift/image_abs.hdf5]

Exits 0 if identical, 1 otherwise.
"""
import argparse
import collections
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECORD = os.path.join(REPO, "tests", "ycb", "lift_record.py")


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], check=True, capture_output=True, text=True).stdout


def build_ref_tree(ref, dest):
    """Symlink copy of robosuite/ with the files that differ from @ref swapped for their @ref versions."""
    pkg = os.path.join(dest, "robosuite")
    shutil.copytree(os.path.join(REPO, "robosuite"), pkg, symlinks=False,
                    copy_function=os.symlink, ignore=shutil.ignore_patterns("__pycache__"))

    changed = []
    for line in git("diff", "--name-status", "--no-renames", ref, "--", "robosuite").splitlines():
        status, path = line.split("\t", 1)
        target = os.path.join(dest, path)
        if os.path.lexists(target):
            os.remove(target)
        if status != "A":  # added since @ref -> just leave it out
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                f.write(subprocess.run(["git", "-C", REPO, "show", f"{ref}:{path}"],
                                       check=True, capture_output=True).stdout)
        changed.append(f"{status} {path}")
    # untracked files don't exist at @ref either
    for path in git("ls-files", "--others", "--exclude-standard", "robosuite").splitlines():
        target = os.path.join(dest, path)
        if os.path.lexists(target):
            os.remove(target)
            changed.append(f"? {path}")
    return changed


def record(pythonpath, cwd, out, dataset):
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"))
    if pythonpath:
        env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    # Print which robosuite gets imported, so a wrong import path can't silently compare a version against itself
    which = subprocess.run([sys.executable, "-c", "import robosuite; print(robosuite.__file__)"],
                           cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]
    print(f"  robosuite: {which}")
    cmd = [sys.executable, RECORD, out] + (["--dataset", dataset] if dataset else [])
    subprocess.run(cmd, cwd=cwd, env=env, check=True)
    return which


def compare(path_a, path_b):
    a, b = np.load(path_a), np.load(path_b)
    if set(a.files) != set(b.files):
        print("FAIL: different recorded keys:", sorted(set(a.files) ^ set(b.files))[:10])
        return False
    diff = [k for k in a.files if a[k].shape != b[k].shape or not np.array_equal(a[k], b[k])]
    print(f"compared {len(a.files)} arrays; {len(diff)} differ")
    if diff:
        print("differing, by config:", dict(collections.Counter(k.split("/")[0] for k in diff)))
        for k in diff[:10]:
            print("  DIFF", k)
    return not diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="HEAD", help="git ref to compare the working tree against")
    parser.add_argument("--dataset", default=None, help="Lift robomimic hdf5, to also check the eval env path")
    args = parser.parse_args()
    if args.dataset is not None:
        args.dataset = os.path.abspath(args.dataset)  # recordings run from a temp dir

    with tempfile.TemporaryDirectory(prefix="lift_check_") as tmp:
        ref_dir = os.path.join(tmp, "ref")
        changed = build_ref_tree(args.ref, ref_dir)
        print(f"files under robosuite/ that differ from {args.ref}:", changed or "none")

        # Run both from the temp dir: running from the repo root would put the working tree first on sys.path
        print(f"recording {args.ref} ...")
        ref_pkg = record(ref_dir, tmp, os.path.join(tmp, "ref.npz"), args.dataset)
        print("recording working tree ...")
        cur_pkg = record(None, tmp, os.path.join(tmp, "cur.npz"), args.dataset)
        assert ref_pkg.startswith(ref_dir) and not cur_pkg.startswith(ref_dir), \
            "import paths are wrong; the two runs did not use different code"

        ok = compare(os.path.join(tmp, "ref.npz"), os.path.join(tmp, "cur.npz"))
    print("PASS: Lift is identical" if ok else "FAIL: Lift differs")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
