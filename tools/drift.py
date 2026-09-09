"""What has moved in the research ladder since this package was vendored?

Advisory by design. The package diverges from the ladder ON PURPOSE — knobs frozen, dead
branches deleted, call caches unified — so a hash mismatch is the normal state, not a
failure. This tool answers "is there a newer version of the kernels I'm shipping", which is
a question you ask before a release, not on every test run.

The check that the package still computes the right thing is tests/test_parity.py.

    python tools/drift.py                 # every family
    python tools/drift.py --family kda
    python tools/drift.py --strict        # exit 1 if anything moved (for a release gate)
    python tools/drift.py --ladder ~/Code/kernel-fun-dev

The ladder is a SEPARATE repo (github.com/allenai/kernel-fun-dev) as of 2026-09-09; see
find_ladder below for how it is located.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "src" / "kernel_fun"

# --- Locating the research ladder --------------------------------------------------------
# Until 2026-09-09 this package lived INSIDE the ladder repo, and the ladder was simply
# HERE.parents[2]. Now it is a separate checkout, so it has to be found. In order:
#   --ladder PATH  >  $KERNEL_FUN_LADDER  >  a sibling directory named kernel-fun
# The sibling default is what a `git clone` of both repos into the same parent produces, so
# the common case still needs no argument.
LADDER_ENV = "KERNEL_FUN_LADDER"
LADDER_DEFAULT = HERE.parents[1] / "kernel-fun-dev"   # ../kernel-fun-dev, beside this repo


def find_ladder(explicit: str | None) -> Path:
    """The research-ladder checkout these kernels are cut from
    (github.com/allenai/kernel-fun-dev)."""
    for cand, whence in (
        (explicit, "--ladder"),
        (os.environ.get(LADDER_ENV), f"${LADDER_ENV}"),
        (LADDER_DEFAULT, "the sibling default"),
    ):
        if not cand:
            continue
        path = Path(cand).expanduser().resolve()
        if not path.exists():
            sys.exit(
                f"{whence} points at {path}, which does not exist.\n"
                f"  git clone git@github.com:allenai/kernel-fun-dev.git {path}\n"
                f"or pass --ladder / set ${LADDER_ENV}."
            )
        if not (path / ".git").exists():
            sys.exit(
                f"{whence} points at {path}, which is not a git checkout.\n"
                f"Clone the ladder next to this repo, or pass --ladder / set ${LADDER_ENV}."
            )
        if not (path / "kernels").is_dir():
            sys.exit(
                f"{whence} points at {path}, which has no kernels/ — that is not the "
                f"research ladder (github.com/allenai/kernel-fun-dev)."
            )
        return path
    raise AssertionError("unreachable: LADDER_DEFAULT is always truthy")


def load_provenance(family: str):
    path = PKG / family / "_provenance.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"_prov_{family}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def head_hash(ladder: Path, path: str) -> str | None:
    """`path` is relative to the LADDER repo — that is what _provenance.py records."""
    out = subprocess.run(
        ["git", "show", f"HEAD:{path}"], cwd=ladder, capture_output=True, check=False
    )
    if out.returncode != 0:
        return None
    return hashlib.sha256(out.stdout).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default=None)
    ap.add_argument("--strict", action="store_true", help="exit 1 if anything moved")
    ap.add_argument("--ladder", default=None, metavar="PATH",
                    help=f"the research-ladder checkout (default: ${LADDER_ENV}, else "
                         f"the sibling {LADDER_DEFAULT})")
    args = ap.parse_args()
    ladder = find_ladder(args.ladder)
    print(f"ladder: {ladder} @ HEAD")

    families = [args.family] if args.family else sorted(
        p.name for p in PKG.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "_provenance.py").exists()
    )
    moved = 0
    for family in families:
        prov = load_provenance(family)
        if prov is None:
            print(f"{family}: no provenance (never vendored?)")
            continue
        print(f"{family}: vendored from {prov.SOURCE_COMMIT}")
        for src, want in sorted(prov.SOURCE_FILES.items()):
            got = head_hash(ladder, src)
            if got is None:
                print(f"  GONE     {src}")
                moved += 1
            elif got != want:
                print(f"  CHANGED  {src}")
                moved += 1
    if moved:
        print(
            f"\n{moved} source file(s) moved since vendoring. That is expected while the "
            f"ladder is being worked on; re-vendor when an idea WINS a recorded row, not "
            f"every time one changes."
        )
    else:
        print("\nup to date with HEAD")
    return 1 if (moved and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
