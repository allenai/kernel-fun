"""Copy a winning kernel chain out of the research ladder into this package.

The package is a RELEASE ARTIFACT of a research result, not a live mirror of the ladder.
That distinction is the whole design:

  - The ladder must stay free to fork, falsify and abandon. Idea 004 alone falsified three
    rebuilds in one phase; under a live mirror each of those is an edit to a shipping
    package.
  - `loop.ideas` computes an idea's `code_sha` from files in ITS OWN directory. If the
    ladder imported kernels from here, editing one would not change any idea's code_sha and
    `code_unchanged_since` — the field that separates a real improvement from node variance
    — would quietly start lying.
  - The package must diverge on purpose: dead branches deleted, 18 env knobs frozen, four
    copies of the call cache unified. A byte-for-byte drift check would be red on day one,
    and a check that is always red gets deleted.

So this tool does the mechanical 90% of a vendor — copy at a named commit, rewrite the
cross-idea imports — and then prints the checklist of edits it cannot do. It is run by a
human, once per win, and its output is reviewed like any other commit. `drift.py` reports
what has moved upstream since; it is advisory and never part of a test run.

The ladder is a SEPARATE repo (github.com/allenai/kernel-fun-dev) as of 2026-09-09 — `--from-commit`
names a commit in IT, not in this one, and `find_ladder` below says how the checkout is located.

Usage:
    python tools/vendor.py --family kda --from-commit <sha>      # writes and reports
    python tools/vendor.py --family kda --from-commit <sha> -n   # dry run
    python tools/vendor.py --family kda --from-commit <sha> --only bwd_wy_t.py
        # copy ONE new module; the other vendored files keep their hand edits. The
        # provenance is still regenerated for the whole family at that commit, so use it
        # only when the files you skip are unchanged there (drift.py says).
"""

from __future__ import annotations

import argparse
import hashlib
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

# family -> {destination module under kernel_fun/<family>/_kernels/: source path in the repo}
# Only modules that are LIVE on the default path at production shapes. What is deliberately
# left behind, and why, is in the package README.
FILE_MAP: dict[str, dict[str, str]] = {
    "kda": {
        "fwd_state.py": "kernels/kda/ideas/002-cute-bwd/kernel_fwd.py",
        "bwd_scan.py": "kernels/kda/ideas/002-cute-bwd/kernel_scan.py",
        "bwd_dhu.py": "kernels/kda/ideas/002-cute-bwd/kernel_dhu.py",
        "bwd_wy.py": "kernels/kda/ideas/002-cute-bwd/kernel_wy2.py",
        "bwd_intra_triton.py": "kernels/kda/ideas/002-cute-bwd/kernel_intra.py",
        "bwd_intra.py": "kernels/kda/ideas/003-intra-mma/kernel_intra_cute.py",
        "fwd_intra_triton.py": "kernels/kda/ideas/004-fwd-block/kernel_fwd_intra_triton.py",
        "bwd_wy_t.py": "kernels/kda/ideas/005-wy-transposed/kernel_wy_t.py",
    },
    "cconv": {
        "strip.py": "kernels/cconv/ideas/001-strip-onepass/kernel.py",
    },
}

# The ladder reaches across idea folders by importlib string, because folder names start
# with digits and are not importable identifiers. In the package they are siblings.
IMPORT_REWRITES: tuple[tuple[str, str], ...] = (
    (
        'import_module("kernels.kda.ideas.002-cute-bwd.kernel_intra")',
        'import_module("kernel_fun.kda._kernels.bwd_intra_triton")',
    ),
    (
        'import_module("kernels.kda.ideas.002-cute-bwd.kernel_fwd")',
        'import_module("kernel_fun.kda._kernels.fwd_state")',
    ),
    (
        'importlib.import_module("kernels.kda.ideas.002-cute-bwd.kernel_wy2")',
        'importlib.import_module("kernel_fun.kda._kernels.bwd_wy")',
    ),
    ("from .kernel_intra import", "from .bwd_intra_triton import"),
    ("from .kernel_wy_cute import", "from .bwd_wy_cute import"),
    ("from .kernel_fwd_intra_triton import", "from .fwd_intra_triton import"),
)

# What the tool cannot do, restated at the end of every run so it is never assumed done.
CHECKLIST = """
Hand edits this tool does NOT do — work through them and diff before committing:

  1. FREEZE THE KNOBS. Every os.environ.get("KDA00*" / "CCONV_*") read becomes a constant at its
     default. The six read at MODULE scope are the urgent ones: in a library an env var
     read at import time makes numerics depend on what was exported before the first
     import. Grep: os.environ.get in the vendored files.
       _SKIP / SKIP_DIAG / SKIP_OFFDIAG / SKIP_SOLVE  -> delete (attribution only, WRONG
                                                         results by design)
       *_MAXREG -> _MAXREG = 128 ;  KDA002_INTRA_BK -> _BK = 64
       KDA003_DGFOLD, KDA003_BF16 -> True (part of the measured win)
       KDA005_WY (forcing knob) -> delete; KDA005_MAIN_STAGES/SIDE_* -> the pinned
                                   constants; PROBE_SKIP -> delete (attribution only)
       CCONV_TARGET_PROGRAMS -> TARGET_PROGRAMS = 2048
     Preserve `emit_bf16 &= (g.shape[2] == q.shape[2])` verbatim — dropping that guard
     silently degrades GVA gradients through the fp32 group reduction.

  2. DELETE THE DEAD BRANCHES the knobs selected, and the modules they pointed at.

  3. UNIFY THE CALL CACHE. Each vendored kernel carries its own _cute_view/_retarget/
     _release_keepalives/_call_key. Replace with kernel_fun._common.cache and hoist
     MIN_CTAS (three copies) to _common.support.

  4. CHECK THE fla IMPORTS against _common/compat.py's FLA_SYMBOLS table; add anything new.

  5. RUN THE TESTS ON A B300. Nothing in this tool verifies anything.
"""


def git_show(ladder: Path, commit: str, path: str) -> str:
    """`path` is relative to the LADDER repo; `commit` is a commit in it."""
    out = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ladder, capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        sys.exit(f"git show {commit}:{path} failed:\n{out.stderr}")
    return out.stdout


def rewrite(text: str) -> tuple[str, list[str]]:
    applied = []
    for old, new in IMPORT_REWRITES:
        if old in text:
            text = text.replace(old, new)
            applied.append(f"{old} -> {new}")
    return text, applied


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=sorted(FILE_MAP))
    ap.add_argument("--from-commit", required=True,
                    help="the LADDER commit whose recorded bench row this package ships")
    ap.add_argument("--ladder", default=None, metavar="PATH",
                    help=f"the research-ladder checkout (default: ${LADDER_ENV}, else "
                         f"the sibling {LADDER_DEFAULT})")
    ap.add_argument("-n", "--dry-run", action="store_true")
    ap.add_argument("--only", action="append", default=None, metavar="DEST",
                    help="write only these destination modules (repeatable); the rest "
                         "are hashed for provenance but left as they are on disk")
    args = ap.parse_args()
    ladder = find_ladder(args.ladder)
    only = set(args.only or ())
    unknown = only - set(FILE_MAP[args.family])
    if unknown:
        sys.exit(f"--only names not in FILE_MAP[{args.family!r}]: {sorted(unknown)}")

    dest_dir = PKG / args.family / "_kernels"
    hashes: dict[str, str] = {}
    print(f"vendoring {args.family} from {ladder}@{args.from_commit} -> {dest_dir}")
    for dest, src in FILE_MAP[args.family].items():
        text = git_show(ladder, args.from_commit, src)
        hashes[src] = hashlib.sha256(text.encode()).hexdigest()
        text, applied = rewrite(text)
        note = f"  ({len(applied)} import rewrite(s))" if applied else ""
        if only and dest not in only:
            print(f"  {src}  ->  {dest}  (hashed only; --only)")
            continue
        print(f"  {src}  ->  {dest}{note}")
        for line in applied:
            print(f"      {line}")
        if not args.dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / dest).write_text(text)

    prov = PKG / args.family / "_provenance.py"
    body = [
        '"""Generated by tools/vendor.py — what this family was cut from.',
        "",
        "Hand-edited after vendoring (knobs frozen, dead branches removed, call cache",
        "unified), so these hashes identify the SOURCE, not the files here. The paths and",
        "the commit are in the research ladder (github.com/allenai/kernel-fun-dev), NOT this",
        "repo; drift.py compares them against that checkout's current HEAD.",
        '"""',
        "",
        f'SOURCE_COMMIT = "{args.from_commit}"',
        "SOURCE_FILES = {",
        *[f'    "{src}": "{h}",' for src, h in sorted(hashes.items())],
        "}",
    ]
    if not args.dry_run:
        prov.write_text("\n".join(body) + "\n")
    print(f"  provenance -> {prov}")
    print(CHECKLIST)


if __name__ == "__main__":
    main()
