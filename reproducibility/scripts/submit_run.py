"""Validate a fresh run JSON and copy it (plus sibling artifacts) into the canonical layout.

Used by:
- Internal trusted contributors after running the example pipeline.
- External fork contributors before opening a PR.
- The one-time SIGIR backfill (re-emitting legacy JSONs under v1).

The example pipeline writes pipeline_summary.json (v1 schema) plus run.txt and
reformulated_queries.tsv to its --output-dir. This script picks those up and lays them
into reproducibility/data/runs/{dataset_id}/{method_id}/{model}/{params_hash}.{ext}.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reproducibility.lib import validate, ValidationError  # noqa: E402

DEFAULT_RUNS_DIR = _REPO_ROOT / "reproducibility" / "data" / "runs"

# Common filenames the example pipeline produces in --output-dir.
SUMMARY_CANDIDATES = ("pipeline_summary.json",)
RUN_FILE_CANDIDATES = ("runs/run.txt", "run.txt")
QUERIES_CANDIDATES = (
    "queries/reformulated_queries.tsv",
    "reformulated_queries.tsv",
)


def _find(from_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for c in candidates:
        p = from_dir / c
        if p.exists():
            return p
    return None


def _resolve_inputs(from_dir: Path) -> tuple[Path, Path, Path]:
    summary = _find(from_dir, SUMMARY_CANDIDATES)
    if summary is None:
        raise SystemExit(
            f"could not find pipeline_summary.json under {from_dir}. "
            f"Did the pipeline complete?"
        )
    run_file = _find(from_dir, RUN_FILE_CANDIDATES)
    if run_file is None:
        raise SystemExit(
            f"could not find run.txt under {from_dir} (looked in: {RUN_FILE_CANDIDATES})"
        )
    queries = _find(from_dir, QUERIES_CANDIDATES)
    if queries is None:
        raise SystemExit(
            f"could not find reformulated_queries.tsv under {from_dir} "
            f"(looked in: {QUERIES_CANDIDATES})"
        )
    return summary, run_file, queries


def _canonical_dir(runs_dir: Path, payload: dict) -> Path:
    p = payload["pipeline"]
    return runs_dir / p["dataset_id"] / p["method_id"] / p["model"]


def _copy(src: Path, dst: Path, *, force: bool) -> None:
    if dst.exists() and not force:
        raise SystemExit(
            f"refusing to overwrite {dst.relative_to(_REPO_ROOT)} (use --force)."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a run output directory and copy its files into the canonical "
            "reproducibility/data/runs/ layout."
        )
    )
    parser.add_argument(
        "--from-dir",
        type=Path,
        required=True,
        help="Directory produced by examples/querygym_pyserini/pipeline.py.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Target runs directory (default: {DEFAULT_RUNS_DIR.relative_to(_REPO_ROOT)}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing run with the same params_hash.",
    )
    parser.add_argument(
        "--skip-registry-checks",
        action="store_true",
        help="Skip dataset/method registry validation (use only for synthetic test runs).",
    )
    args = parser.parse_args()

    if not args.from_dir.is_dir():
        raise SystemExit(f"--from-dir does not exist: {args.from_dir}")

    summary, run_file, queries = _resolve_inputs(args.from_dir)

    with summary.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    try:
        validate(payload, skip_registry_checks=args.skip_registry_checks)
    except ValidationError as e:
        raise SystemExit(f"validation failed for {summary}: {e}") from e

    target_dir = _canonical_dir(args.runs_dir, payload)
    h = payload["params_hash"]

    json_dst = target_dir / f"{h}.json"
    run_dst = target_dir / f"{h}.run.txt"
    queries_dst = target_dir / f"{h}.queries.tsv"

    target_dir.mkdir(parents=True, exist_ok=True)

    # Write the validated payload (not a verbatim copy of summary) — guarantees
    # the on-disk JSON is byte-identical to what the validator just OK'd.
    if json_dst.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite {json_dst.relative_to(_REPO_ROOT)} (use --force)."
        )
    with json_dst.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")

    _copy(run_file, run_dst, force=args.force)
    _copy(queries, queries_dst, force=args.force)

    rel = json_dst.relative_to(_REPO_ROOT).as_posix()
    print(f"wrote {rel}")
    print("Now run:\n  make repro-aggregate\nand commit the diff.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
