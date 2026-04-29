"""Walk reproducibility/data/runs/, validate each JSON, emit results.csv + manifest.json.

Deterministic by design: sorted rows, fixed column order, LF line endings, sorted JSON.
The committed CSV must equal the output of this script for any given runs/ tree, which
the CI workflow enforces via --check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# Make `reproducibility.lib` importable when invoked as a script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reproducibility.lib import SCHEMA_VERSION, validate, ValidationError  # noqa: E402

DATA_DIR = _REPO_ROOT / "reproducibility" / "data"
RUNS_DIR = DATA_DIR / "runs"
RESULTS_CSV = DATA_DIR / "results.csv"
MANIFEST_JSON = DATA_DIR / "manifest.json"

CSV_COLUMNS = [
    "schema_version",
    "run_id",
    "dataset_id",
    "method_id",
    "model",
    "params_hash",
    "method_params_json",
    "llm_temperature",
    "llm_max_tokens",
    "metric",
    "value",
    "num_queries",
    "total_time_seconds",
    "querygym_version",
    "run_file_path",
]


def _iter_run_files(runs_dir: Path) -> Iterator[Path]:
    yield from sorted(runs_dir.rglob("*.json"))


def _load_and_validate(path: Path, dataset_registry, method_registry) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    try:
        validate(
            payload,
            dataset_registry=dataset_registry,
            method_registry=method_registry,
        )
    except ValidationError as e:
        raise SystemExit(f"validation failed for {path}: {e}") from e
    return payload


def _payload_to_rows(payload: dict, run_path: Path) -> list[list]:
    """One row per metric. Returns rows in CSV_COLUMNS order."""
    pipeline = payload["pipeline"]
    config = payload["config"]
    rel_path = run_path.relative_to(_REPO_ROOT).as_posix()

    base = [
        payload["schema_version"],
        payload["run_id"],
        pipeline["dataset_id"],
        pipeline["method_id"],
        pipeline["model"],
        payload["params_hash"],
        json.dumps(config["method_params"], sort_keys=True, separators=(",", ":")),
        config["llm_config"]["temperature"],
        config["llm_config"]["max_tokens"],
        # metric / value filled per row below
        None,
        None,
        config["dataset_config"]["num_queries"],
        pipeline["total_time_seconds"],
        payload["querygym_version"],
        rel_path,
    ]

    rows = []
    for metric in sorted(payload["metrics"].keys()):
        row = list(base)
        row[9] = metric
        row[10] = payload["metrics"][metric]
        rows.append(row)
    return rows


def _write_csv(rows: list[list]) -> str:
    """Render CSV to a string with deterministic settings."""
    buf = io.StringIO(newline="")
    # csv.writer with QUOTE_MINIMAL + LF is deterministic across platforms.
    writer = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(CSV_COLUMNS)
    writer.writerows(rows)
    return buf.getvalue()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _querygym_version() -> str:
    try:
        import querygym  # type: ignore
        return getattr(querygym, "__version__", "unknown")
    except ImportError:
        return "unknown"


def aggregate(runs_dir: Path) -> tuple[str, dict]:
    """Build the canonical CSV text and a manifest dict from runs_dir.

    Returns (csv_text, manifest) — both deterministic for a given runs_dir.
    """
    # Lazy load registries once for the whole walk (saves file IO per run).
    from reproducibility.lib.validate import (
        _load_dataset_registry,
        _load_method_registry,
    )

    dataset_registry = _load_dataset_registry(None)
    method_registry = list(_load_method_registry())

    all_rows: list[list] = []
    run_count = 0
    for run_path in _iter_run_files(runs_dir):
        payload = _load_and_validate(run_path, dataset_registry, method_registry)
        all_rows.extend(_payload_to_rows(payload, run_path))
        run_count += 1

    # Sort by (dataset_id, method_id, model, params_hash, metric) for stable diffs.
    sort_idx = (
        CSV_COLUMNS.index("dataset_id"),
        CSV_COLUMNS.index("method_id"),
        CSV_COLUMNS.index("model"),
        CSV_COLUMNS.index("params_hash"),
        CSV_COLUMNS.index("metric"),
    )
    all_rows.sort(key=lambda r: tuple(r[i] for i in sort_idx))

    csv_text = _write_csv(all_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "querygym_version": _querygym_version(),
        "run_count": run_count,
        "row_count": len(all_rows),
        "content_hash": _content_hash(csv_text),
    }
    return csv_text, manifest


def _read_committed_files() -> tuple[str | None, dict | None]:
    csv_text = RESULTS_CSV.read_text(encoding="utf-8") if RESULTS_CSV.exists() else None
    manifest = (
        json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        if MANIFEST_JSON.exists()
        else None
    )
    return csv_text, manifest


def cmd_write(runs_dir: Path) -> int:
    csv_text, manifest = aggregate(runs_dir)
    manifest["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV.write_text(csv_text, encoding="utf-8")
    MANIFEST_JSON.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {RESULTS_CSV.relative_to(_REPO_ROOT)} "
        f"({manifest['run_count']} runs, {manifest['row_count']} rows, "
        f"content_hash={manifest['content_hash'][:12]}...)"
    )
    return 0


def cmd_check(runs_dir: Path) -> int:
    csv_text, manifest = aggregate(runs_dir)
    committed_csv, committed_manifest = _read_committed_files()

    failures = []

    if committed_csv is None:
        failures.append(f"{RESULTS_CSV.relative_to(_REPO_ROOT)} is missing")
    elif committed_csv != csv_text:
        failures.append(
            f"{RESULTS_CSV.relative_to(_REPO_ROOT)} is out of date "
            f"(committed != regenerated)"
        )

    if committed_manifest is None:
        failures.append(f"{MANIFEST_JSON.relative_to(_REPO_ROOT)} is missing")
    else:
        # Compare data-correctness fields only. querygym_version and generated_at
        # are informational provenance — they reflect *where* and *when* the
        # manifest was produced and are expected to differ between contributor
        # machines and CI. content_hash already pins the actual aggregate data.
        for key in ("schema_version", "run_count", "row_count", "content_hash"):
            committed_val = committed_manifest.get(key)
            fresh_val = manifest.get(key)
            if committed_val != fresh_val:
                failures.append(
                    f"{MANIFEST_JSON.relative_to(_REPO_ROOT)}: {key} mismatch "
                    f"(committed={committed_val!r}, regenerated={fresh_val!r})"
                )

    if failures:
        print("Aggregator --check failed:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            "\nFix by running:\n"
            "  python -m reproducibility.scripts.aggregate_runs\n"
            "and committing the diff.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: {manifest['run_count']} runs, {manifest['row_count']} rows, "
        f"content_hash={manifest['content_hash'][:12]}..."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate run JSONs into results.csv + manifest.json."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify committed files match what the aggregator would produce. "
        "Exits non-zero on mismatch. Used by CI.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help=f"Directory to walk for run JSONs (default: {RUNS_DIR.relative_to(_REPO_ROOT)}).",
    )
    args = parser.parse_args()

    if not args.runs_dir.exists():
        args.runs_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        return cmd_check(args.runs_dir)
    return cmd_write(args.runs_dir)


if __name__ == "__main__":
    raise SystemExit(main())
