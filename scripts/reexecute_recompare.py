"""Redo the numbers comparison for finished re-executions with the current comparison rules (draft).

    python3 scripts/reexecute_recompare.py --tool-image ssa-study:<tag> [--result-root ~/ssa-reexec]

The re-executions themselves are not repeated: each result folder keeps the re-executed `results.json` and `predictions.csv`
(`reexecuted_*`), and the comparison against the reported run is computed again. Used when the comparison rule changes (for example how
a method that was reported as failed is judged). Updates `result.json` (methods, all_within_tolerance, taxonomy_run_level) and keeps the
old comparison as `compare_before_recompare.json`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import reexecute_run, ssa_run  # noqa: E402


def recompute_fields(result: dict, methods: list[dict]) -> dict:
    judged = [m for m in methods if m.get("within_tolerance") is not None]
    ran = result.get("first_attempt") == "ran"
    updated = dict(result, methods=methods)
    updated["all_within_tolerance"] = bool(judged) and all(m["within_tolerance"] for m in judged) and len(judged) == len(methods)
    updated["taxonomy_run_level"] = "a" if not ran else ("none" if updated["all_within_tolerance"] else "b")
    return updated


def main(argv: list[str] | None = None, compare=reexecute_run.compare_in_image) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-root", type=Path, default=reexecute_run.RESULT_ROOT)
    parser.add_argument("--runs-dir", type=Path, default=ssa_run.RUNS_DIR)
    parser.add_argument("--tool-image", required=True)
    parser.add_argument("--tolerances", type=Path, default=reexecute_run.TOLERANCES)
    args = parser.parse_args(argv)
    changed = checked = 0
    for result_path in sorted(args.result_root.glob("ssa-*/exec*/result.json")):
        folder = result_path.parent
        if "_repair" in folder.name or not (folder / "reexecuted_results.json").exists():
            continue
        result = reexecute_run.read_json(result_path) or {}
        run_dir = args.runs_dir / folder.parent.name
        with tempfile.TemporaryDirectory() as tmp:
            new_dir, out_dir = Path(tmp) / "new", Path(tmp) / "out"
            new_dir.mkdir(), out_dir.mkdir()
            shutil.copy2(folder / "reexecuted_results.json", new_dir / "results.json")
            if (folder / "reexecuted_predictions.csv").exists():
                shutil.copy2(folder / "reexecuted_predictions.csv", new_dir / "predictions.csv")
            for path in (new_dir, out_dir):
                path.chmod(0o777)
            outcome = compare(args.tool_image, run_dir / "out", new_dir, out_dir, args.tolerances)
        if not outcome:
            print(f"{folder}: comparison gave no result; left as it was", file=sys.stderr)
            continue
        checked += 1
        updated = recompute_fields(result, outcome["methods"])
        if (updated["all_within_tolerance"], updated["taxonomy_run_level"]) != (result.get("all_within_tolerance"), result.get("taxonomy_run_level")) \
                or updated["methods"] != result.get("methods"):
            changed += 1
            if (folder / "compare.json").exists():
                shutil.copy2(folder / "compare.json", folder / "compare_before_recompare.json")
            updated["recompared_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            result_path.write_text(json.dumps(updated, indent=1))
    print(f"recompared {checked} results, {changed} changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
