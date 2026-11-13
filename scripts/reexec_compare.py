"""Compare one re-execution with the reported run. Runs inside the image (needs pandas); no network, no agent code.

    python scripts/reexec_compare.py --reported-dir <run out/> --new-dir <re-executed out/> --out compare.json [--tolerances file]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import reexec  # noqa: E402


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reported-dir", required=True, type=Path)
    parser.add_argument("--new-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tolerances", default=str(Path(__file__).with_name("reexec_tolerances.json")))
    args = parser.parse_args()
    tolerances = reexec.load_tolerances(args.tolerances)
    reported = read_json(args.reported_dir / "results.json")
    old = reexec.read_predictions(args.reported_dir / "predictions.csv")
    new = reexec.read_predictions(args.new_dir / "predictions.csv")
    new_results = read_json(args.new_dir / "results.json")
    methods = reexec.compare(reported, old, new, tolerances)
    for entry in methods:  # what the re-executed code itself says about each method
        status = (((new_results or {}).get("methods") or {}).get(entry["name_reported"]) or {}).get("status")
        entry["status_in_reexecuted_results"] = status
    out = {"tolerances": tolerances, "reported_results_found": reported is not None, "reexecuted_results_found": new_results is not None,
           "reexecuted_predictions_found": new is not None, "methods": methods}
    args.out.write_text(json.dumps(out, indent=1))
    judged = [m for m in methods if m["within_tolerance"] is not None]
    print(f"{sum(1 for m in judged if m['within_tolerance'])} of {len(judged)} methods within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
