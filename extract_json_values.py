
"""
extract_scores_to_txt.py

Converts a JSON file with structure:
{
  "code_examples": [ { "input": "...", "output": "...", "explanation": "...",
                       "adherence_score": 2, "violation_severity": 4, ... }, ... ]
}

into a TXT file containing ONLY a JSON array of objects with:
  - explanation
  - adherence_score
  - violation_severity

Robustness:
- Accepts key variants: "adherence score", "adherance_score", "violation severity"
  (case-insensitive; underscores/spaces ignored)
- Coerces scores to integers in [1..5] when possible; otherwise null
- Leaves "explanation" as-is

Usage:
  python extract_scores_to_txt.py --in data.json --out results.txt --pretty
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

REQUIRED_KEYS = ["explanation", "adherence_score", "violation_severity"]

CANONICAL = {
    "explanation": "explanation",
    "adherencescore": "adherence_score",
    "adherancescore": "adherence_score",
    "violationseverity": "violation_severity",
}

ALIASES = {
    "adherence score": "adherence_score",
    "adherance score": "adherence_score",
    "violation severity": "violation_severity",
}

def _norm_key(k: str) -> str:
    return k.lower().replace("_", "").replace(" ", "")

def _canonicalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        nk = _norm_key(k)
        if nk in CANONICAL:
            out[CANONICAL[nk]] = v
    for alias, canon in ALIASES.items():
        for k, v in d.items():
            if k.strip().lower() == alias:
                out[canon] = v
    if "explanation" in d and "explanation" not in out:
        out["explanation"] = d["explanation"]
    return out

def _to_int_1_5(v: Any) -> Optional[int]:
    try:
        n = int(v)
    except Exception:
        try:
            f = float(v)
            n = int(round(f))
        except Exception:
            return None
    return max(1, min(5, n))

def _extract_item(raw: Dict[str, Any]) -> Dict[str, Any]:
    c = _canonicalize_keys(raw)
    rec: Dict[str, Any] = {
        "explanation": c.get("explanation", None),
        "adherence_score": _to_int_1_5(c.get("adherence_score")) if "adherence_score" in c else None,
        "violation_severity": _to_int_1_5(c.get("violation_severity")) if "violation_severity" in c else None,
    }
    return rec

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            sys.exit(f"ERROR: Failed to parse JSON file '{path}': {e}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract explanation/adherence_score/violation_severity to TXT (JSON array).")
    parser.add_argument("--in", dest="in_path", required=True, help="Path to the source JSON file.")
    parser.add_argument("--out", dest="out_path", required=True, help="Path to write the TXT file.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the output JSON array.")
    args = parser.parse_args()

    data = _read_json(args.in_path)
    examples = data.get("code_examples", None)
    if not isinstance(examples, list):
        sys.exit("ERROR: The JSON must have a 'code_examples' array.")

    results: List[Dict[str, Any]] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        results.append(_extract_item(ex))

    with open(args.out_path, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            json.dump(results, f, separators=(",", ":"), ensure_ascii=False)

    print(f"Wrote {len(results)} records to {args.out_path}")

if __name__ == "__main__":
    main()
