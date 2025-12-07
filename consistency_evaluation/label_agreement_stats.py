"""
Reads a .txt file containing a JSON array of objects with the following shape:

[
  {
    "index": 0,
    "labels": ["SRP", "SRP", "SRP", "SRP"],
    "label_agreement": 1.0
    },...
]
"""

import json
import sys
from typing import List, Dict, Any


def load_instances(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Top-level JSON structure must be a list of instances.")

    return data


def compute_mean_label_agreement(instances: List[Dict[str, Any]]) -> float:
    """
    Compute the mean of the 'label_agreement' field across all instances.
    """
    if not instances:
        raise ValueError("No instances found in the data.")

    total = 0.0
    count = 0

    for inst in instances:
        if "label_agreement" not in inst:
            # You can decide to skip, raise, or treat missing as 0.0.
            raise KeyError("Instance missing 'label_agreement' field: {}".format(inst))
        total += float(inst["label_agreement"])
        count += 1

    return total / count


def main():
    if len(sys.argv) != 2:
        print("Usage: python label_agreement_stats.py <path_to_txt_file>")
        sys.exit(1)

    path = sys.argv[1]

    try:
        instances = load_instances(path)
        mean_agreement = compute_mean_label_agreement(instances)
        print(f"Mean label_agreement over {len(instances)} instances: {mean_agreement:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()