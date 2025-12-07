"""
Batch wrapper that reuses the strings-only evaluator from the first file.

- Supports:
  --text "<one string>"                      → evaluates one snippet
  --text '["code A","code B"]'               → evaluates many (JSON array BUT BETWEEN QUOTES!)
  --from-file snippets.txt                   → extracts multiple "input": "..." blocks from a loose TXT

- Retunrs a JSON array, one object per snippet, each with:
  explanation, adherence_score, violation_severity
"""

from __future__ import annotations

import argparse
import json
import re
from typing import List, Union

from adherence_violation_evaluation.solid_accuracy_evaluator_strings_only import (
    AccuracyPromptTemplate,
    LlamaJudgeClient,
    SolidAccuracyEvaluator,
)
# Accepts loose TXT containing repeated: "input": "...."
_INPUT_PAIR_RE = re.compile(
    r'(?P<key>["\']?input["\']?)\s*:\s*"(?P<val>(?:[^"\\]|\\.)*)"',
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

def parse_inputs_from_txt(content: str) -> List[str]:
    """
    Extract all code snippets written as  "input": "..."  from a TXT file.
    - If no pattern is found, the entire file is treated as ONE snippet.
    """
    matches = list(_INPUT_PAIR_RE.finditer(content))
    if matches:
        snippets: List[str] = []
        for m in matches:
            raw = m.group("val")
            try:
                decoded = json.loads(f'"{raw}"')  # unescape sequences
            except Exception:
                decoded = (
                    raw.replace('\\n', '\n')
                       .replace('\\t', '\t')
                       .replace('\\r', '\r')
                       .replace('\\"', '"')
                )
            snippets.append(decoded)
        return snippets
    return [content]  

def _maybe_json_array(text: str) -> Union[str, List[str]]:
    """If `text` looks like a JSON array of strings, parse and return it; else return the original string."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except Exception:
        pass
    return text

def _read_inputs(args: argparse.Namespace) -> List[str]:
    """Return a list of code snippets from CLI inputs."""
    if args.text is not None:
        maybe = _maybe_json_array(args.text)
        return maybe if isinstance(maybe, list) else [maybe]
    if args.from_file is not None:
        with open(args.from_file, "r", encoding="utf-8") as f:
            content = f.read()
        return parse_inputs_from_txt(content)
    raise SystemExit("Provide --text or --from-file. For multiple snippets in TXT, use \"input\": \"...\" blocks.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch SOLID Accuracy evaluation (reuses single-snippet evaluator).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Single code string or a JSON array of strings.")
    src.add_argument("--from-file", type=str, help="Path to a TXT file with repeated entries like: \"input\": \"...\"")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output (array).")
    parser.add_argument("--model", type=str, default="Llama-4-Maverick-17B-128E-Instruct-FP8", help="Model override.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=6000, help="Max completion tokens.")
    args = parser.parse_args()

    client = LlamaJudgeClient(model=args.model, temperature=args.temperature, max_completion_tokens=args.max_tokens)
    template = AccuracyPromptTemplate(max_words=80)
    evaluator = SolidAccuracyEvaluator(client=client, template=template)

    snippets = _read_inputs(args)
    results = [evaluator.evaluate(code) for code in snippets]

    if args.pretty:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(results, separators=(",", ":"), ensure_ascii=False))

if __name__ == "__main__":
    main()