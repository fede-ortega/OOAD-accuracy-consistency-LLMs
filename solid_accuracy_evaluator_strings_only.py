
"""
Purpose:
    Run an LLM (Llama) "self-judge" on SOLID adherence for a given code string,
    focused on the "Accuracy" of adherence to SOLID principles.
    The model must return a brief rationale and Likert Scale for our annotation:
    adherence_score and violation_severity (both 1–5 integers).

Environment:
    API credentials/endpoint go in a .env file if the client requires it.
    This script loads .env automatically.

Usage:
    # Pass the code as a single CLI argument (quote your string)
    python solid_accuracy_evaluator_strings_only.py --text "<your code here>" --pretty

    # Or read from a file to avoid shell-escaping issues
    python solid_accuracy_evaluator_strings_only.py --from-file path/to/code.txt --pretty
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from llama_api_client import LlamaAPIClient


@dataclass
class AccuracyPromptTemplate:
    """Evaluation prompt for SOLID Accuracy"""

    max_words: int = 80

    def build_system_prompt(self) -> str:
        return (
            "You are a strict software design reviewer familiar with OOAD principles. "
            "Judge the provided code's adherence to SOLID principles with focus on ACCURACY: "
            "how well responsibilities, abstractions, and coupling in the code align with SOLID. "
            "Respond ONLY with valid JSON using the keys: "
            "explanation, adherence_score, violation_severity. "
            f"Limit 'explanation' to <= {self.max_words} words. "
            "Use integers for scores (1–5). "
            "No extra prose or code fences."
        )

    def build_user_prompt(self, code: str) -> str:
        scale = (
            "Scales:\n"
            "- adherence_score (1–5): 1=largely non-adherent; 2=poor; 3=mixed; 4=good; 5=excellent.\n"
            "- violation_severity (1–5): 1=negligible; 2=low; 3=moderate; 4=high; 5=critical.\n\n"
            "Evaluate the following code for SOLID ACCURACY. "
            "Cite concrete issues (e.g., SRP violations, tight coupling, hard-coded deps, magic numbers, leaky abstractions). "
            "If mixed, balance strengths and weaknesses. Return ONLY JSON."
        )
        schema_hint = (
            "Return JSON like:\n"
            "{"
            "\"explanation\": \"<=80 words rationale\", "
            "\"adherence_score\": 1, "
            "\"violation_severity\": 3"
            "}"
        )
        return f"{scale}\n\n{schema_hint}\n\nCODE TO REVIEW:\n\n{code}"


class LlamaJudgeClient:
    """wrapper around LlamaAPIClient."""

    def __init__(self, model: str = "Llama-4-Maverick-17B-128E-Instruct-FP8", temperature: float = 0.0, max_completion_tokens: int = 6000):
        load_dotenv()
        self.client = LlamaAPIClient()
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
        )
        return completion.completion_message.content.text


# --------------------------------
# Evaluator / Pipeline (OOP core)
# --------------------------------

class JsonExtractor:
    """JSON extractor to handle occasional model formatting issues."""

    @staticmethod
    def first_json_object(text: str) -> str:
        # This handles cases where the model wraps JSON in prose or code fences.
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found in model response.")
        return match.group(0)

    @staticmethod
    def parse(text: str) -> Dict[str, Any]:
        raw = JsonExtractor.first_json_object(text)
        return json.loads(raw)


@dataclass
class SolidAccuracyEvaluator:
    """Coordinates prompt building, model call, and JSON parsing"""

    client: LlamaJudgeClient
    template: AccuracyPromptTemplate

    def evaluate(self, code: str) -> Dict[str, Any]:
        system_prompt = self.template.build_system_prompt()
        user_prompt = self.template.build_user_prompt(code)
        raw_text = self.client.complete(system_prompt, user_prompt)
        try:
            parsed = JsonExtractor.parse(raw_text)
        except Exception as e:
            # If parsing fails, surface the raw text to aid debugging.
            parsed = {
                "explanation": "Model did not return valid JSON; see raw field.",
                "adherence_score": None,
                "violation_severity": None,
                "raw": raw_text,
                "error": str(e),
            }
        parsed = SolidAccuracyEvaluator._coerce_and_clip(parsed)
        return parsed

    @staticmethod
    def _coerce_and_clip(data: Dict[str, Any]) -> Dict[str, Any]:
        def to_int_1_5(v: Any) -> Optional[int]:
            try:
                n = int(v)
            except Exception:
                return None
            return max(1, min(5, n))

        if "adherence_score" in data:
            data["adherence_score"] = to_int_1_5(data.get("adherence_score"))
        if "violation_severity" in data:
            data["violation_severity"] = to_int_1_5(data.get("violation_severity"))
        if "explanation" in data and isinstance(data["explanation"], str):
            words = data["explanation"].split()
            if len(words) > 80:
                data["explanation"] = " ".join(words[:80])
        return data


# CLI

def _read_input(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.from_file is not None:
        with open(args.from_file, "r", encoding="utf-8") as f:
            return f.read()
    raise SystemExit("Provide --text or --from-file.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SOLID Accuracy adherence of a code snippet via Llama.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Code snippet as a string (quote it).")
    src.add_argument("--from-file", type=str, help="Path to a file containing the code snippet.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON result.")
    parser.add_argument("--model", type=str, default="Llama-4-Maverick-17B-128E-Instruct-FP8", help="Override model name if needed.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=6000, help="Max completion tokens.")
    args = parser.parse_args()

    code_input = _read_input(args)

    client = LlamaJudgeClient(model=args.model, temperature=args.temperature, max_completion_tokens=args.max_tokens)
    template = AccuracyPromptTemplate(max_words=80)
    evaluator = SolidAccuracyEvaluator(client=client, template=template)

    result = evaluator.evaluate(code_input)
    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
