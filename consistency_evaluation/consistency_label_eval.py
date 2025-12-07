"""
AB/BA consistency using a *label-only* JSON schema:
  {"violation":"SRP|OCP|LSP|ISP|DIP"}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from adherence_violation_evaluation.solid_accuracy_evaluator_strings_only import LlamaJudgeClient, JsonExtractor 

# ---------------- Prompt Template (label-only, no NONE) ----------------

@dataclass
class LabelOnlyPromptTemplate:
    def system(self) -> str:
        return (
            "You are an OOAD reviewer. Evaluate SOLID (SRP, OCP, LSP, ISP, DIP). "
            "Return ONLY this minified JSON: "
            "{\"violation\":\"SRP|OCP|LSP|ISP|DIP\"}. "
            "Choose the single most severe violated principle; you MUST pick exactly one of these five. "
            "No extra text, no code fences."
        )

    @staticmethod
    def shot_a_pair() -> List[Dict[str, str]]:
        code = (
            "Language: Java\n"
            "public abstract class Document {\n"
            "  protected String content;\n"
            "  public Document(String content){ this.content = content; }\n"
            "  public abstract void save();\n"
            "  public abstract void print();\n"
            "}\n"
            "public class PDFDocument extends Document {\n"
            "  public PDFDocument(String c){ super(c); }\n"
            "  public void save(){ System.out.println(\"Saving PDF\"); }\n"
            "  public void print(){ System.out.println(\"Printing PDF\"); }\n"
            "}\n"
            "public class ReadOnlyDocument extends Document {\n"
            "  public ReadOnlyDocument(String c){ super(c); }\n"
            "  public void save(){ throw new UnsupportedOperationException(\"Cannot save\"); }\n"
            "  public void print(){ System.out.println(\"Printing\"); }\n"
            "}\n"
        )
        return [
            {"role": "user", "content": code},
            {"role": "assistant", "content": '{"violation":"LSP"}'},
        ]

    @staticmethod
    def shot_b_pair() -> List[Dict[str, str]]:
        code = (
            "Language: Python\n"
            "from enum import Enum\n"
            "class DeviceType(Enum): LIGHT=1; THERMOSTAT=2; DOOR_LOCK=3\n"
            "class SmartDevice:\n"
            "  def __init__(self, id, t): self._id=id; self._t=t\n"
            "  def get_id(self): return self._id\n"
            "  def get_type(self): return self._t\n"
            "class SmartHomeController:\n"
            "  def turn_on(self, d):\n"
            "    if d.get_type()==DeviceType.LIGHT: print(f\"Light {d.get_id()} ON\")\n"
            "    elif d.get_type()==DeviceType.THERMOSTAT: print(f\"Thermostat {d.get_id()} heat 22C\")\n"
            "    elif d.get_type()==DeviceType.DOOR_LOCK: print(f\"Door lock {d.get_id()} cannot be ON\")\n"
        )
        return [
            {"role": "user", "content": code},
            {"role": "assistant", "content": '{"violation":"OCP"}'},
        ]

    def eval_user(self, code: str) -> Dict[str, str]:
        return {
            "role": "user",
            "content": (
                "Evaluate the following code and return ONLY: "
                '{"violation":"SRP|OCP|LSP|ISP|DIP"}\n\n'
                "CODE TO REVIEW:\n" + code
            ),
        }

    def messages_for_order(self, code: str, order: str) -> List[Dict[str, str]]:
        assert order in ("AB", "BA")
        msgs: List[Dict[str, str]] = [{"role": "system", "content": self.system()}]
        a = self.shot_a_pair()
        b = self.shot_b_pair()
        msgs += (a + b) if order == "AB" else (b + a)
        msgs.append(self.eval_user(code))
        return msgs


# ---------------- Label Agreement (5 labels only) ----------------

class LabelAgreement:
    LABELS = ("SRP", "OCP", "LSP", "ISP", "DIP")

    @staticmethod
    def normalize(label: str) -> str:
        return label.strip().upper()

    @staticmethod
    def compute(labels: List[str]) -> float:
        if not labels:
            return 0.0
        counts = {l: 0 for l in LabelAgreement.LABELS}
        for l in labels:
            n = LabelAgreement.normalize(l)
            if n in counts:
                counts[n] += 1
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return max(counts.values()) / total


# Fallback: flatten multi-message to single user turn 

def _flatten_messages_for_single_user(messages: List[Dict[str, str]]) -> tuple[str, str]:
    if not messages or messages[0]["role"] != "system":
        raise ValueError("First message must be a system prompt.")
    system_prompt = messages[0]["content"]

    parts: List[str] = []
    for m in messages[1:]:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            parts.append("### User:\n" + content)
        elif role == "assistant":
            parts.append("### Assistant:\n" + content)
        else:
            parts.append(content)
    user_prompt = "\n\n".join(parts) + "\n\n### Assistant:\n" \
                  'Return ONLY: {"violation":"SRP|OCP|LSP|ISP|DIP"}'
    return system_prompt, user_prompt


@dataclass
class ABBAEvaluator:
    client: LlamaJudgeClient
    template: LabelOnlyPromptTemplate
    temperature: float = 0.0
    max_completion_tokens: int = 512
    seed: Optional[int] = 42

    def _run_once(self, code: str, order: str) -> str:
        msgs = self.template.messages_for_order(code, order)

        if hasattr(self.client, "complete_messages"):
            raw = self.client.complete_messages(
                msgs,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                seed=self.seed,
            )
        else:
            sys_prompt, usr_prompt = _flatten_messages_for_single_user(msgs)
            raw = self.client.complete(sys_prompt, usr_prompt)

        data = JsonExtractor.parse(raw)
        return str(data.get("violation", "")).strip()

    def evaluate_item(self, code: str, k: int) -> Dict[str, Any]:
        labels: List[str] = []
        for order in ("AB", "BA"):
            for _ in range(k):
                labels.append(self._run_once(code, order))
        la = LabelAgreement.compute(labels)
        return {"labels": labels, "label_agreement": la}


# Dataset helpers (.txt with JSON array)

def _load_json_array_txt(path: str) -> List[str]:
    """
    Expects a .txt file that contains a JSON array:
      [{"input":"<code1>"}, {"input":"<code2>"}]
    Returns the list of code strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        arr = json.loads(content)
        if not isinstance(arr, list):
            raise ValueError("Top-level JSON must be an array.")
        items: List[str] = []
        for i, obj in enumerate(arr):
            if not isinstance(obj, dict) or "input" not in obj:
                raise ValueError(f"Element {i} must be an object with key 'input'.")
            items.append(str(obj["input"]))
        return items
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in {path}: {e}")

# CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Label-only AB/BA consistency (no NONE).")
    ap.add_argument("--dataset", type=str, required=True,
                    help="Path to .txt file containing a JSON array: [{\"input\":\"<code>\"}, ...]")
    ap.add_argument("--k", type=int, default=3, help="Repeats per permutation.")
    ap.add_argument("--model", type=str, default="Llama-4-Maverick-17B-128E-Instruct-FP8")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    codes = _load_json_array_txt(args.dataset)

    client = LlamaJudgeClient(model=args.model,
                              temperature=args.temperature,
                              max_completion_tokens=args.max_tokens)
    template = LabelOnlyPromptTemplate()
    runner = ABBAEvaluator(client=client,
                           template=template,
                           temperature=args.temperature,
                           max_completion_tokens=args.max_tokens,
                           seed=args.seed)

    out: List[Dict[str, Any]] = []
    for idx, code in enumerate(codes):
        res = runner.evaluate_item(code, k=args.k)
        out.append({"index": idx, **res})

    print(json.dumps(out, indent=2 if args.pretty else None, ensure_ascii=False))


if __name__ == "__main__":
    main()