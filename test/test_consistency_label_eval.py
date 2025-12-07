import json

import pytest

from consistency_evaluation import consistency_label_eval as cle


def test_label_only_prompt_template_ab_ba_order():
    template = cle.LabelOnlyPromptTemplate()
    code = "public class Foo {}"

    msgs_ab = template.messages_for_order(code, "AB")
    msgs_ba = template.messages_for_order(code, "BA")

    # First message must be system
    assert msgs_ab[0]["role"] == "system"
    assert "OOAD reviewer" in msgs_ab[0]["content"]

    # AB vs BA should differ in ordering of the few-shot examples
    assert msgs_ab[1]["content"] != msgs_ba[1]["content"]

    # Both should end with an eval_user message containing the code
    assert msgs_ab[-1]["content"].endswith("CODE TO REVIEW:\n" + code)
    assert msgs_ba[-1]["content"].endswith("CODE TO REVIEW:\n" + code)


def test_label_agreement_compute_simple():
    # 3 SRP, 1 OCP -> 3/4
    labels = ["SRP", "srp", " OCP ", "SRP"]
    la = cle.LabelAgreement.compute(labels)
    assert la == pytest.approx(0.75)

    # Empty list -> 0.0
    assert cle.LabelAgreement.compute([]) == 0.0

    # All labels unknown -> 0.0
    assert cle.LabelAgreement.compute(["foo", "bar"]) == 0.0


def test_flatten_messages_for_single_user():
    messages = [
        {"role": "system", "content": "You are a test system."},
        {"role": "user", "content": "First example"},
        {"role": "assistant", "content": '{"violation":"SRP"}'},
        {"role": "user", "content": "Second example"},
    ]

    system_prompt, user_prompt = cle._flatten_messages_for_single_user(messages)

    assert "You are a test system." in system_prompt
    # Roles should be tagged
    assert "### User:\nFirst example" in user_prompt
    assert "### Assistant:\n" in user_prompt
    # Final instruction should be present
    assert '{"violation":"SRP|OCP|LSP|ISP|DIP"}' in user_prompt


class FakeJsonExtractor:
    @staticmethod
    def parse(raw: str):
        # Parse the JSON string returned by the fake client
        return json.loads(raw)


class FakeClientMessages:
    """Fake client that implements complete_messages; always returns SRP."""

    def complete_messages(self, msgs, temperature, max_completion_tokens, seed):
        return '{"violation": "SRP"}'


def test_abba_evaluator_evaluate_item_all_same_label(monkeypatch):
    # Patch JsonExtractor used inside consistency_label_eval
    monkeypatch.setattr(cle, "JsonExtractor", FakeJsonExtractor)

    client = FakeClientMessages()
    template = cle.LabelOnlyPromptTemplate()
    evaluator = cle.ABBAEvaluator(
        client=client,
        template=template,
        temperature=0.0,
        max_completion_tokens=16,
        seed=123,
    )

    res = evaluator.evaluate_item("public class Foo {}", k=2)
    # k=2 -> 2 permutations * 2 repeats = 4 labels
    assert res["labels"] == ["SRP", "SRP", "SRP", "SRP"]
    assert res["label_agreement"] == pytest.approx(1.0)


def test_load_json_array_txt(tmp_path):
    data = [
        {"input": "code 1"},
        {"input": "code 2"},
    ]
    path = tmp_path / "dataset.txt"
    path.write_text(json.dumps(data), encoding="utf-8")

    codes = cle._load_json_array_txt(str(path))
    assert codes == ["code 1", "code 2"]


def test_load_json_array_txt_invalid_json(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("[{bad json]", encoding="utf-8")

    with pytest.raises(SystemExit):
        cle._load_json_array_txt(str(path))