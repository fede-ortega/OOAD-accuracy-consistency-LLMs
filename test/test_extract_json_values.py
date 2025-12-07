import json

import pytest

from adherence_violation_evaluation import extract_json_values as ejv


def test_norm_key_normalizes_variants():
    assert ejv._norm_key("Adherence_Score") == "adherencescore"
    assert ejv._norm_key(" violation severity ") == "violationseverity"
    assert ejv._norm_key("EXPLANATION") == "explanation"


def test_canonicalize_keys_handles_typos_and_aliases():
    raw = {
        "Explanation": "foo",
        "adheranceScore": "3",
        "violation severity": "4",
    }
    c = ejv._canonicalize_keys(raw)

    assert c["explanation"] == "foo"
    assert "adherence_score" in c
    assert "violation_severity" in c


def test_to_int_1_5_coercion_and_clipping():
    assert ejv._to_int_1_5("3") == 3
    assert ejv._to_int_1_5("4.7") == 5            # rounded then clipped
    assert ejv._to_int_1_5("0") == 1            # lower bound
    assert ejv._to_int_1_5("6.2") == 5          # upper bound
    assert ejv._to_int_1_5("not-a-number") is None


def test_extract_item_basic_pass_through():
    raw = {
        "explanation": "x",
        "adherence_score": 2,
        "violation_severity": 4,
    }
    rec = ejv._extract_item(raw)
    assert rec == {
        "explanation": "x",
        "adherence_score": 2,
        "violation_severity": 4,
    }


def test_extract_item_with_weird_keys_and_strings():
    raw = {
        "Explanation": "x",
        "Adherance score": "2.8",
        "violationSeverity": "10",
    }
    rec = ejv._extract_item(raw)

    assert rec["explanation"] == "x"
    # 2.8 -> round to 3, 10 -> clipped to 5
    assert rec["adherence_score"] == 3
    assert rec["violation_severity"] == 5


def test_read_json_valid(tmp_path):
    data = {"code_examples": []}
    path = tmp_path / "in.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = ejv._read_json(str(path))
    assert loaded == data


def test_read_json_invalid_exits(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json}", encoding="utf-8")

    with pytest.raises(SystemExit):
        ejv._read_json(str(path))
