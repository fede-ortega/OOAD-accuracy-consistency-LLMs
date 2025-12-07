import json
from types import SimpleNamespace

import pytest

from adherence_violation_evaluation import solid_accuracy_evaluator_lists_strings as batch


def test_parse_inputs_from_txt_with_input_blocks():
    content = '''
    { "input": "print(\\"hello\\")" }
    some other line
    { "input": "int x = 1;" }
    '''
    snippets = batch.parse_inputs_from_txt(content)

    assert len(snippets) == 2
    # First snippet should have quotes unescaped
    assert 'print("hello")' in snippets[0]
    assert "int x = 1;" in snippets[1]


def test_parse_inputs_from_txt_falls_back_to_whole_file():
    content = "some random file with no input key"
    snippets = batch.parse_inputs_from_txt(content)
    assert snippets == [content]


def test_maybe_json_array_single_string():
    text = "print('hi')"
    assert batch._maybe_json_array(text) == text


def test_maybe_json_array_parses_array_of_strings():
    text = '["a", "b"]'
    res = batch._maybe_json_array(text)
    assert isinstance(res, list)
    assert res == ["a", "b"]


def test_read_inputs_from_text_json_array():
    args = SimpleNamespace(text='["foo","bar"]', from_file=None)
    snippets = batch._read_inputs(args)
    assert snippets == ["foo", "bar"]


def test_read_inputs_from_file_uses_parse_inputs(tmp_path):
    file = tmp_path / "inputs.txt"
    file.write_text('{ "input": "code 1" }\n{ "input": "code 2" }', encoding="utf-8")
    args = SimpleNamespace(text=None, from_file=str(file))

    snippets = batch._read_inputs(args)
    assert snippets == ["code 1", "code 2"]


def test_read_inputs_requires_source():
    args = SimpleNamespace(text=None, from_file=None)
    with pytest.raises(SystemExit):
        batch._read_inputs(args)