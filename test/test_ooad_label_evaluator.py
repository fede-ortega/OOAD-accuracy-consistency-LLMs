import json
import os

import pytest

from consistency_evaluation import ooad_label_evaluator as ole


def test_majority_label_selection_basic():
    strat = ole.MajorityLabelSelection()
    # Clear majority
    assert strat.select_label(["SRP", "SRP", "OCP"]) == "SRP"
    assert strat.select_label(["SRP", "OCP", "OCP", "SRP"]) == "SRP"

    with pytest.raises(ValueError):
        strat.select_label([])


def test_gold_loader(tmp_path):
    gold_data = [
        {"labels": "DIP"},
        {"labels": "ISP"},
    ]
    path = tmp_path / "gold.txt"
    path.write_text(json.dumps(gold_data), encoding="utf-8")

    loader = ole.GoldLoader()
    labels = loader.load(str(path))
    assert labels == ["DIP", "ISP"]


def test_gold_loader_invalid_structure(tmp_path):
    path = tmp_path / "gold.txt"
    # Not a list
    path.write_text(json.dumps({"labels": "DIP"}), encoding="utf-8")
    loader = ole.GoldLoader()

    with pytest.raises(ValueError):
        loader.load(str(path))


def test_prediction_loader_with_majority_selection(tmp_path):
    pred_data = [
        {"index": 0, "labels": ["SRP", "SRP", "OCP"], "label_agreement": 0.66},
        {"index": 1, "labels": ["DIP", "OCP", "DIP"], "label_agreement": 0.66},
    ]
    path = tmp_path / "pred.txt"
    path.write_text(json.dumps(pred_data), encoding="utf-8")

    strat = ole.MajorityLabelSelection()
    loader = ole.PredictionLoader(selection_strategy=strat)

    preds = loader.load(str(path))
    assert preds == ["SRP", "DIP"]


def test_build_examples_mismatch_lengths():
    gold = ["SRP", "OCP"]
    preds = ["SRP"]

    with pytest.raises(ValueError):
        ole.build_examples(gold, preds)


def test_classification_evaluator_simple_case():
    # Two labels: SRP and OCP; 4 examples
    examples = [
        ole.LabeledExample(index=0, gold_label="SRP", predicted_label="SRP"),
        ole.LabeledExample(index=1, gold_label="SRP", predicted_label="OCP"),
        ole.LabeledExample(index=2, gold_label="OCP", predicted_label="OCP"),
        ole.LabeledExample(index=3, gold_label="OCP", predicted_label="SRP"),
    ]

    evaluator = ole.ClassificationEvaluator(labels=["SRP", "OCP"])
    result = evaluator.evaluate(examples)

    # 2/4 correct
    assert result.accuracy == pytest.approx(0.5)

    cm = result.confusion_matrix
    # gold=SRP: 1 SRP, 1 OCP
    assert cm["SRP"]["SRP"] == 1
    assert cm["SRP"]["OCP"] == 1
    # gold=OCP: 1 OCP, 1 SRP
    assert cm["OCP"]["OCP"] == 1
    assert cm["OCP"]["SRP"] == 1


def test_confusion_matrix_plotter_creates_file(tmp_path):
    labels = ["DIP", "ISP", "LSP", "OCP", "SRP"]
    # Simple confusion: identity matrix (perfect accuracy)
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for l in labels:
        confusion[l][l] = 5

    plotter = ole.ConfusionMatrixPlotter(labels=labels)
    out_path = tmp_path / "cm.png"

    plotter.plot(confusion, str(out_path), title="Test Confusion")
    assert out_path.exists()
    assert os.path.getsize(out_path) > 0