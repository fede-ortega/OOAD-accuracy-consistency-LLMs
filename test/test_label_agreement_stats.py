import json

import pytest

from consistency_evaluation import label_agreement_stats as las


def test_load_instances_valid(tmp_path):
    instances = [
        {"index": 0, "labels": ["SRP"], "label_agreement": 1.0},
        {"index": 1, "labels": ["OCP"], "label_agreement": 0.5},
    ]
    path = tmp_path / "instances.txt"
    path.write_text(json.dumps(instances), encoding="utf-8")

    loaded = las.load_instances(str(path))
    assert loaded == instances


def test_load_instances_not_list(tmp_path):
    path = tmp_path / "instances.txt"
    path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

    with pytest.raises(ValueError):
        las.load_instances(str(path))


def test_compute_mean_label_agreement_normal_case():
    instances = [
        {"index": 0, "label_agreement": 1.0},
        {"index": 1, "label_agreement": 0.5},
        {"index": 2, "label_agreement": 0.0},
    ]
    mean = las.compute_mean_label_agreement(instances)
    # (1.0 + 0.5 + 0.0) / 3 = 0.5
    assert mean == pytest.approx(0.5)


def test_compute_mean_label_agreement_empty():
    with pytest.raises(ValueError):
        las.compute_mean_label_agreement([])


def test_compute_mean_label_agreement_missing_field():
    instances = [{"index": 0, "labels": []}]
    with pytest.raises(KeyError):
        las.compute_mean_label_agreement(instances)
