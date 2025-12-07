import json
import os

import numpy as np
import pytest

from adherence_violation_evaluation import spearman_pearson_correlation_from_txt as sp


def test_load_list_json_array(tmp_path):
    path = tmp_path / "a.txt"
    objs = [{"adherence_score": 1}, {"adherence_score": 2}]
    path.write_text(json.dumps(objs), encoding="utf-8")

    items = sp._load_list(str(path))
    assert items == objs


def test_load_list_json_lines(tmp_path):
    path = tmp_path / "a.txt"
    lines = [
        json.dumps({"adherence_score": 1}),
        json.dumps({"adherence_score": 2}),
        "not json at all",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    items = sp._load_list(str(path))
    assert len(items) == 2
    assert items[0]["adherence_score"] == 1
    assert items[1]["adherence_score"] == 2


def test_load_list_empty_error(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("just garbage", encoding="utf-8")

    with pytest.raises(SystemExit):
        sp._load_list(str(path))


def test_coerce_num_handles_valid_and_invalid():
    assert sp._coerce_num(3) == 3.0
    assert sp._coerce_num("4") == 4.0
    assert sp._coerce_num("4.5") == 4.5
    assert sp._coerce_num(None) is None
    assert sp._coerce_num("not-num") is None


def test_extract_metric_produces_floats_or_none():
    items = [
        {"adherence_score": "1"},
        {"adherence_score": 2.5},
        {},
    ]
    vals = sp._extract_metric(items, "adherence_score")
    assert vals == [1.0, 2.5, None]


def test_paired_drops_missing():
    a = [1.0, None, 3.0]
    b = [1.0, 2.0, None]

    xs, ys, dropped = sp._paired(a, b)
    assert xs == [1.0]
    assert ys == [1.0]
    assert dropped == 2


def test_bins_from_integers_generates_centered_edges():
    edges = sp._bins_from_integers(1.0, 5.0)
    # Should be 0.5, 1.5, ..., 5.5
    assert np.allclose(edges, np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]))


def test_plot_matrix5_creates_png(tmp_path):
    xs = [1, 2, 3, 4, 5]
    ys = [1, 2, 3, 4, 5]
    out = tmp_path / "matrix.png"

    sp._plot_matrix5(xs, ys, str(out), r_s=0.9, r_p=0.9)
    assert out.exists()
    assert os.path.getsize(out) > 0


def test_corrs_with_stubbed_scipy(monkeypatch):
    # Pretend scipy is installed
    monkeypatch.setattr(sp, "_HAVE_SCIPY", True)

    # Replace spearmanr / pearsonr with simple stubs
    monkeypatch.setattr(sp, "spearmanr", lambda xs, ys: (0.1, 0.2))
    monkeypatch.setattr(sp, "pearsonr", lambda xs, ys: (0.3, 0.4))

    r_s, p_s, r_p, p_p = sp._corrs([1, 2], [1, 2])
    assert (r_s, p_s, r_p, p_p) == (0.1, 0.2, 0.3, 0.4)


def test_corrs_requires_scipy(monkeypatch):
    monkeypatch.setattr(sp, "_HAVE_SCIPY", False)

    with pytest.raises(SystemExit):
        sp._corrs([1, 2], [1, 2])