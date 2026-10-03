"""
test_data_integrity.py — Automated verification of paper tables and claims manifest
against raw experimental JSON results.
"""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PREV = ROOT / "previous_results"
GEN_TABLES = ROOT / "paper" / "generated_tables"


def test_claims_manifest_exists():
    manifest_path = GEN_TABLES / "claims_manifest.json"
    assert manifest_path.exists(), "claims_manifest.json must exist"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest) >= 170, f"Expected at least 170 claims, got {len(manifest)}"


def test_generated_tables_exist():
    for table_num in [2, 3, 4, 5, 6, 7]:
        tex_path = GEN_TABLES / f"table{table_num}.tex"
        assert tex_path.exists(), f"table{table_num}.tex must exist"
        content = tex_path.read_text(encoding="utf-8")
        assert "\\begin{tabular}" in content
        assert "\\end{tabular}" in content


def test_phase1_table_matches_raw_json():
    src = RESULTS / "phase1_mittag_leffler_data.json"
    assert src.exists()
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    fh = next(d for d in data if "Full-History" in d["method"])
    assert abs(fh["forward_error"] - 0.0010378906374973616) < 1e-8
    assert abs(fh["runtime_ms"] - 48.570699989795685) < 1e-4


def test_phase4_ablation_matches_raw_json():
    src = RESULTS / "phase4_neural_fde_training.json"
    assert src.exists()
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert abs(data["proposed"]["gradient_error_mean"] - 1.6993748557902175) < 1e-4
    assert abs(data["baseline_no_jump"]["gradient_error_mean"] - 69.20937102096742) < 1e-4
    assert abs(data["proposed"]["final_loss_mean"] - 0.0022277463019139757) < 1e-6
    assert abs(data["baseline_no_jump"]["final_loss_mean"] - 0.0024761038319583645) < 1e-6


def test_phase6_scaling_matches_raw_json():
    src = RESULTS / "phase6_long_horizon_scaling_data.json"
    assert src.exists()
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    last_adamem = data["adamem_times"][-1]
    assert last_adamem[0] == 100000
    assert abs(last_adamem[1] - 14.234681000001729) < 1e-4


def test_phase9_graded_matches_raw_json():
    src = RESULTS / "phase9_graded_singularity_quenching.json"
    assert src.exists()
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_vals = data["N_values"]
    run50 = data["runs"]["0.5"]
    assert 400 in n_vals
    idx400 = n_vals.index(400)
    u_err = run50["uniform_errors"][idx400]
    g_err = run50["graded_errors"][idx400]
    ratio = u_err / g_err
    assert abs(ratio - 117.46) < 0.1


def test_phase10_incommensurate_matches_raw_json():
    src = RESULTS / "phase10_incommensurate_multi_order.json"
    assert src.exists()
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    final_loss = data["history_loss"][-1]
    assert abs(final_loss - 4.111204907530919e-05) < 1e-8
    assert abs(data["commensurate_errors"]["0.60"] - 0.001035017056058223) < 1e-6
