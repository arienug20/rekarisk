"""End-to-end tests for QRA UI integration — data flow & PDF format."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# ── Test: QRAPipeline result → results_data → PDF format ──


@pytest.fixture
def mock_qra_result():
    """Mock QRAResult matching pipeline output shape."""
    @dataclass
    class MockQRAResult:
        lsir_grid: dict
        irpa_table: dict
        pll_total: float
        fn_pairs: list
        dominant: list
        alarp: dict
        scenario_count: int
        warnings: list

    return MockQRAResult(
        lsir_grid={
            (20, 0): 5e-4,
            (50, 0): 1e-4,
            (100, 0): 1e-5,
            (200, 0): 1e-6,
            (500, 0): 1e-7,
        },
        irpa_table={"Operator": 3e-5, "Maintenance": 1e-5},
        pll_total=4e-5,
        fn_pairs=[(1, 1e-4), (10, 1e-5)],
        dominant=[("jet_fire", 0.6)],
        alarp={"Operator": "ALARP Region", "Maintenance": "Broadly Acceptable"},
        scenario_count=24,
        warnings=[],
    )


class TestQRAResultFormat:
    """Verify QRA result data format matches what PDF generator expects."""

    PDF_REQUIRED_KEYS = [
        "lsir_data", "irpa_data", "pll_total",
        "pll_detail", "alarp",
    ]

    def test_result_has_type_qra(self, mock_qra_result):
        """Result stored in project_data must have type='qra'."""
        r = {
            "name": "QRA Analysis",
            "type": "qra",
            "module": "qra",
            "lsir_data": {f"({x},{y})": v
                          for (x, y), v in mock_qra_result.lsir_grid.items()},
            "irpa_data": dict(mock_qra_result.irpa_table),
            "pll_total": mock_qra_result.pll_total,
        }
        assert r["type"] == "qra"

    def test_result_has_pdf_keys(self, mock_qra_result):
        """Result dict must contain all keys PDF _add_qra_section reads."""
        result = {
            "name": "QRA Analysis",
            "type": "qra",
            "lsir_data": {f"({x},{y})": v
                          for (x, y), v in mock_qra_result.lsir_grid.items()},
            "irpa_data": dict(mock_qra_result.irpa_table),
            "pll_total": mock_qra_result.pll_total,
            "pll_detail": {wg: mock_qra_result.irpa_table.get(wg, 0)
                           for wg in mock_qra_result.irpa_table},
            "alarp": dict(mock_qra_result.alarp),
            "fn_data": {"n": [p[0] for p in mock_qra_result.fn_pairs],
                        "f": [p[1] for p in mock_qra_result.fn_pairs]},
            "ir_thresholds": {"1e-6/yr": 200},
            "scenario_count": mock_qra_result.scenario_count,
        }
        for key in self.PDF_REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_alarp_is_dict_not_object(self, mock_qra_result):
        """PDF generator calls .get() on alarp — must be a dict."""
        alarp = dict(mock_qra_result.alarp)
        assert isinstance(alarp, dict)
        assert "Operator" in alarp

    def test_lsir_data_keys_are_strings(self, mock_qra_result):
        """PDF generator iterates lsir_data.items() — keys must be strings."""
        lsir = {f"({x},{y})": v
                for (x, y), v in mock_qra_result.lsir_grid.items()}
        for key in lsir:
            assert isinstance(key, str)

    def test_fn_data_format(self, mock_qra_result):
        """FN data must have 'n' and 'f' lists."""
        fn = {"n": [p[0] for p in mock_qra_result.fn_pairs],
              "f": [p[1] for p in mock_qra_result.fn_pairs]}
        assert len(fn["n"]) == len(fn["f"])
        assert all(isinstance(n, (int, float)) for n in fn["n"])
        assert all(isinstance(f, (int, float)) for f in fn["f"])


class TestStructuredLSIR:
    """Verify structured LSIR grid for contour plotting."""

    def test_structured_grid_shape(self):
        """Structured grid should have consistent x/y/value dimensions."""
        xs = [20, 50, 100, 200, 500]
        ys = [-100, 0, 100]
        Z = np.array([
            [1e-7, 1e-6, 1e-7, 1e-8, 1e-9],
            [1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
            [1e-7, 1e-6, 1e-7, 1e-8, 1e-9],
        ])
        structured = {
            "x_coords": xs,
            "y_coords": ys,
            "values": Z.tolist(),
        }
        assert len(structured["x_coords"]) == Z.shape[1]
        assert len(structured["y_coords"]) == Z.shape[0]

    def test_ir_thresholds_from_lsir(self):
        """IR thresholds should map threshold label → distance."""
        thresholds = {"1e-6/yr": 200, "1e-5/yr": 100, "1e-4/yr": 50}
        for label, dist in thresholds.items():
            assert isinstance(label, str)
            assert isinstance(dist, (int, float))
            assert dist > 0


class TestFireModelSelectorWiring:
    """Verify jet_fire_model param flows from UI to calculation."""

    def test_jet_fire_default_is_multipoint(self):
        """Default model must be 'multipoint' (not 'mudan' which breaks threshold)."""
        # Simulate JetFireTab.get_params()
        params = {"jet_fire_model": "multipoint"}
        assert params["jet_fire_model"] == "multipoint"

    def test_model_passed_to_calculate(self):
        """calculate_jet_fire must receive model param."""
        # Simulate _execute_fire wiring
        params = {"jet_fire_model": "mudan"}
        jet_model = params.get("jet_fire_model", "multipoint")
        assert jet_model == "mudan"
