"""
Rekarisk UI — Monte Carlo / sensitivity / batch handler methods.

Extracted from main_window.py for maintainability.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from PyQt6.QtWidgets import QMessageBox

from ..analysis.monte_carlo import Uniform, Normal, LogNormal


class MCHandlerMixin:
    """Monte Carlo / batch / sensitivity handler methods, mixed into MainWindow."""

    def _open_batch_runner(self):
        """Open the batch runner dialog."""
        from .batch_dialog import BatchDialog
        from ..analysis.batch_runner import BatchInput

        # Build a model function from the currently active scenario
        model_fn = self._build_active_model_function()
        if model_fn is None:
            QMessageBox.warning(
                self, "Batch Runner",
                "No active scenario to run. Please open a scenario panel first.",
            )
            return

        # Collect scenario templates from project data
        templates = {}
        scenarios = self._project_data.get("scenarios", {})
        for name, sdata in scenarios.items():
            templates[name] = sdata.get("params", {})

        # Weather presets
        weather_opts = [
            {"label": "D – Neutral (3 m/s)", "wind_speed": 3.0, "stability": "D"},
            {"label": "F – Stable (2 m/s)", "wind_speed": 2.0, "stability": "F"},
            {"label": "C – Slightly unstable (4 m/s)", "wind_speed": 4.0, "stability": "C"},
        ]

        dialog = BatchDialog(
            model_function=model_fn,
            scenario_templates=templates if templates else None,
            weather_options=weather_opts,
            parent=self,
        )
        dialog.exec()

    def _open_sensitivity(self):
        """Open sensitivity analysis dialog."""
        from .sensitivity_dialog import SensitivityDialog

        model_fn = self._build_active_model_function()
        if model_fn is None:
            QMessageBox.warning(
                self, "Sensitivity Analysis",
                "No active scenario to run. Please open a scenario panel first.",
            )
            return

        # Try to extract base params from the last active panel
        base_params = self._extract_active_panel_params()

        dialog = SensitivityDialog(
            model_function=model_fn,
            base_params=base_params,
            output_key="max_concentration",
            parent=self,
        )
        dialog.exec()

    def _open_monte_carlo(self):
        """Open Monte Carlo simulation dialog."""
        from .monte_carlo_dialog import MonteCarloDialog
        from ..analysis.monte_carlo import Uniform, Normal

        model_fn = self._build_active_model_function()
        if model_fn is None:
            QMessageBox.warning(
                self, "Monte Carlo",
                "No active scenario to run. Please open a scenario panel first.",
            )
            return

        # Default parameter distributions based on active panel
        params = self._extract_active_panel_distributions()

        active = self._tab_widget.currentWidget()
        panel = getattr(active, 'panel', active) if active else None

        # Determine output keys based on panel type
        if isinstance(panel, QRAPanel):
            output_keys = ["pll_total"]
        else:
            output_keys = ["max_concentration"]

        dialog = MonteCarloDialog(
            model_function=model_fn,
            parameters=params if params else None,
            output_keys=output_keys,
            parent=self,
        )

        dialog.exec()

    # ── Helper: build model function from active panel ────────────

    def _build_active_model_function(self):
        """Build a callable model_function(params_dict) -> result from the
        currently active scenario panel."""
        active = self._tab_widget.currentWidget()
        if active is None:
            return None

        # If wrapped in container, get the actual panel
        panel = getattr(active, 'panel', active)

        if isinstance(panel, SourceTermPanel):
            from ..models.source_term.orifice import OrificeInput, calculate_orifice

            def fn(params):
                inp = OrificeInput(**params)
                return calculate_orifice(inp)
            return fn

        elif isinstance(panel, DispersionPanel):
            from ..models.dispersion.gaussian_plume import PlumeInput, calculate_plume

            def fn(params):
                inp = PlumeInput(**params)
                return calculate_plume(inp)
            return fn

        elif isinstance(panel, FirePanel):
            from ..models.fire.pool_fire import PoolFireInput, calculate_pool_fire

            def fn(params):
                inp = PoolFireInput(**params)
                return calculate_pool_fire(inp)
            return fn

        elif isinstance(panel, ExplosionPanel):
            from ..models.explosion.tnt_equivalency import TNTInput, calculate_tnt_equivalency

            def fn(params):
                inp = TNTInput(**params)
                return calculate_tnt_equivalency(inp)
            return fn

        elif isinstance(panel, QRAPanel):
            from ..models.qra.qra_pipeline import (
                QRAPipeline, IsoSection, ReceptorPoint, WorkerGroup,
            )

            def fn(params):
                """QRA model function for Monte Carlo.

                Accepts multiplier params (freq_mult, ign_mult, wind_mult,
                occ_mult) and returns pipeline result.
                """
                freq_mult = params.get("freq_mult", 1.0)
                ign_mult = params.get("ign_mult", 1.0)
                wind_mult = params.get("wind_mult", 1.0)
                occ_mult = params.get("occ_mult", 1.0)

                import copy
                iso_sections = [
                    IsoSection(
                        name="Process Area", P=50e5, T=320.0, volume=10.0,
                        composition="natural_gas", molecular_weight=18.0,
                        fill_fraction=0.0, x=0, y=0, n_equipment=3,
                    ),
                ]
                receptors = [
                    ReceptorPoint(label=f"R{d}m", x=d, y=0)
                    for d in [20, 50, 100, 200, 500]
                ]
                workers = [
                    WorkerGroup(name="Operator", count=3,
                               locations=[(20, 0, 0.5 * occ_mult)]),
                    WorkerGroup(name="Maintenance", count=2,
                               locations=[(50, 0, 0.3 * occ_mult)]),
                ]

                leak_freq = {k: v * freq_mult for k, v in {
                    "small": 2.4e-5, "medium": 4.0e-6,
                    "large": 4.0e-7, "fullbore": 1.0e-7,
                }.items()}
                imm_ign = {k: v * ign_mult for k, v in {
                    "small": 0.02, "medium": 0.03,
                    "large": 0.05, "fullbore": 0.08,
                }.items()}
                del_ign = {k: v * ign_mult for k, v in {
                    "small": 0.03, "medium": 0.04,
                    "large": 0.06, "fullbore": 0.10,
                }.items()}

                pipeline = QRAPipeline(
                    iso_sections=iso_sections,
                    receptor_grid=receptors,
                    worker_groups=workers,
                    leak_freq_map=leak_freq,
                    imm_ign_map=imm_ign,
                    del_ign_map=del_ign,
                )

                if abs(wind_mult - 1.0) > 0.001:
                    from ..models.qra.qra_pipeline import WeatherScenario
                    pipeline.weathers = [
                        WeatherScenario(
                            name=w.name, wind_speed=w.wind_speed * wind_mult,
                            stability_class=w.stability_class,
                            ambient_temperature=w.ambient_temperature,
                            relative_humidity=w.relative_humidity,
                            direction=w.direction, probability=w.probability,
                        )
                        for w in pipeline.weathers
                    ]

                return pipeline.run()

            return fn

        return None

    def _extract_active_panel_params(self) -> dict:
        """Extract base parameters from the currently active panel."""
        active = self._tab_widget.currentWidget()
        if active is None:
            return {}

        panel = getattr(active, 'panel', active)

        if isinstance(panel, SourceTermPanel):
            return {
                "Cd": 0.62, "d_hole": 0.025, "P_upstream": 5e5,
                "P_downstream": 101325, "T": 298.15, "phase": "gas",
            }
        elif isinstance(panel, DispersionPanel):
            return {
                "source_rate": 1.0, "wind_speed": 3.0,
                "stability_class": "D", "release_height": 0.0,
            }
        elif isinstance(panel, FirePanel):
            return {
                "pool_diameter": 10.0, "substance": "gasoline",
                "wind_speed": 3.0,
            }
        elif isinstance(panel, ExplosionPanel):
            return {
                "mass_flammable": 1000.0, "heat_of_combustion": 46.4e6,
            }
        return {}

    def _extract_active_panel_distributions(self) -> dict:
        """Extract parameter distributions for Monte Carlo."""
        from ..analysis.monte_carlo import Uniform, Normal

        active = self._tab_widget.currentWidget()
        if active is None:
            return {}

        panel = getattr(active, 'panel', active)

        if isinstance(panel, SourceTermPanel):
            return {
                "d_hole": Uniform(0.01, 0.05),
                "P_upstream": Uniform(3e5, 15e5),
                "T": Normal(298.15, 10.0),
            }
        elif isinstance(panel, DispersionPanel):
            return {
                "source_rate": Uniform(0.5, 5.0),
                "wind_speed": Uniform(1.0, 8.0),
                "release_height": Uniform(0.0, 30.0),
            }
        elif isinstance(panel, FirePanel):
            return {
                "pool_diameter": Uniform(3.0, 20.0),
                "wind_speed": Uniform(0.0, 8.0),
            }
        elif isinstance(panel, ExplosionPanel):
            return {
                "mass_flammable": Uniform(500.0, 5000.0),
            }
        elif isinstance(panel, QRAPanel):
            from ..analysis.monte_carlo import LogNormal
            return {
                "freq_mult": LogNormal(mu=0.0, sigma=0.47),  # CV≈50%
                "ign_mult": LogNormal(mu=0.0, sigma=0.30),  # CV≈30%
                "wind_mult": LogNormal(mu=0.0, sigma=0.20),  # CV≈20%
                "occ_mult": LogNormal(mu=0.0, sigma=0.10),  # CV≈10%
            }

