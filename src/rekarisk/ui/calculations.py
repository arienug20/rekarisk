"""
Rekarisk UI — Calculation execution handlers.

Extracted from main_window.py for maintainability.
All methods assume they're called with a MainWindow `self` via mixin.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from PyQt6.QtWidgets import QMessageBox

from ..core.audit_trail import AuditAction


class CalculationMixin:
    """Calculation execution methods, mixed into MainWindow."""

    def _execute_source_term(self, calc_type: str, params: dict, results_panel):
        """Execute a source term calculation and display results."""
        try:
            if calc_type == "orifice":
                from ..models.source_term.orifice import (
                    OrificeInput, calculate_orifice,
                )
                inp = OrificeInput(
                    Cd=params.get("Cd", 0.62),
                    d_hole=params.get("d_hole", 0.025),
                    P_upstream=params.get("P_upstream", 5e5),
                    P_downstream=params.get("P_downstream", 101325),
                    T=params.get("T", 300),
                    phase=params.get("phase", "auto"),
                    rho=params.get("rho", 1.2),
                    molecular_weight=params.get("molecular_weight", 0.029),
                    cp_cv_ratio=params.get("cp_cv_ratio", 1.4),
                    h_liquid_head=params.get("h_liquid_head", 0),
                    duration=params.get("duration"),
                )
                result = calculate_orifice(inp)
                results_panel.show_orifice_result(result)

                # Cache for downstream modules
                self._last_source_term_result = {
                    "calc_type": "orifice",
                    "mass_flow_rate": result.mdot_initial,
                    "exit_velocity": result.velocity,
                    "temperature": params.get("T", 300),
                    "molecular_weight": params.get("molecular_weight", 0.029),
                    "phase": result.phase,
                    "hole_diameter": params.get("d_hole", 0.025),
                    "is_choked": result.is_choked,
                    "total_mass": result.total_mass,
                }

            elif calc_type == "vessel":
                from ..models.source_term.vessel_depressur import (
                    VesselInput, calculate_vessel_blowdown,
                )
                inp = VesselInput(
                    V=params.get("V", 10),
                    A_wall=params.get("A_wall", 25),
                    P_initial=params.get("P_initial", 6e5),
                    T_initial=params.get("T_initial", 300),
                    orifice_d=params.get("orifice_d", 0.025),
                    Cd=params.get("Cd", 0.62),
                    t_max=params.get("t_max", 60),
                    P_target=params.get("P_target", 101325),
                    phase=params.get("phase", "gas"),
                    mode=params.get("mode", "api521"),
                    molecular_weight=params.get("molecular_weight", 0.029),
                    cp_cv_ratio=params.get("cp_cv_ratio", 1.4),
                    rho_liquid=params.get("rho_liquid", 1000),
                )
                result = calculate_vessel_blowdown(inp)
                results_panel.show_vessel_result(result)

                # Cache — use average mass flow rate for downstream
                avg_mdot = (
                    sum(result.mdot) / len(result.mdot) if result.mdot else 0
                )
                self._last_source_term_result = {
                    "calc_type": "vessel",
                    "mass_flow_rate": avg_mdot,
                    "exit_velocity": result.mdot[0] / (
                        3.14159 * (params.get("orifice_d", 0.025) / 2) ** 2 * max(result.m[0], 0.1) / max(result.V[0], 1)
                    ) if result.mdot else 0,
                    "temperature": result.T[0] if result.T else 300,
                    "molecular_weight": params.get("molecular_weight", 0.029),
                    "phase": params.get("phase", "gas"),
                    "hole_diameter": params.get("orifice_d", 0.025),
                    "total_mass": result.total_mass_released,
                }

            elif calc_type == "pipe":
                from ..models.source_term.pipe_flow import (
                    PipeInput, calculate_pipe_flow,
                )
                inp = PipeInput(
                    D=params.get("D", 0.1),
                    L=params.get("L", 100),
                    P_inlet=params.get("P_inlet", 5e5),
                    P_outlet=params.get("P_outlet", 101325),
                    T=params.get("T", 300),
                    phase=params.get("phase", "gas"),
                    rho=params.get("rho", 1.2),
                    molecular_weight=params.get("molecular_weight", 0.029),
                    cp_cv_ratio=params.get("cp_cv_ratio", 1.4),
                    roughness=params.get("roughness", 4.5e-5),
                )
                result = calculate_pipe_flow(inp)
                results_panel.show_pipe_result(result)

                self._last_source_term_result = {
                    "calc_type": "pipe",
                    "mass_flow_rate": result.mdot,
                    "exit_velocity": result.velocity,
                    "temperature": params.get("T", 300),
                    "molecular_weight": params.get("molecular_weight", 0.029),
                    "phase": result.flow_regime,
                    "total_mass": None,
                }

            elif calc_type == "psv":
                from ..models.source_term.relief_valve import (
                    ReliefValveInput, calculate_relief_valve,
                )
                inp = ReliefValveInput(
                    W_required=params.get("W_required", 1.0),
                    P_set=params.get("P_set", 5e5),
                    P_back=params.get("P_back", 101325),
                    T=params.get("T", 300),
                    molecular_weight=params.get("molecular_weight", 0.029),
                    cp_cv_ratio=params.get("cp_cv_ratio", 1.4),
                    rho=params.get("rho", 1.2),
                    valve_type=params.get("valve_type", "conventional"),
                    overpressure_pct=params.get("overpressure_pct", 10),
                    rupture_disk=params.get("rupture_disk_used", False),
                )
                result = calculate_relief_valve(inp)
                results_panel.show_psv_result(result)

                self._last_source_term_result = {
                    "calc_type": "psv",
                    "mass_flow_rate": result.W_relieving,
                    "temperature": params.get("T", 300),
                    "molecular_weight": params.get("molecular_weight", 0.029),
                    "phase": "gas",
                }

            elif calc_type == "pool":
                from ..models.source_term.pool_evaporation import (
                    PoolInput, calculate_pool_evaporation,
                )
                inp = PoolInput(
                    spill_mass=params.get("spill_mass", 1000),
                    rho_l=params.get("rho_l", 1000),
                    boiling_point=params.get("boiling_point", 373.15),
                    heat_of_vaporization=params.get("heat_of_vaporization", 2.26e6),
                    vapor_pressure=params.get("vapor_pressure", 3000),
                    molecular_weight=params.get("molecular_weight", 0.018),
                    T_ambient=params.get("T_ambient", 298.15),
                    wind_speed=params.get("wind_speed", 3.0),
                    surface=params.get("surface", "land"),
                    bunded_area=params.get("bunded_area"),
                    t_max=params.get("t_max", 120),
                )
                result = calculate_pool_evaporation(inp)
                results_panel.show_pool_result(result)

                self._last_source_term_result = {
                    "calc_type": "pool",
                    "mass_flow_rate": result.avg_evap_rate * result.pool_area[-1] if result.pool_area else 0,
                    "temperature": params.get("T_ambient", 298.15),
                    "molecular_weight": params.get("molecular_weight", 0.018),
                    "phase": "gas",
                }
            else:
                self.statusBar().showMessage(
                    f"Unknown source term type: {calc_type}", 3000
                )
                return

            # Store result in project data
            self._project_data.setdefault("results", []).append({
                "module": "source_term",
                "calc_type": calc_type,
                "inputs": params,
                "summary": self._last_source_term_result,
                "timestamp": datetime.now().isoformat(),
            })

            self._audit_trail.log(
                action=AuditAction.RUN,
                module="source_term",
                description=f"Source term calculated: {calc_type}",
                details={"calc_type": calc_type},
            )

            self.set_dirty()
            self._action_export.setEnabled(True)
            self.statusBar().showMessage(
                f"✅ Source Term ({calc_type}) calculation complete", 5000
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Calculation Error",
                f"Source Term ({calc_type}) failed:\n{e}"
            )

    def _execute_dispersion(self, params: dict, results_panel):
        """Execute a dispersion calculation."""
        try:
            from ..models.dispersion.dispersion_dispatcher import (
                ReleaseInfo, WeatherInfo, DispersionDispatcher,
            )

            release = ReleaseInfo(
                mass_rate=params.get("source_rate", 1.0),
                mass=params.get("source_mass", 0),
                duration=params.get("duration", 0),
                substance_density=params.get("cloud_density", 1.2),
                molecular_weight=params.get("molecular_weight", 29.0),
                temperature=params.get("temperature", 298.15),
                phase=params.get("phase", "gas"),
                release_height=params.get("release_height", 0),
                release_velocity=params.get("exit_velocity", 0),
                release_diameter=params.get("release_diameter", 0),
                heat_release_rate=params.get("heat_release_rate", 0),
            )

            weather = WeatherInfo(
                wind_speed=params.get("wind_speed", 3.0),
                stability_class=params.get("stability_class", "D"),
                terrain_type=params.get("terrain_type", "rural"),
            )

            dispatcher = DispersionDispatcher()
            result = dispatcher.dispatch(release, weather)
            results_panel.display_result(result)

            # Cache for downstream
            self._last_dispersion_result = {
                "model_used": result.model_used if hasattr(result, "model_used") else "unknown",
                "concentrations": result.concentrations if hasattr(result, "concentrations") else None,
                "x_grid": result.x_grid if hasattr(result, "x_grid") else None,
                "max_concentration": result.max_concentration if hasattr(result, "max_concentration") else 0,
                "params": params,
            }

            self._project_data.setdefault("results", []).append({
                "module": "dispersion",
                "inputs": params,
                "summary": {
                    "model": self._last_dispersion_result["model_used"],
                    "max_conc": self._last_dispersion_result["max_concentration"],
                },
                "timestamp": datetime.now().isoformat(),
            })

            self._audit_trail.log(
                action=AuditAction.RUN,
                module="dispersion",
                description="Dispersion calculation complete",
                details={"model": self._last_dispersion_result["model_used"]},
            )

            self.set_dirty()
            self._action_export.setEnabled(True)
            self.statusBar().showMessage("✅ Dispersion calculation complete", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Dispersion failed:\n{e}")

    def _execute_fire(self, model_type: str, params: dict, results_panel):
        """Execute a fire consequence calculation."""
        try:
            if model_type == "pool_fire":
                from ..models.fire.pool_fire import PoolFireInput, calculate_pool_fire
                inp = PoolFireInput(
                    pool_diameter=params.get("pool_diameter", 10),
                    substance=params.get("substance", "gasoline"),
                    burning_rate=params.get("burning_rate"),
                    heat_of_combustion=params.get("heat_of_combustion"),
                    radiative_fraction=params.get("radiative_fraction", 0.35),
                    wind_speed=params.get("wind_speed", 3.0),
                    ambient_temperature=params.get("ambient_temperature", 293.15),
                    relative_humidity=params.get("relative_humidity", 0.5),
                )
                result = calculate_pool_fire(inp)
                results_panel.display_pool_fire_result(result)

            elif model_type == "jet_fire":
                from ..models.fire.jet_fire import JetFireInput, calculate_jet_fire
                inp = JetFireInput(
                    orifice_diameter=params.get("orifice_diameter", 0.05),
                    discharge_velocity=params.get("discharge_velocity", 100),
                    mass_flow_rate=params.get("mass_flow_rate"),
                    substance=params.get("substance", "propane"),
                    heat_of_combustion=params.get("heat_of_combustion"),
                    radiative_fraction=params.get("radiative_fraction", 0.30),
                    wind_speed=params.get("wind_speed", 3.0),
                    release_direction=params.get("release_direction", "horizontal"),
                    ambient_temperature=params.get("ambient_temperature", 293.15),
                    relative_humidity=params.get("relative_humidity", 0.5),
                    discharge_density=params.get("discharge_density"),
                )
                jet_model = params.get("jet_fire_model", "multipoint")
                result = calculate_jet_fire(inp, model=jet_model)
                results_panel.display_jet_fire_result(result)

            elif model_type == "bleve":
                from ..models.fire.bleve import BLEVEInput, calculate_bleve
                inp = BLEVEInput(
                    vessel_mass=params.get("vessel_mass", 5000),
                    substance=params.get("substance", "propane"),
                    heat_of_combustion=params.get("heat_of_combustion"),
                    radiative_fraction=params.get("radiative_fraction", 0.35),
                    ambient_temperature=params.get("ambient_temperature", 293.15),
                    relative_humidity=params.get("relative_humidity", 0.5),
                )
                result = calculate_bleve(inp)
                results_panel.display_bleve_result(result)

            elif model_type == "flash_fire":
                from ..models.fire.flash_fire import FlashFireInput, calculate_flash_fire
                inp = FlashFireInput(
                    substance=params.get("substance", "methane"),
                    lfl=params.get("lfl", 0.05),
                    ufl=params.get("ufl", 0.15),
                    cloud_volume=params.get("cloud_volume", 1000),
                    ambient_temperature=params.get("ambient_temperature", 293.15),
                )
                result = calculate_flash_fire(inp)
                results_panel.display_flash_fire_result(result)
            else:
                self.statusBar().showMessage(f"Unknown fire type: {model_type}", 3000)
                return

            self._last_fire_result = {"model_type": model_type, "params": params}

            self._project_data.setdefault("results", []).append({
                "module": "fire",
                "calc_type": model_type,
                "inputs": params,
                "timestamp": datetime.now().isoformat(),
            })

            self._audit_trail.log(
                action=AuditAction.RUN,
                module="fire",
                description=f"Fire calculation: {model_type}",
            )

            self.set_dirty()
            self._action_export.setEnabled(True)
            self.statusBar().showMessage(f"✅ Fire ({model_type}) calculation complete", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Fire ({model_type}) failed:\n{e}")

    def _execute_explosion(self, params: dict, results_panel):
        """Execute an explosion consequence calculation."""
        try:
            results = []

            if params.get("tnt_enabled", True):
                from ..models.explosion.tnt_equivalency import TNTInput, calculate_tnt_equivalency
                inp = TNTInput(
                    mass_flammable=params.get("mass_flammable", 1000),
                    heat_of_combustion=params.get("heat_of_combustion", 50.35e6),
                    efficiency=params.get("tnt_efficiency", 0.05),
                )
                tnt_result = calculate_tnt_equivalency(inp)
                results.append(("TNT", tnt_result))

            if params.get("tno_enabled", True):
                from ..models.explosion.tno_multi_energy import TNOInput, calculate_tno_multi_energy
                inp = TNOInput(
                    confinement_class=params.get("tno_confinement_class", "2D"),
                    blast_strength=params.get("tno_blast_strength", 7),
                    energy=params.get("tno_energy", 1e9),
                )
                tno_result = calculate_tno_multi_energy(inp)
                results.append(("TNO", tno_result))

            if params.get("bst_enabled", True):
                from ..models.explosion.baker_strehlow import BSTInput, calculate_bst
                inp = BSTInput(
                    mass_flammable=params.get("bst_mass_flammable", 1000),
                    heat_of_combustion=params.get("bst_heat_of_combustion", 50.35e6),
                    fuel_reactivity=params.get("fuel_reactivity", "medium"),
                    confinement_class=params.get("bst_confinement_class", "2D"),
                    congestion_level=params.get("bst_congestion_level", "medium"),
                    flame_mach=params.get("flame_mach"),
                )
                bst_result = calculate_bst(inp)
                results.append(("BST", bst_result))

            if results:
                # Build dict of model_name -> result for the panel
                result_dict = {name: res for name, res in results}
                results_panel.display_results(result_dict)

            self._last_explosion_result = {"params": params, "num_models": len(results)}

            self._project_data.setdefault("results", []).append({
                "module": "explosion",
                "inputs": params,
                "timestamp": datetime.now().isoformat(),
            })

            self._audit_trail.log(
                action=AuditAction.RUN,
                module="explosion",
                description=f"Explosion: {len(results)} models",
            )

            self.set_dirty()
            self._action_export.setEnabled(True)
            self.statusBar().showMessage("✅ Explosion calculation complete", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Explosion failed:\n{e}")

    def _execute_vulnerability(self, params: dict, results_panel):
        """Execute a vulnerability assessment."""
        try:
            from ..models.vulnerability.vulnerability_calculator import (
                VulnerabilityInput, calculate_vulnerability,
            )

            inp = VulnerabilityInput(
                hazard_type=params.get("hazard_type", "toxic"),
                substance=params.get("substance"),
                thermal_model=params.get("thermal_model"),
                overpressure_model=params.get("overpressure_model"),
                exposure_time=params.get("exposure_time", 30),
                manual_intensity=params.get("manual_intensity", 100),
                intensity_source=params.get("intensity_source", "manual"),
                use_shelter=params.get("use_shelter", False),
                ach=params.get("ach"),
                x_min=params.get("x_min", 10),
                x_max=params.get("x_max", 5000),
                y_min=params.get("y_min", -500),
                y_max=params.get("y_max", 500),
                n_x=params.get("n_x", 100),
                n_y=params.get("n_y", 100),
            )

            result = calculate_vulnerability(inp)
            results_panel.set_result(result)

            self._last_vulnerability_result = {
                "hazard_type": params.get("hazard_type"),
                "result": result,
            }

            self._project_data.setdefault("results", []).append({
                "module": "vulnerability",
                "inputs": params,
                "timestamp": datetime.now().isoformat(),
            })

            self._audit_trail.log(
                action=AuditAction.RUN,
                module="vulnerability",
                description=f"Vulnerability: {params.get('hazard_type')}",
            )

            self.set_dirty()
            self._action_export.setEnabled(True)
            self.statusBar().showMessage("✅ Vulnerability assessment complete", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Vulnerability failed:\n{e}")

    def _execute_qra(self, panel, results_panel):
        """Execute a QRA calculation using end-to-end QRAPipeline.

        Reads actual data from the QRAPanel tabs:
        - Event Tree → scenarios + leak frequencies
        - Frequency Tab → base frequency + modifiers
        - Population Tab → population grid → receptors + workers
        - Risk Criteria → IR thresholds
        """
        try:
            from ..models.qra.qra_pipeline import (
                QRAPipeline, IsoSection, ReceptorPoint, WorkerGroup,
            )

            # ── 1. Read panel data ──
            scenarios = panel.get_scenarios() if hasattr(panel, 'get_scenarios') else []
            pop_grid = panel.get_population_grid() if hasattr(panel, 'get_population_grid') else []
            base_freq = panel.get_frequency() if hasattr(panel, 'get_frequency') else 0.0
            ir_standard = panel.get_ir_standard() if hasattr(panel, 'get_ir_standard') else "hse_uk"
            event_tree = panel.get_event_tree() if hasattr(panel, 'get_event_tree') else None

            params = {
                "frequency": base_freq,
                "scenarios": scenarios,
                "population_grid": pop_grid,
                "ir_standard": ir_standard,
                "event_tree": event_tree,
            }

            # ── 2. Build IsoSections ──
            # If event tree has scenario data, derive sections from it.
            # Otherwise use a sensible default for standalone QRA.
            iso_sections = []
            if scenarios:
                # Group scenarios by section name if available
                section_names = set()
                for s in scenarios:
                    name = getattr(s, 'section', None) or s.get('section', 'Process Area') if isinstance(s, dict) else 'Process Area'
                    section_names.add(name)
                for name in section_names:
                    iso_sections.append(IsoSection(
                        name=name,
                        P=50e5, T=320.0, volume=10.0,
                        composition="natural_gas",
                        molecular_weight=18.0,
                        fill_fraction=0.0,
                        x=0, y=0, n_equipment=3,
                    ))
            else:
                # Default section
                iso_sections = [
                    IsoSection(
                        name="Process Area",
                        P=50e5, T=320.0, volume=10.0,
                        composition="natural_gas",
                        molecular_weight=18.0,
                        fill_fraction=0.0,
                        x=0, y=0, n_equipment=3,
                    ),
                ]

            # ── 3. Build receptors & workers from population grid ──
            receptors = []
            workers = []
            if pop_grid and any(sum(row) > 0 for row in pop_grid):
                cell_size = 50.0
                n_rows = len(pop_grid)
                for i, row in enumerate(pop_grid):
                    for j, pop in enumerate(row):
                        if pop > 0:
                            x = j * cell_size
                            y = (n_rows // 2 - i) * cell_size
                            receptors.append(ReceptorPoint(
                                label=f"R{i},{j}", x=x, y=y,
                            ))
                            workers.append(WorkerGroup(
                                name=f"Grid {i},{j}",
                                count=int(pop),
                                locations=[(x, y, 0.5)],
                            ))
            else:
                # Default receptors at standard distances
                for d in [20, 50, 100, 200, 500]:
                    receptors.append(ReceptorPoint(
                        label=f"R{d}m", x=d, y=0,
                    ))
                workers = [
                    WorkerGroup(name="Operator", count=3,
                               locations=[(20, 0, 0.5)]),
                    WorkerGroup(name="Maintenance", count=2,
                               locations=[(50, 0, 0.3)]),
                ]

            # ── 4. Run pipeline ──
            pipeline = QRAPipeline(
                iso_sections=iso_sections,
                receptor_grid=receptors,
                worker_groups=workers,
                shelter_ach=1.0,
            )
            qra_result = pipeline.run()

            # ── 5. Build LSIR contour grid from pipeline LSIR ──
            # Convert dict-based lsir_grid into structured arrays for contour
            ir_thresholds = {}
            if qra_result.lsir_grid:
                coords = list(qra_result.lsir_grid.keys())
                xs = sorted(set(c[0] for c in coords))
                ys = sorted(set(c[1] for c in coords))
                import numpy as np
                Z = np.full((len(ys), len(xs)), 1e-12)
                for (x, y), v in qra_result.lsir_grid.items():
                    xi = xs.index(x)
                    yi = ys.index(y)
                    Z[yi, xi] = max(v, 1e-12)
                # Build IR thresholds from LSIR grid
                from ..models.qra.individual_risk import RISK_THRESHOLDS
                for label, thresh in RISK_THRESHOLDS.items():
                    # Find minimum distance where LSIR > threshold
                    min_d = None
                    for d in [20, 50, 100, 200, 500]:
                        vals = [v for (x, y), v in qra_result.lsir_grid.items()
                                if abs((x**2 + y**2)**0.5 - d) < 30 and v > thresh]
                        if vals:
                            min_d = d
                            break
                    if min_d is not None:
                        ir_thresholds[label] = min_d

            # ── 6. Build results data ──
            results_data = {
                "name": "QRA Analysis",
                "pipeline_result": qra_result,
                "lsir_data": {f"({x},{y})": v
                              for (x, y), v in qra_result.lsir_grid.items()},
                "irpa_data": dict(qra_result.irpa_table),
                "pll_total": qra_result.pll_total,
                "pll_detail": {wg: qra_result.irpa_table.get(wg, 0)
                               for wg in qra_result.irpa_table},
                "fn_data": {"n": [p[0] for p in qra_result.fn_pairs],
                            "f": [p[1] for p in qra_result.fn_pairs]} if qra_result.fn_pairs else None,
                "ir_thresholds": ir_thresholds,
                "dominant": qra_result.dominant,
                "alarp": dict(qra_result.alarp) if qra_result.alarp else {},
                "scenario_count": qra_result.scenario_count,
                "warnings": qra_result.warnings,
            }

            # Attach structured grid if we have enough points
            if qra_result.lsir_grid and len(qra_result.lsir_grid) >= 4:
                results_data["lsir_grid_structured"] = {
                    "x_coords": xs,
                    "y_coords": ys,
                    "values": Z.tolist(),
                }

            # Display results
            if hasattr(results_panel, 'set_result'):
                results_panel.set_result(results_data)
            elif hasattr(results_panel, 'panel') and hasattr(results_panel.panel(), 'set_result'):
                results_panel.panel().set_result(results_data)

            self._project_data.setdefault("results", []).append({
                "name": "QRA Analysis",
                "type": "qra",
                "module": "qra",
                "inputs": params,
                "lsir_data": {f"({x},{y})": v
                              for (x, y), v in qra_result.lsir_grid.items()},
                "irpa_data": dict(qra_result.irpa_table),
                "pll_total": qra_result.pll_total,
                "pll_detail": {wg: qra_result.irpa_table.get(wg, 0)
                               for wg in qra_result.irpa_table},
                "alarp": dict(qra_result.alarp) if qra_result.alarp else {},
                "ir_thresholds": ir_thresholds,
                "fn_data": {"n": [p[0] for p in qra_result.fn_pairs],
                            "f": [p[1] for p in qra_result.fn_pairs]} if qra_result.fn_pairs else None,
                "dominant": qra_result.dominant,
                "scenario_count": qra_result.scenario_count,
                "warnings": qra_result.warnings,
                "timestamp": datetime.now().isoformat(),
            })

            self._audit_trail.log(
                action=AuditAction.RUN,
                module="qra",
                description=f"QRA pipeline: {qra_result.scenario_count} scenarios, "
                           f"PLL={qra_result.pll_total:.2e}",
            )

            self.set_dirty()
            self._action_export.setEnabled(True)
            self.statusBar().showMessage(
                f"✅ QRA complete: {qra_result.scenario_count} scenarios, "
                f"PLL={qra_result.pll_total:.2e}/yr", 5000,
            )

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"QRA failed:\n{e}")

    def _execute_domino(self, params: dict, results_panel):
        """Execute a domino / escalation analysis."""
        try:
            from ..models.qra.domino import (
                Equipment, EquipmentType, SubstanceCategory,
                PrimaryEvent, run_domino_analysis,
            )

            # Build equipment list
            eq_type_map = {v: k for k, v in [
                (EquipmentType.ATMOSPHERIC_TANK, "atmospheric_tank"),
                (EquipmentType.PRESSURE_VESSEL, "pressure_vessel"),
                (EquipmentType.REACTOR, "reactor"),
                (EquipmentType.HEAT_EXCHANGER, "heat_exchanger"),
                (EquipmentType.PIPELINE, "pipeline"),
                (EquipmentType.COLUMN, "column"),
                (EquipmentType.SEPARATOR, "separator"),
                (EquipmentType.PUMP, "pump"),
                (EquipmentType.COMPRESSOR, "compressor"),
                (EquipmentType.FIN_FAN_COOLER, "fin_fan_cooler"),
                (EquipmentType.STRUCTURE, "structure"),
            ]}
            cat_map = {v: k for k, v in [
                (SubstanceCategory.FLAMMABLE_LIQUID, "flammable_liquid"),
                (SubstanceCategory.FLAMMABLE_GAS, "flammable_gas"),
                (SubstanceCategory.FLAMMABLE_LPG, "flammable_lpg"),
                (SubstanceCategory.TOXIC, "toxic"),
                (SubstanceCategory.REACTIVE, "reactive"),
                (SubstanceCategory.INERT, "inert"),
            ]}

            equipment_list = []
            for eq_data in params.get("equipment", []):
                eq = Equipment(
                    id=eq_data["id"],
                    name=eq_data.get("name", eq_data["id"]),
                    equipment_type=eq_type_map.get(eq_data.get("equipment_type"), EquipmentType.PRESSURE_VESSEL),
                    substance=eq_data.get("substance", "Unknown"),
                    substance_category=cat_map.get(eq_data.get("substance_category"), SubstanceCategory.FLAMMABLE_LIQUID),
                    inventory_kg=eq_data.get("inventory_kg", 0),
                    x=eq_data.get("x", 0), y=eq_data.get("y", 0),
                    diameter=eq_data.get("diameter", 2),
                    height=eq_data.get("height", 5),
                    operating_pressure=eq_data.get("operating_pressure", 1),
                    is_insulated=eq_data.get("is_insulated", False),
                    has_deluge=eq_data.get("has_deluge", False),
                )
                equipment_list.append(eq)

            primary = PrimaryEvent(
                equipment_id=params["primary_equipment_id"],
                event_type=params.get("event_type", "pool_fire"),
                frequency=params.get("frequency", 1e-6),
                thermal_power_kw=params.get("thermal_power_kw", 0),
                tnt_mass_kg=params.get("tnt_mass_kg", 0),
                fireball_radius_m=params.get("fireball_radius_m", 0),
                source_height_m=params.get("source_height_m", 0),
                pool_radius_m=params.get("pool_radius_m", 0),
            )

            result = run_domino_analysis(
                primary_event=primary,
                equipment_list=equipment_list,
                max_escalation_order=params.get("max_escalation_order", 3),
                response_time_min=params.get("response_time_min", 10),
                include_thermal=params.get("include_thermal", True),
                include_overpressure=params.get("include_overpressure", True),
                include_impingement=params.get("include_impingement", True),
            )

            results_panel.set_result(result)

            self._project_data.setdefault("results", []).append({
                "module": "domino",
                "inputs": params,
                "timestamp": datetime.now().isoformat(),
            })

            self._audit_trail.log(
                action=AuditAction.RUN,
                module="domino",
                description=f"Domino analysis: {result.summary.get('domino_scenarios', 0)} scenarios",
            )

            self.set_dirty()
            self._action_export.setEnabled(True)
            self.statusBar().showMessage("✅ Domino analysis complete", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Domino analysis failed:\n{e}")


