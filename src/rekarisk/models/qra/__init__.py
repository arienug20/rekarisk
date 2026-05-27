"""
Rekarisk QRA Models — Quantitative Risk Assessment.

Implements comprehensive QRA methodology following CCPS (Center for
Chemical Process Safety), TNO Purple Book, HSE UK, and API RP 752/753
guidance.

Modules:
  - failure_frequency: Equipment failure frequency database & lookup
  - event_tree: Event tree analysis with branching logic
  - ignition_prob: Ignition probability models (Cox/Lees/Amey, TNO, HSE)
  - individual_risk: Individual Risk Per Annum (IRPA) calculation
  - societal_risk: Societal risk FN curves and ALARP assessment
  - risk_matrix: ISO 17776 / API 5×5 risk matrix classification
  - domino: Escalation & domino effect analysis (Cozzani, CCPS)
"""

from .failure_frequency import (
    FailureFrequencyDB,
    FrequencyClass,
    lookup_frequency,
    combine_frequencies,
    adjust_frequency,
    classify_frequency,
    get_default_db,
)

from .event_tree import (
    Scenario,
    EventTreeNode,
    EventTree,
    create_generic_vessel_tree,
    create_generic_pipeline_tree,
)

from .ignition_prob import (
    IgnitionModel,
    immediate_ignition_probability,
    delayed_ignition_probability,
    explosion_probability,
    default_ignition_data,
)

from .individual_risk import (
    IndividualRiskResult,
    calculate_ir_at_point,
    calculate_ir_grid,
    ir_contour,
    RISK_THRESHOLDS,
)

from .societal_risk import (
    FNData,
    FNCriterion,
    calculate_fn_curve,
    fn_data_to_plot,
    compare_to_criterion,
    FN_CRITERIA,
)

from .risk_matrix import (
    LikelihoodLevel,
    ConsequenceLevel,
    RiskLevel,
    RiskMatrixEntry,
    classify_likelihood,
    classify_consequence,
    classify_consequence_cost,
    risk_level,
    risk_matrix_table,
    risk_matrix_html,
)

from .domino import (
    EquipmentType,
    EscalationVector,
    DamageLevel,
    SubstanceCategory,
    Equipment,
    PrimaryEvent,
    EscalationLink,
    DominoScenario,
    DominoAnalysisResult,
    run_domino_analysis,
    calculate_distance,
    thermal_radiation_at_distance,
    overpressure_at_distance_tnt,
    assess_damage_level,
    calculate_escalation_probability,
    calculate_ttf,
    plot_domino_map,
    plot_escalation_summary,
    plot_domino_chain,
)

from .qra_pipeline import (
    IsoSection,
    HoleSize,
    WeatherScenario,
    ReceptorPoint,
    WorkerGroup,
    QRAResult,
    QRAPipeline,
    DEFAULT_HOLE_SIZES,
)
