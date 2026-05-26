"""
Rekarisk Core Module — substance definitions, property engine, units, constants, validation.

This module provides the foundational building blocks for all consequence analysis
capabilities: substance property calculations via DIPPR correlations, comprehensive
unit conversion (SI ↔ Imperial ↔ field units), physical and regulatory constants,
and input validation/sanity checking.
"""

from .constants import *
from .substance import (
    DIPPRParam,
    HazardClass,
    FireClass,
    SubstancePhase,
    Substance,
    compute_mixture_molecular_weight,
    compute_mixture_vapor_pressure,
    is_mixture_flammable,
)
from .substance_db import SubstanceDatabase, get_database
from .units import (
    UnitConverter,
    UnitDef,
    Quantity,
    get_converter,
    convert,
    format_si,
)
from .audit_trail import AuditTrail, AuditEntry, AuditAction
from .project_file import (
    ProjectFile, ProjectMetadata, ProjectData,
    FORMAT_VERSION, FILE_EXTENSION, FILE_FILTER,
)
from .checkpoint import (
    Checkpoint, get_project_id_from_path, get_project_id_from_name,
    get_total_checkpoint_size, purge_all_checkpoints,
)

from .validation import (
    ValidationResult,
    ValidationMessage,
    Severity,
    validate_required,
    validate_positive,
    validate_range,
    validate_percentage,
    validate_temperature,
    validate_pressure,
    validate_gauge_pressure,
    validate_wind_speed,
    validate_stability_class,
    validate_surface_roughness,
    validate_release_rate,
    validate_release_duration,
    validate_hole_size,
    validate_volume,
    validate_temperature_pressure_consistency,
    validate_dispersion_inputs,
    validate_fire_inputs,
    validate_explosion_inputs,
    sanity_check_release,
    sanity_check_concentration,
    register_validator,
    validate_field,
    validate_all,
)
