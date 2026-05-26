"""
Rekarisk Explosion Models — Vapor Cloud Explosion (VCE) Consequence Analysis.

Provides three explosion blast models:

    tnt_equivalency  — TNT Equivalency with Kingery-Bulmash blast parameters
    tno_multi_energy — TNO Multi-Energy method (confinement/congestion)
    baker_strehlow   — Baker-Strehlow-Tang (BST) method (flame speed)

References:
    - CCPS (1994). Guidelines for Evaluating the Characteristics of
      Vapor Cloud Explosions, Flash Fires, and BLEVEs.
    - TNO Yellow Book (2005). Methods for the Calculation of Physical
      Effects (CPR 14E).
    - Baker, Q.A. et al. (1994). "Vapor Cloud Explosion Analysis."
      Process Safety Progress, 18(4).
    - Kingery, C.N., Bulmash, G. (1984). Airblast Parameters from TNT
      Spherical Air Burst and Hemispherical Surface Burst. ARBRL-TR-02555.
    - UFC 3-340-02 (2008). Structures to Resist the Effects of
      Accidental Explosions.
"""

from .tnt_equivalency import (
    TNTInput,
    TNTResult,
    ExplosionResult,
    calculate_tnt_equivalency,
    overpressure_at_distance,
    distance_to_overpressure,
    distance_to_thresholds,
    kingery_bulmash_overpressure,
    kingery_bulmash_impulse,
    kingery_bulmash_duration,
)
from .tno_multi_energy import (
    TNOInput,
    TNOResult,
    calculate_tno_multi_energy,
    auto_blast_strength,
    tno_overpressure,
    tno_impulse,
    tno_positive_duration,
    blast_strength_description,
)
from .baker_strehlow import (
    BSTInput,
    BSTResult,
    calculate_bst,
    mach_from_confinement_congestion,
    fuel_reactivity_category,
    bst_overpressure,
    bst_impulse,
    REACTIVITY_HIGH,
    REACTIVITY_MEDIUM,
    REACTIVITY_LOW,
)

__all__ = [
    # TNT Equivalency
    "TNTInput",
    "TNTResult",
    "ExplosionResult",
    "calculate_tnt_equivalency",
    "overpressure_at_distance",
    "distance_to_overpressure",
    "distance_to_thresholds",
    "kingery_bulmash_overpressure",
    "kingery_bulmash_impulse",
    "kingery_bulmash_duration",
    # TNO Multi-Energy
    "TNOInput",
    "TNOResult",
    "calculate_tno_multi_energy",
    "auto_blast_strength",
    "tno_overpressure",
    "tno_impulse",
    "tno_positive_duration",
    "blast_strength_description",
    # Baker-Strehlow-Tang
    "BSTInput",
    "BSTResult",
    "calculate_bst",
    "mach_from_confinement_congestion",
    "fuel_reactivity_category",
    "bst_overpressure",
    "bst_impulse",
    "REACTIVITY_HIGH",
    "REACTIVITY_MEDIUM",
    "REACTIVITY_LOW",
]
