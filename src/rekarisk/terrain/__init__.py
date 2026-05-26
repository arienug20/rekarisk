"""
Rekarisk Terrain Module — Obstacle Modeling, LOS Engine, and DEM Support.

Phase 8 foundation module: data structures and core algorithms for
terrain-based consequence modeling — terrain elevation, obstacle
blocking of thermal radiation, and line-of-sight calculations.

Submodules:
    obstacle   — Building/equipment obstacle dataclasses and collections
    los_engine — Ray-casting line-of-sight for thermal radiation blocking
    dem_loader — Digital Elevation Model loader (CSV grid foundation)
"""

from .obstacle import Obstacle, ObstacleCollection
from .los_engine import LOSEngine, LOSResult, SourceGeometry
from .dem_loader import DEMData, ContourLine

__all__ = [
    # Obstacle
    "Obstacle",
    "ObstacleCollection",
    # LOS Engine
    "LOSEngine",
    "LOSResult",
    "SourceGeometry",
    # DEM
    "DEMData",
    "ContourLine",
]
