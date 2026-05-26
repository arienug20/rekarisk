"""
Rekarisk GIS Export — GeoJSON, KML overlay export.

Exports consequence analysis contour data in GIS-friendly formats:
    - GeoJSON: Contour lines as LineString, areas as Polygon
    - KML: Contour overlays for Google Earth (color-coded, with placemarks)

No heavy GIS dependencies (no fiona, no rasterio).
Optional pyproj for CRS conversion; falls back to local coordinates.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


# ══════════════════════════════════════════════════════════════════════════════
# Colour Mapping
# ══════════════════════════════════════════════════════════════════════════════

# Severity colour palette (green → yellow → orange → red → purple)
_SEVERITY_COLORS = [
    "#00FF00",  # Green  — very low
    "#7FFF00",  # Chartreuse
    "#FFFF00",  # Yellow
    "#FFBF00",  # Amber
    "#FF7F00",  # Orange
    "#FF3F00",  # Red-orange
    "#FF0000",  # Red
    "#BF00FF",  # Purple — extreme
]


def _severity_color(index: int, total: int) -> str:
    """Pick a colour from the severity ramp."""
    if total <= 1:
        return _SEVERITY_COLORS[3]
    idx = int((index / (total - 1)) * (len(_SEVERITY_COLORS) - 1))
    return _SEVERITY_COLORS[min(idx, len(_SEVERITY_COLORS) - 1)]


# ══════════════════════════════════════════════════════════════════════════════
# Contour to GeoJSON
# ══════════════════════════════════════════════════════════════════════════════

def contours_to_geojson(
    contour_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    crs: str = "local",
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Convert contour data to GeoJSON FeatureCollection.

    Parameters
    ----------
    contour_data : dict or list[dict]
        Contour data. Can be:
        - dict with keys: "x" (1D array), "y" (1D array), "Z" (2D array), "levels" (list)
        - list of dicts, each with "contour" (Nx2 array), "level" (float), "scenario" (optional)
    crs : str
        Coordinate reference system identifier (e.g., "EPSG:4326" for WGS84).
        If pyproj is available, attempts conversion from local (meters) to target CRS.
    output_path : str or Path, optional
        If provided, writes the GeoJSON to this path.

    Returns
    -------
    str
        GeoJSON string (always returned, even if written to file).
    """
    features = []

    if isinstance(contour_data, dict) and "x" in contour_data and "y" in contour_data and "Z" in contour_data:
        # Full grid with levels — need to compute contours
        features = _grid_to_geojson_features(
            contour_data["x"], contour_data["y"], contour_data["Z"],
            contour_data.get("levels", []),
            contour_data.get("scenario", "Scenario"),
            contour_data.get("substance", ""),
            crs,
        )
    elif isinstance(contour_data, list):
        for i, item in enumerate(contour_data):
            if isinstance(item, dict):
                features += _contour_item_to_features(
                    item, i, crs
                )

    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:{crs.replace(':', '::')}"},
        } if crs != "local" else None,
        "features": features,
        "metadata": {
            "generator": "Rekarisk",
            "generated_at": datetime.now().isoformat(),
        },
    }

    # Remove null crs
    if geojson["crs"] is None:
        del geojson["crs"]

    geo_str = json.dumps(geojson, indent=2, ensure_ascii=False, default=_json_default)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(geo_str, encoding="utf-8")

    return geo_str


def _grid_to_geojson_features(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    levels: list,
    scenario: str,
    substance: str,
    crs: str,
) -> List[Dict[str, Any]]:
    """Extract contour lines from a grid and convert to GeoJSON features."""
    import matplotlib.pyplot as plt

    features = []
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    Z_arr = np.asarray(Z, dtype=float)

    for i, level in enumerate(levels):
        try:
            contour_set = plt.contour(x_arr, y_arr, Z_arr, levels=[level])
            plt.close(contour_set.collections[0].figure if hasattr(contour_set, 'collections') and contour_set.collections else None)
            segments = contour_set.allsegs[0] if contour_set.allsegs else []
        except Exception:
            # Fallback: try manual extraction
            import matplotlib._contour as _contour
            try:
                quad = _contour.QuadContourGenerator(x_arr, y_arr, Z_arr, None, True, 0)
                segments = quad.create_contour(level)
            except Exception:
                segments = []

        for seg_i, seg in enumerate(segments):
            if len(seg) < 2:
                continue
            coords = _transform_coords(seg.tolist() if hasattr(seg, 'tolist') else seg, crs)
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "level": float(level),
                    "scenario": scenario,
                    "substance": substance,
                    "segment": seg_i,
                    "severity_color": _severity_color(i, len(levels)),
                },
            })

    # Cleanup
    plt.close('all')
    return features


def _contour_item_to_features(
    item: Dict[str, Any],
    index: int,
    crs: str,
) -> List[Dict[str, Any]]:
    """Convert a contour item dict to GeoJSON features."""
    features = []

    contour = item.get("contour")
    level = item.get("level", index)
    scenario = item.get("scenario", f"Scenario {index + 1}")
    substance = item.get("substance", "")

    if contour is None:
        return features

    if isinstance(contour, list):
        # Multiple segments
        for seg_i, seg in enumerate(contour):
            if isinstance(seg, (list, np.ndarray)) and len(seg) >= 2:
                coords = _to_coord_list(seg)
                coords = _transform_coords(coords, crs)
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "level": level,
                        "scenario": scenario,
                        "substance": substance,
                        "segment": seg_i,
                    },
                })

    return features


def _to_coord_list(seg: Any) -> List[List[float]]:
    """Convert segment to list of [x, y] coordinates."""
    if hasattr(seg, 'tolist'):
        seg = seg.tolist()
    result = []
    for pt in seg:
        if isinstance(pt, (list, tuple)):
            result.append([float(pt[0]), float(pt[1])])
        elif hasattr(pt, 'x') and hasattr(pt, 'y'):
            result.append([float(pt.x), float(pt.y)])
    return result


def _transform_coords(
    coords: List[List[float]],
    crs: str,
) -> List[List[float]]:
    """Transform coordinates from local meters to target CRS if pyproj available."""
    if crs == "local" or not HAS_PYPROJ:
        return coords

    # Assume source is a local projected CRS centered at 0,0
    # For real use, the caller should provide proper source CRS
    try:
        transformer = Transformer.from_crs("EPSG:3857", crs, always_xy=True)
        return [[float(x), float(y)] for x, y in
                transformer.transform([c[0] for c in coords],
                                       [c[1] for c in coords])]
    except Exception:
        return coords


# ══════════════════════════════════════════════════════════════════════════════
# Contour to KML
# ══════════════════════════════════════════════════════════════════════════════

def contours_to_kml(
    contour_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    output_path: Optional[Union[str, Path]] = None,
    name: str = "Rekarisk Contour Overlay",
) -> str:
    """Convert contour data to KML for Google Earth visualization.

    Parameters
    ----------
    contour_data : dict or list[dict]
        Same format as contours_to_geojson.
    output_path : str or Path, optional
        If provided, writes the KML to this path.
    name : str
        Document name for the KML.

    Returns
    -------
    str
        KML string.
    """
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")

    ET.SubElement(doc, "name").text = name
    ET.SubElement(doc, "description").text = "Generated by Rekarisk"

    # Styles
    styles = {}
    for i in range(len(_SEVERITY_COLORS)):
        style_id = f"severityStyle{i}"
        style = ET.SubElement(doc, "Style", id=style_id)
        line = ET.SubElement(style, "LineStyle")
        ET.SubElement(line, "color").text = _rgb_to_abgr(_SEVERITY_COLORS[i])
        ET.SubElement(line, "width").text = "2"
        poly = ET.SubElement(style, "PolyStyle")
        ET.SubElement(poly, "color").text = _rgb_to_abgr(_SEVERITY_COLORS[i], alpha_hex="80")
        styles[style_id] = _SEVERITY_COLORS[i]

    # Placemark for source
    source_pm = ET.SubElement(doc, "Placemark")
    ET.SubElement(source_pm, "name").text = "Release Source (0, 0)"
    source_pt = ET.SubElement(source_pm, "Point")
    ET.SubElement(source_pt, "coordinates").text = "0,0,0"

    # Contour features
    if isinstance(contour_data, dict) and "x" in contour_data:
        x = np.asarray(contour_data["x"], dtype=float)
        y = np.asarray(contour_data["y"], dtype=float)
        Z = np.asarray(contour_data["Z"], dtype=float)
        levels = contour_data.get("levels", [])
        for i, level in enumerate(levels):
            _add_kml_contour_level(doc, x, y, Z, level, i, len(levels),
                                   contour_data.get("scenario", "Scenario"),
                                   styles)
    elif isinstance(contour_data, list):
        for i, item in enumerate(contour_data):
            if isinstance(item, dict):
                contour = item.get("contour")
                level = item.get("level", i)
                scenario = item.get("scenario", f"Scenario {i + 1}")
                if contour is not None:
                    _add_kml_contour_segments(doc, contour, level, scenario,
                                              i, len(contour_data), styles)

    # Placemark for receptors (if provided)
    if isinstance(contour_data, dict) and "receptors" in contour_data:
        receptors = contour_data["receptors"]
        for i, rec in enumerate(receptors):
            pm = ET.SubElement(doc, "Placemark")
            ET.SubElement(pm, "name").text = f"Receptor {i + 1}"
            pt = ET.SubElement(pm, "Point")
            x_val = rec[0] if isinstance(rec, (list, tuple)) else rec.get("x", 0)
            y_val = rec[1] if isinstance(rec, (list, tuple)) else rec.get("y", 0)
            ET.SubElement(pt, "coordinates").text = f"{x_val},{y_val},0"

    xml_str = _pretty_xml(kml)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(xml_str, encoding="utf-8")

    return xml_str


def _add_kml_contour_level(
    doc: ET.Element,
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    level: float,
    idx: int,
    total: int,
    scenario: str,
    styles: Dict[str, str],
) -> None:
    """Extract a single contour level and add as KML LineString."""
    import matplotlib.pyplot as plt

    try:
        cs = plt.contour(x, y, Z, levels=[level])
        segments = cs.allsegs[0] if cs.allsegs else []
        plt.close('all')
    except Exception:
        segments = []

    style_id = f"severityStyle{min(idx, len(_SEVERITY_COLORS) - 1)}"

    for seg_i, seg in enumerate(segments):
        if len(seg) < 2:
            continue
        pm = ET.SubElement(doc, "Placemark")
        ET.SubElement(pm, "name").text = f"{scenario} — Level {level:.3g}"
        ET.SubElement(pm, "styleUrl").text = f"#{style_id}"
        ls = ET.SubElement(pm, "LineString")
        ET.SubElement(ls, "altitudeMode").text = "clampToGround"
        coords = " ".join(f"{pt[0]},{pt[1]},0" for pt in seg)
        ET.SubElement(ls, "coordinates").text = coords


def _add_kml_contour_segments(
    doc: ET.Element,
    contour: Any,
    level: float,
    scenario: str,
    idx: int,
    total: int,
    styles: Dict[str, str],
) -> None:
    """Add pre-computed contour segments as KML Placemarks."""
    style_id = f"severityStyle{min(idx, len(_SEVERITY_COLORS) - 1)}"

    if isinstance(contour, list):
        segments = contour
    else:
        segments = [contour]

    for seg_i, seg in enumerate(segments):
        if not hasattr(seg, '__len__') or len(seg) < 2:
            continue
        if isinstance(seg, np.ndarray):
            seg = seg.tolist()

        pm = ET.SubElement(doc, "Placemark")
        ET.SubElement(pm, "name").text = f"{scenario} — Level {level:.3g}"
        ET.SubElement(pm, "styleUrl").text = f"#{style_id}"

        ls = ET.SubElement(pm, "LineString")
        ET.SubElement(ls, "tessellate").text = "1"
        ET.SubElement(ls, "altitudeMode").text = "clampToGround"

        try:
            coords_str = " ".join(f"{pt[0]},{pt[1]},0" for pt in seg)
        except (TypeError, IndexError):
            coords_str = str(seg)
        ET.SubElement(ls, "coordinates").text = coords_str


# ══════════════════════════════════════════════════════════════════════════════
# IR Contours to GeoJSON
# ══════════════════════════════════════════════════════════════════════════════

def ir_contours_to_geojson(
    ir_grid: Dict[str, Any],
    thresholds: Optional[List[float]] = None,
    crs: str = "local",
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Convert Individual Risk grid to GeoJSON contours.

    Parameters
    ----------
    ir_grid : dict
        Must contain "x" (1D), "y" (1D), "values" (2D risk values).
    thresholds : list[float], optional
        Risk thresholds for contouring. Defaults to standard IR thresholds:
        [1e-4, 1e-5, 3e-6, 1e-6, 1e-7] per year.
    crs : str
        Coordinate reference system.
    output_path : str or Path, optional
        If provided, writes to file.

    Returns
    -------
    str
        GeoJSON string.
    """
    if thresholds is None:
        thresholds = [1e-4, 1e-5, 3e-6, 1e-6, 1e-7]

    contour_data = {
        "x": np.asarray(ir_grid.get("x", [])),
        "y": np.asarray(ir_grid.get("y", [])),
        "Z": np.asarray(ir_grid.get("values", ir_grid.get("Z", []))),
        "levels": thresholds,
        "scenario": ir_grid.get("name", "IR Contour"),
        "substance": ir_grid.get("substance", ""),
    }

    return contours_to_geojson(contour_data, crs=crs, output_path=output_path)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _rgb_to_abgr(hex_color: str, alpha_hex: str = "FF") -> str:
    """Convert hex RGB (#RRGGBB) to KML ABGR (AABBGGRR)."""
    r = hex_color[1:3]
    g = hex_color[3:5]
    b = hex_color[5:7]
    return f"{alpha_hex}{b}{g}{r}"


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return str(obj)


def _pretty_xml(elem: ET.Element) -> str:
    """Pretty-print XML element tree."""
    ET.indent(elem, space="  ")
    return ET.tostring(elem, encoding="unicode", method="xml")
