"""
Rekarisk Meteorology — Indonesian Location Weather Presets.

Typical meteorological conditions for major Indonesian cities and
oil & gas facility locations. Data represents annual averages suitable
for QRA screening studies. For detailed QRA, site-specific met data
should be used.

Sources: BMKG climate normals, typical refinery/platform conditions,
         engineering judgment for remote locations.

Usage:
    from rekarisk.meteorology.indonesia_locations import INDONESIA_LOCATIONS, get_location

    loc = get_location("Cepu")
    print(loc["wind_speed_ms"])  # 2.3
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Location data structure
# ---------------------------------------------------------------------------

# Each location entry:
#   name: Display name
#   province: Province / region
#   lat: Approximate latitude (°S, positive)
#   lon: Approximate longitude (°E, positive)
#   elevation_m: Elevation above sea level [m]
#   category: "major_city" | "oil_gas" | "industrial" | "coastal"
#   wind_speed_ms: Annual average wind speed at 10m [m/s]
#   wind_direction_deg: Prevailing wind direction [degrees from N]
#   temperature_k: Annual average temperature [K]
#   humidity_pct: Annual average relative humidity [%]
#   pressure_pa: Typical atmospheric pressure [Pa]
#   cloud_cover_oktas: Typical cloud cover [oktas 0-8]
#   stability_class: Predominant Pasquill-Gifford stability class
#   surface_roughness_m: Typical surface roughness z0 [m]
#   mixing_height_m: Typical mixing height [m]
#   notes: Additional context

INDONESIA_LOCATIONS: List[Dict] = [
    # ═══════════════════════════════════════════════════════════════════
    # MAJOR CITIES
    # ═══════════════════════════════════════════════════════════════════
    {
        "name": "Jakarta",
        "province": "DKI Jakarta",
        "lat": -6.21, "lon": 106.85, "elevation_m": 8,
        "category": "major_city",
        "wind_speed_ms": 2.5, "wind_direction_deg": 270,
        "temperature_k": 301.0, "humidity_pct": 80,
        "pressure_pa": 101225, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 1.0,  # urban
        "mixing_height_m": 1200,
        "notes": "Tropical urban, high humidity, land-sea breeze regime",
    },
    {
        "name": "Surabaya",
        "province": "Jawa Timur",
        "lat": -7.26, "lon": 112.75, "elevation_m": 5,
        "category": "major_city",
        "wind_speed_ms": 3.0, "wind_direction_deg": 225,
        "temperature_k": 300.0, "humidity_pct": 75,
        "pressure_pa": 101290, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 0.8,
        "mixing_height_m": 1100,
        "notes": "East Java industrial corridor, port city",
    },
    {
        "name": "Medan",
        "province": "Sumatera Utara",
        "lat": 3.59, "lon": 98.67, "elevation_m": 23,
        "category": "major_city",
        "wind_speed_ms": 2.0, "wind_direction_deg": 180,
        "temperature_k": 301.0, "humidity_pct": 82,
        "pressure_pa": 101050, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.8,
        "mixing_height_m": 1000,
        "notes": "Near Arun LNG, Belawan port, heavy rainfall",
    },
    {
        "name": "Bandung",
        "province": "Jawa Barat",
        "lat": -6.91, "lon": 107.61, "elevation_m": 768,
        "category": "major_city",
        "wind_speed_ms": 2.0, "wind_direction_deg": 270,
        "temperature_k": 295.0, "humidity_pct": 75,
        "pressure_pa": 92400, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.8,
        "mixing_height_m": 900,
        "notes": "Highland basin, cooler temperatures, lower pressure",
    },
    {
        "name": "Semarang",
        "province": "Jawa Tengah",
        "lat": -6.97, "lon": 110.42, "elevation_m": 5,
        "category": "major_city",
        "wind_speed_ms": 2.8, "wind_direction_deg": 225,
        "temperature_k": 300.0, "humidity_pct": 78,
        "pressure_pa": 101290, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 0.5,
        "mixing_height_m": 1100,
        "notes": "Port city, Java north coast industrial zone",
    },
    {
        "name": "Makassar",
        "province": "Sulawesi Selatan",
        "lat": -5.15, "lon": 119.43, "elevation_m": 5,
        "category": "major_city",
        "wind_speed_ms": 3.2, "wind_direction_deg": 270,
        "temperature_k": 301.0, "humidity_pct": 78,
        "pressure_pa": 101290, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 0.5,
        "mixing_height_m": 1200,
        "notes": "Eastern Indonesia gateway, monsoon influence",
    },
    {
        "name": "Palembang",
        "province": "Sumatera Selatan",
        "lat": -2.98, "lon": 104.75, "elevation_m": 8,
        "category": "major_city",
        "wind_speed_ms": 1.8, "wind_direction_deg": 180,
        "temperature_k": 301.0, "humidity_pct": 83,
        "pressure_pa": 101250, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.5,
        "mixing_height_m": 900,
        "notes": "Near PERTAMINA refinery, Musi river industrial",
    },
    {
        "name": "Balikpapan",
        "province": "Kalimantan Timur",
        "lat": -1.24, "lon": 116.85, "elevation_m": 10,
        "category": "major_city",
        "wind_speed_ms": 2.5, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 82,
        "pressure_pa": 101250, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.5,
        "mixing_height_m": 1000,
        "notes": "PERTAMINA refinery, oil & gas hub",
    },
    {
        "name": "Pekanbaru",
        "province": "Riau",
        "lat": 0.51, "lon": 101.45, "elevation_m": 10,
        "category": "major_city",
        "wind_speed_ms": 1.8, "wind_direction_deg": 180,
        "temperature_k": 301.0, "humidity_pct": 84,
        "pressure_pa": 101250, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 900,
        "notes": "Central Sumatra oil operations, Dumai pipeline",
    },
    {
        "name": "Pontianak",
        "province": "Kalimantan Barat",
        "lat": -0.02, "lon": 109.34, "elevation_m": 1,
        "category": "major_city",
        "wind_speed_ms": 1.5, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 85,
        "pressure_pa": 101300, "cloud_cover_oktas": 7,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 800,
        "notes": "Equatorial, very calm winds, high humidity",
    },

    # ═══════════════════════════════════════════════════════════════════
    # OIL & GAS FACILITY LOCATIONS
    # ═══════════════════════════════════════════════════════════════════
    {
        "name": "Cepu",
        "province": "Jawa Tengah",
        "lat": -7.15, "lon": 111.58, "elevation_m": 50,
        "category": "oil_gas",
        "wind_speed_ms": 2.0, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 78,
        "pressure_pa": 101100, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 1000,
        "notes": "Banyu Urip field, ExxonMobil Cepu Ltd, major onshore oil",
    },
    {
        "name": "Lhokseumawe",
        "province": "Aceh",
        "lat": 5.18, "lon": 97.14, "elevation_m": 5,
        "category": "oil_gas",
        "wind_speed_ms": 2.5, "wind_direction_deg": 225,
        "temperature_k": 300.0, "humidity_pct": 82,
        "pressure_pa": 101290, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 1000,
        "notes": "Arun LNG, PERTAMINA gas processing, Aceh industrial zone",
    },
    {
        "name": "Dumai",
        "province": "Riau",
        "lat": 1.67, "lon": 101.45, "elevation_m": 5,
        "category": "oil_gas",
        "wind_speed_ms": 2.2, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 82,
        "pressure_pa": 101290, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 1000,
        "notes": "PERTAMINA Dumai refinery (170 MBPD), port, oil terminal",
    },
    {
        "name": "Plaju",
        "province": "Sumatera Selatan",
        "lat": -2.94, "lon": 104.74, "elevation_m": 5,
        "category": "oil_gas",
        "wind_speed_ms": 1.8, "wind_direction_deg": 180,
        "temperature_k": 301.0, "humidity_pct": 83,
        "pressure_pa": 101290, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 900,
        "notes": "PERTAMINA Plaju refinery, Palembang area, Musi river",
    },
    {
        "name": "Cilacap",
        "province": "Jawa Tengah",
        "lat": -7.73, "lon": 109.01, "elevation_m": 5,
        "category": "oil_gas",
        "wind_speed_ms": 3.0, "wind_direction_deg": 225,
        "temperature_k": 300.0, "humidity_pct": 78,
        "pressure_pa": 101290, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 1100,
        "notes": "PERTAMINA Cilacap refinery (348 MBPD), south coast port",
    },
    {
        "name": "Balongan",
        "province": "Jawa Barat",
        "lat": -6.55, "lon": 108.14, "elevation_m": 5,
        "category": "oil_gas",
        "wind_speed_ms": 2.8, "wind_direction_deg": 270,
        "temperature_k": 301.0, "humidity_pct": 80,
        "pressure_pa": 101290, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 1100,
        "notes": "PERTAMINA Balongan refinery (125 MBPD), north coast Java",
    },
    {
        "name": "Tuban",
        "province": "Jawa Timur",
        "lat": -6.90, "lon": 112.05, "elevation_m": 5,
        "category": "oil_gas",
        "wind_speed_ms": 2.8, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 78,
        "pressure_pa": 101290, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 1100,
        "notes": "Tuban refinery (Pertamina Rosneft JV), East Java coast",
    },
    {
        "name": "Sungai Pakning",
        "province": "Riau",
        "lat": 1.30, "lon": 102.15, "elevation_m": 5,
        "category": "oil_gas",
        "wind_speed_ms": 2.0, "wind_direction_deg": 180,
        "temperature_k": 301.0, "humidity_pct": 84,
        "pressure_pa": 101250, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 900,
        "notes": "PERTAMINA Sungai Pakning refinery, coastal Sumatra",
    },
    {
        "name": "Bontang",
        "province": "Kalimantan Timur",
        "lat": 0.13, "lon": 117.48, "elevation_m": 5,
        "category": "oil_gas",
        "wind_speed_ms": 2.5, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 82,
        "pressure_pa": 101250, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 1000,
        "notes": "PT Badak NGL (LNG plant 22 MTPA), major gas processing",
    },
    {
        "name": "Sangatta",
        "province": "Kalimantan Timur",
        "lat": -0.50, "lon": 117.56, "elevation_m": 10,
        "category": "oil_gas",
        "wind_speed_ms": 2.2, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 83,
        "pressure_pa": 101250, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 1000,
        "notes": "Kaltim Prima Coal, oil/gas support facilities, East Kutai",
    },
    {
        "name": "Pangkalan Brandan",
        "province": "Sumatera Utara",
        "lat": 4.02, "lon": 98.21, "elevation_m": 5,
        "category": "oil_gas",
        "wind_speed_ms": 2.3, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 82,
        "pressure_pa": 101290, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 1000,
        "notes": "Historic PERTAMINA oil field, North Sumatra coast",
    },
    {
        "name": "Rantau",
        "province": "Aceh",
        "lat": 4.70, "lon": 97.65, "elevation_m": 15,
        "category": "oil_gas",
        "wind_speed_ms": 2.0, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 82,
        "pressure_pa": 101150, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 900,
        "notes": "PERTAMINA EP Rantau field, onshore North Aceh",
    },
    {
        "name": "Prabumulih",
        "province": "Sumatera Selatan",
        "lat": -3.43, "lon": 104.23, "elevation_m": 50,
        "category": "oil_gas",
        "wind_speed_ms": 1.8, "wind_direction_deg": 180,
        "temperature_k": 301.0, "humidity_pct": 83,
        "pressure_pa": 101100, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 900,
        "notes": "South Sumatra oil fields, PERTAMINA EP operations",
    },

    # ═══════════════════════════════════════════════════════════════════
    # INDUSTRIAL / PETROCHEMICAL
    # ═══════════════════════════════════════════════════════════════════
    {
        "name": "Cilegon",
        "province": "Banten",
        "lat": -6.02, "lon": 106.00, "elevation_m": 5,
        "category": "industrial",
        "wind_speed_ms": 3.0, "wind_direction_deg": 270,
        "temperature_k": 301.0, "humidity_pct": 78,
        "pressure_pa": 101290, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 0.5,
        "mixing_height_m": 1200,
        "notes": "Petrochemical (Chandra Asri, Krakatau Steel), Sunda Strait",
    },
    {
        "name": "Gresik",
        "province": "Jawa Timur",
        "lat": -7.16, "lon": 112.65, "elevation_m": 5,
        "category": "industrial",
        "wind_speed_ms": 2.8, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 76,
        "pressure_pa": 101290, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 0.5,
        "mixing_height_m": 1100,
        "notes": "Petrochemical, smelter, Surabaya industrial satellite",
    },
    {
        "name": "Cikarang",
        "province": "Jawa Barat",
        "lat": -6.31, "lon": 107.16, "elevation_m": 20,
        "category": "industrial",
        "wind_speed_ms": 2.2, "wind_direction_deg": 270,
        "temperature_k": 301.0, "humidity_pct": 78,
        "pressure_pa": 101200, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 0.8,
        "mixing_height_m": 1000,
        "notes": "MM2100, Jababeka, largest industrial estate in SEA",
    },
    {
        "name": "Tanjung Priok",
        "province": "DKI Jakarta",
        "lat": -6.10, "lon": 106.88, "elevation_m": 2,
        "category": "industrial",
        "wind_speed_ms": 2.5, "wind_direction_deg": 270,
        "temperature_k": 301.0, "humidity_pct": 80,
        "pressure_pa": 101290, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 1.0,
        "mixing_height_m": 1100,
        "notes": "Port, fuel terminals, tank farms, Jakarta bay",
    },

    # ═══════════════════════════════════════════════════════════════════
    # COASTAL / OFFSHORE SUPPORT
    # ═══════════════════════════════════════════════════════════════════
    {
        "name": "Batam",
        "province": "Kepulauan Riau",
        "lat": 1.04, "lon": 104.17, "elevation_m": 5,
        "category": "coastal",
        "wind_speed_ms": 3.0, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 82,
        "pressure_pa": 101290, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 1100,
        "notes": "Oil & gas support, shipyard, Singapore Strait",
    },
    {
        "name": "Tanjung Balai Karimun",
        "province": "Kepulauan Riau",
        "lat": 1.00, "lon": 103.45, "elevation_m": 5,
        "category": "coastal",
        "wind_speed_ms": 3.0, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 82,
        "pressure_pa": 101290, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 0.2,
        "mixing_height_m": 1100,
        "notes": "Oil storage, ship-to-ship transfer, Malacca Strait",
    },
    {
        "name": "Natuna",
        "province": "Kepulauan Riau",
        "lat": 3.80, "lon": 108.30, "elevation_m": 5,
        "category": "oil_gas",
        "wind_speed_ms": 3.5, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 80,
        "pressure_pa": 101290, "cloud_cover_oktas": 5,
        "stability_class": "D",
        "surface_roughness_m": 0.2,
        "mixing_height_m": 1200,
        "notes": "Natuna gas field (Largest in Asia), offshore platform support",
    },
    {
        "name": "Sorong",
        "province": "Papua Barat",
        "lat": -0.88, "lon": 131.25, "elevation_m": 5,
        "category": "coastal",
        "wind_speed_ms": 2.8, "wind_direction_deg": 180,
        "temperature_k": 301.0, "humidity_pct": 82,
        "pressure_pa": 101290, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.2,
        "mixing_height_m": 1100,
        "notes": "Eastern Indonesia, Tangguh LNG support, port",
    },
    {
        "name": "Bintuni",
        "province": "Papua Barat",
        "lat": -2.08, "lon": 133.57, "elevation_m": 5,
        "category": "oil_gas",
        "wind_speed_ms": 2.2, "wind_direction_deg": 180,
        "temperature_k": 302.0, "humidity_pct": 85,
        "pressure_pa": 101290, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.2,
        "mixing_height_m": 1000,
        "notes": "BP Tangguh LNG plant, remote mangrove coast",
    },
    {
        "name": "Senipah",
        "province": "Kalimantan Timur",
        "lat": -1.25, "lon": 116.90, "elevation_m": 5,
        "category": "oil_gas",
        "wind_speed_ms": 2.3, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 82,
        "pressure_pa": 101250, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.2,
        "mixing_height_m": 1000,
        "notes": "TOTAL E&P Indonesia, Senipah gas plant, Mahakam delta",
    },
    {
        "name": "Tanjung",
        "province": "Kalimantan Selatan",
        "lat": -3.53, "lon": 116.17, "elevation_m": 5,
        "category": "oil_gas",
        "wind_speed_ms": 2.0, "wind_direction_deg": 225,
        "temperature_k": 301.0, "humidity_pct": 82,
        "pressure_pa": 101250, "cloud_cover_oktas": 6,
        "stability_class": "D",
        "surface_roughness_m": 0.3,
        "mixing_height_m": 900,
        "notes": "Asam-Asam power plant, coal & gas, South Kalimantan coast",
    },

    # ═══════════════════════════════════════════════════════════════════
    # WORST-CASE SCREENING SCENARIOS
    # ═══════════════════════════════════════════════════════════════════
    {
        "name": "Worst Case – Stable Night (Indonesia)",
        "province": "Screening",
        "lat": 0.0, "lon": 0.0, "elevation_m": 0,
        "category": "screening",
        "wind_speed_ms": 1.5, "wind_direction_deg": 0,
        "temperature_k": 301.0, "humidity_pct": 90,
        "pressure_pa": 101325, "cloud_cover_oktas": 8,
        "stability_class": "F",
        "surface_roughness_m": 0.1,
        "mixing_height_m": 200,
        "notes": "Conservative worst-case for QRA screening (F stability, low wind)",
    },
    {
        "name": "Worst Case – Calm Humid (Indonesia)",
        "province": "Screening",
        "lat": 0.0, "lon": 0.0, "elevation_m": 0,
        "category": "screening",
        "wind_speed_ms": 1.0, "wind_direction_deg": 0,
        "temperature_k": 305.0, "humidity_pct": 95,
        "pressure_pa": 101325, "cloud_cover_oktas": 8,
        "stability_class": "F",
        "surface_roughness_m": 0.1,
        "mixing_height_m": 150,
        "notes": "Extreme worst-case, tropical calm conditions",
    },
]


def get_location(name: str) -> Optional[Dict]:
    """Look up a location by name (case-insensitive, partial match)."""
    name_lower = name.lower().strip()
    for loc in INDONESIA_LOCATIONS:
        if loc["name"].lower() == name_lower:
            return loc
    # Partial match
    for loc in INDONESIA_LOCATIONS:
        if name_lower in loc["name"].lower():
            return loc
    return None


def get_locations_by_category(category: str) -> List[Dict]:
    """Get all locations of a given category."""
    return [loc for loc in INDONESIA_LOCATIONS if loc["category"] == category]


def get_all_location_names() -> List[str]:
    """Get sorted list of all location names."""
    return sorted(loc["name"] for loc in INDONESIA_LOCATIONS)


def get_location_categories() -> List[str]:
    """Get sorted list of location categories."""
    return sorted(set(loc["category"] for loc in INDONESIA_LOCATIONS))


def location_to_meteorological_state(loc: Dict) -> Dict:
    """Convert a location dict to MeteorologicalState constructor kwargs."""
    return {
        "wind_speed_ms": loc["wind_speed_ms"],
        "wind_direction_deg": loc["wind_direction_deg"],
        "ambient_temperature_k": loc["temperature_k"],
        "ambient_pressure_pa": loc["pressure_pa"],
        "relative_humidity_pct": loc["humidity_pct"],
        "cloud_cover_oktas": loc["cloud_cover_oktas"],
        "surface_roughness_m": loc["surface_roughness_m"],
        "stability_class": loc["stability_class"],
        "mixing_height_m": loc["mixing_height_m"],
        "is_daytime": True,
    }
