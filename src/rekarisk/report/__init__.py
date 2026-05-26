"""
Rekarisk Report Module — Multi-format export and reporting.

Provides:
    - PDF report generation (ReportLab platypus)
    - Excel multi-sheet export (openpyxl)
    - CSV / JSON / TXT text export
    - GeoJSON / KML GIS overlay export
    - Image export for plots (PNG, SVG, PDF)
"""

from .pdf_generator import (
    generate_report,
    add_cover_page,
    add_results_section,
    add_table,
    add_plot,
)
from .excel_export import (
    export_to_excel,
)
from .text_export import (
    export_csv,
    export_json,
    export_summary_text,
)
from .gis_export import (
    contours_to_geojson,
    contours_to_kml,
    ir_contours_to_geojson,
)
from .image_export import (
    export_contour_plot,
    export_fn_curve,
    export_risk_matrix_image,
    export_all_plots,
)

__all__ = [
    # PDF
    "generate_report",
    "add_cover_page",
    "add_results_section",
    "add_table",
    "add_plot",
    # Excel
    "export_to_excel",
    # Text
    "export_csv",
    "export_json",
    "export_summary_text",
    # GIS
    "contours_to_geojson",
    "contours_to_kml",
    "ir_contours_to_geojson",
    # Image
    "export_contour_plot",
    "export_fn_curve",
    "export_risk_matrix_image",
    "export_all_plots",
]
