"""
Rekarisk PDF Report Generator.

Generates professional consequence analysis reports using ReportLab's
platypus (flow-based layout). Supports cover page, table of contents,
executive summary, input data, results (per scenario), QRA section,
conclusion, and appendix.

Report structure:
    1. Cover page: project name, date, version
    2. Table of contents
    3. Executive summary
    4. Input data section
    5. Results section (per scenario with tables, plots, threshold distances)
    6. QRA section (IR contours, FN curve, risk matrix)
    7. Conclusion
    8. Appendix
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm, inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, KeepTogether, ListFlowable, ListItem,
        NextPageTemplate, PageTemplate, Frame, BaseDocTemplate,
        Flowable,
    )
    from reportlab.platypus.tableofcontents import TableOfContents
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


# ══════════════════════════════════════════════════════════════════════════════
# Styles
# ══════════════════════════════════════════════════════════════════════════════

_COLOR_PRIMARY = colors.HexColor("#1a5276")
_COLOR_ACCENT = colors.HexColor("#2e86c1")
_COLOR_LIGHT_BG = colors.HexColor("#ebf5fb")
_COLOR_DARK_TEXT = colors.HexColor("#1b2631")
_COLOR_BORDER = colors.HexColor("#aeb6bf")
_COLOR_HEADER_BG = colors.HexColor("#1a5276")
_COLOR_HEADER_FG = colors.white
_COLOR_ROW_ALT = colors.HexColor("#f2f4f4")


def _build_styles() -> dict:
    """Build custom paragraph styles on top of the default set."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "CoverTitle", parent=styles["Title"],
        fontSize=28, textColor=_COLOR_PRIMARY,
        leading=34, spaceAfter=12, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "CoverSubtitle", parent=styles["Normal"],
        fontSize=14, textColor=_COLOR_ACCENT,
        leading=18, spaceAfter=6, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "CoverInfo", parent=styles["Normal"],
        fontSize=11, textColor=colors.grey,
        leading=15, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "SectionH1", parent=styles["Heading1"],
        fontSize=18, textColor=_COLOR_PRIMARY,
        spaceBefore=24, spaceAfter=12, leading=22,
    ))
    styles.add(ParagraphStyle(
        "SectionH2", parent=styles["Heading2"],
        fontSize=14, textColor=_COLOR_ACCENT,
        spaceBefore=18, spaceAfter=8, leading=18,
    ))
    styles.add(ParagraphStyle(
        "SectionH3", parent=styles["Heading3"],
        fontSize=12, textColor=_COLOR_PRIMARY,
        spaceBefore=14, spaceAfter=6, leading=15,
    ))
    styles.add(ParagraphStyle(
        "BodyText2", parent=styles["BodyText"],
        fontSize=10, leading=14, alignment=TA_JUSTIFY,
        textColor=_COLOR_DARK_TEXT,
    ))
    styles.add(ParagraphStyle(
        "TableCell", parent=styles["Normal"],
        fontSize=9, leading=12, spaceBefore=1, spaceAfter=1,
    ))
    styles.add(ParagraphStyle(
        "TableHeader", parent=styles["Normal"],
        fontSize=9, leading=12, textColor=colors.white,
        spaceBefore=1, spaceAfter=1,
    ))
    styles.add(ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=9, textColor=colors.grey, alignment=TA_CENTER,
        leading=12, spaceBefore=4, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        "Footer", parent=styles["Normal"],
        fontSize=8, textColor=colors.grey, alignment=TA_CENTER,
    ))

    return styles


# ══════════════════════════════════════════════════════════════════════════════
# Page templates
# ══════════════════════════════════════════════════════════════════════════════

_PAGE_W, _PAGE_H = A4
_MARGIN = 2.0 * cm


def _header_footer(canvas_obj, doc):
    """Draw header and footer on every page."""
    canvas_obj.saveState()
    # Header line
    canvas_obj.setStrokeColor(_COLOR_ACCENT)
    canvas_obj.setLineWidth(0.5)
    canvas_obj.line(_MARGIN, _PAGE_H - _MARGIN + 0.3 * cm,
                    _PAGE_W - _MARGIN, _PAGE_H - _MARGIN + 0.3 * cm)
    # Footer
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(colors.grey)
    canvas_obj.drawCentredString(
        _PAGE_W / 2, _MARGIN / 2,
        f"Rekarisk Report — Page {canvas_obj.getPageNumber()}"
    )
    # Header text
    canvas_obj.setFont("Helvetica-Oblique", 8)
    canvas_obj.drawString(
        _MARGIN, _PAGE_H - _MARGIN + 0.6 * cm,
        "Rekarisk Consequence Analysis Report"
    )
    canvas_obj.drawRightString(
        _PAGE_W - _MARGIN, _PAGE_H - _MARGIN + 0.6 * cm,
        datetime.now().strftime("%d %B %Y")
    )
    canvas_obj.restoreState()


# ══════════════════════════════════════════════════════════════════════════════
# Helper flowables
# ══════════════════════════════════════════════════════════════════════════════

class HorizontalLine(Flowable):
    """A thin horizontal line spanning the page width."""
    def __init__(self, color=_COLOR_BORDER, thickness=0.5):
        Flowable.__init__(self)
        self._color = color
        self._thickness = thickness
        self.width = _PAGE_W - 2 * _MARGIN
        self.height = 12

    def draw(self):
        self.canv.setStrokeColor(self._color)
        self.canv.setLineWidth(self._thickness)
        self.canv.line(0, 6, self.width, 6)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(
    project_data: Dict[str, Any],
    results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    *,
    sections: Optional[Dict[str, bool]] = None,
    cover_info: Optional[Dict[str, str]] = None,
    executive_summary: str = "",
    conclusion: str = "",
) -> str:
    """Generate a full PDF report for a consequence analysis project.

    Parameters
    ----------
    project_data : dict
        Project metadata (name, created_at, scenarios, weather_cases, substances).
    results : list[dict]
        List of result dicts, one per scenario. Each must contain at least:
        - name (str)
        - type (str: "dispersion" | "fire" | "explosion" | "source_term" | "qra")
        Optionally: tables, plots, thresholds, fn_curve, ir_grid, risk_matrix.
    output_path : str or Path
        Where to write the PDF file.
    sections : dict, optional
        Boolean flags controlling which sections to include:
        {"cover": True, "toc": True, "summary": True, "input": True,
         "results": True, "qra": True, "conclusion": True, "appendix": True}
    cover_info : dict, optional
        {"project_name": str, "author": str, "date": str, "version": str,
         "organization": str}
    executive_summary : str
        Free-text executive summary.
    conclusion : str
        Free-text conclusion / recommendations.

    Returns
    -------
    str
        The output path.
    """
    if not HAS_REPORTLAB:
        raise ImportError(
            "reportlab is required for PDF generation. "
            "Install with: pip install reportlab"
        )

    if sections is None:
        sections = {
            "cover": True, "toc": True, "summary": True, "input": True,
            "results": True, "qra": True, "conclusion": True, "appendix": True,
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN + 0.4 * cm,
        bottomMargin=_MARGIN + 0.4 * cm,
        title=project_data.get("name", "Consequence Analysis Report"),
        author=cover_info.get("author", "") if cover_info else "",
    )

    styles = _build_styles()
    story: List[Flowable] = []

    # ── 1. Cover Page ──
    if sections.get("cover", True):
        add_cover_page(story, doc, project_data, cover_info, styles)
        story.append(PageBreak())

    # ── 2. Table of Contents ──
    if sections.get("toc", True):
        story.append(Paragraph("Table of Contents", styles["SectionH1"]))
        story.append(HorizontalLine())
        toc_items = _build_toc(project_data, sections)
        for item in toc_items:
            story.append(Paragraph(item, styles["BodyText2"]))
        story.append(Spacer(1, 6))
        story.append(PageBreak())

    # ── 3. Executive Summary ──
    if sections.get("summary", True):
        story.append(Paragraph("1. Executive Summary", styles["SectionH1"]))
        story.append(HorizontalLine())
        if executive_summary:
            story.append(Paragraph(executive_summary.replace("\n", "<br/>"),
                                   styles["BodyText2"]))
        else:
            story.append(Paragraph(
                "This report presents the results of a consequence analysis "
                f"performed using Rekarisk. The analysis covers "
                f"{len(results)} scenario(s) including "
                f"{', '.join(r.get('type', 'unknown') for r in results)} "
                f"modeling.",
                styles["BodyText2"]
            ))
        story.append(Spacer(1, 6))

    # ── 4. Input Data ──
    if sections.get("input", True):
        story.append(Paragraph("2. Input Data", styles["SectionH1"]))
        story.append(HorizontalLine())

        # Weather
        weather_cases = project_data.get("weather_cases", [])
        if weather_cases:
            story.append(Paragraph("2.1 Weather Conditions", styles["SectionH2"]))
            for w in weather_cases:
                story.append(Paragraph(
                    f"<b>Case: {w.get('name', 'N/A')}</b><br/>"
                    + _format_weather(w),
                    styles["BodyText2"]
                ))

        # Substances
        substances = project_data.get("substances", [])
        if substances:
            story.append(Paragraph("2.2 Substances", styles["SectionH2"]))
            for s in substances:
                story.append(Paragraph(
                    f"<b>{s.get('name', 'N/A')}</b> — "
                    f"{s.get('cas', '')}",
                    styles["BodyText2"]
                ))

        # Scenarios input summary
        story.append(Paragraph("2.3 Scenario Parameters", styles["SectionH2"]))
        for i, res in enumerate(results, 1):
            story.append(Paragraph(
                f"<b>Scenario {i}: {res.get('name', 'Unnamed')}</b> "
                f"({res.get('type', 'general')})",
                styles["BodyText2"]
            ))
            inputs = res.get("inputs", {})
            if inputs:
                for k, v in inputs.items():
                    story.append(Paragraph(
                        f"&nbsp;&nbsp;&nbsp;{k}: {v}",
                        styles["BodyText2"]
                    ))

        story.append(Spacer(1, 6))

    # ── 5. Results ──
    if sections.get("results", True):
        story.append(Paragraph("3. Results", styles["SectionH1"]))
        story.append(HorizontalLine())
        for i, res in enumerate(results, 1):
            add_results_section(story, doc, res, i, styles)
        story.append(Spacer(1, 6))

    # ── 6. QRA Section ──
    if sections.get("qra", True):
        qra_results = [r for r in results if r.get("type") == "qra"]
        if qra_results:
            story.append(Paragraph("4. Quantitative Risk Assessment", styles["SectionH1"]))
            story.append(HorizontalLine())
            for i, qra in enumerate(qra_results, 1):
                _add_qra_section(story, qra, i, styles)
                story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("4. Quantitative Risk Assessment", styles["SectionH1"]))
            story.append(HorizontalLine())
            story.append(Paragraph(
                "No QRA results were provided for this analysis.",
                styles["BodyText2"]
            ))

    # ── 7. Conclusion ──
    if sections.get("conclusion", True):
        story.append(Paragraph("5. Conclusion", styles["SectionH1"]))
        story.append(HorizontalLine())
        if conclusion:
            story.append(Paragraph(conclusion.replace("\n", "<br/>"),
                                   styles["BodyText2"]))
        else:
            story.append(Paragraph(
                "The consequence analysis has been completed successfully. "
                "Results should be reviewed by a qualified process safety "
                "engineer before use in safety-critical decisions.",
                styles["BodyText2"]
            ))
        story.append(Spacer(1, 6))

    # ── 8. Appendix ──
    if sections.get("appendix", True):
        story.append(Paragraph("6. Appendix", styles["SectionH1"]))
        story.append(HorizontalLine())
        story.append(Paragraph("6.1 Methodology", styles["SectionH2"]))
        story.append(Paragraph(
            "This analysis was performed using Rekarisk, a consequence and risk "
            "analysis tool for safety engineers. The models used follow "
            "recognized standards and guidelines:",
            styles["BodyText2"]
        ))
        refs = [
            "CCPS (1999). Guidelines for Consequence Analysis of Chemical Releases.",
            "CCPS (2000). Guidelines for Chemical Process Quantitative Risk Analysis.",
            "TNO Yellow Book (2005). Methods for the Calculation of Physical Effects (CPR 14E).",
            "TNO Purple Book (2005). Guidelines for Quantitative Risk Assessment (CPR 18E).",
            "HSE UK. Assessment of the Dangerous Toxic Load.",
            "API RP 521. Pressure-Relieving and Depressuring Systems.",
            "API RP 752/753. Management of Hazards Associated with Location of Process Plant Buildings.",
            "ISO 17776:2000. Petroleum and Natural Gas Industries — Major Accident Hazard Management.",
            "Kepmen LH No. 13/1995. Baku Mutu Emisi Sumber Tidak Bergerak.",
            "Kingery, C.N., Bulmash, G. (1984). Airblast Parameters from TNT Spherical Air Burst.",
        ]
        for ref in refs:
            story.append(Paragraph(f"• {ref}", styles["BodyText2"]))
            story.append(Spacer(1, 2))

        story.append(Paragraph("6.2 Assumptions & Limitations", styles["SectionH2"]))
        assumptions = [
            "Flat, unobstructed terrain unless terrain data is provided.",
            "Steady-state meteorological conditions for dispersion calculations.",
            "Models are applicable within their validated ranges.",
            "Risk results are indicative and should be verified with site-specific data.",
        ]
        for a in assumptions:
            story.append(Paragraph(f"• {a}", styles["BodyText2"]))
            story.append(Spacer(1, 2))

    # Build PDF with header/footer
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return str(output_path)


# ══════════════════════════════════════════════════════════════════════════════
# Cover page
# ══════════════════════════════════════════════════════════════════════════════

def add_cover_page(
    story: List[Flowable],
    doc: Any,
    project_data: Dict[str, Any],
    cover_info: Optional[Dict[str, str]] = None,
    styles: Optional[dict] = None,
) -> None:
    """Build the cover page as a series of centered flowables.

    Parameters
    ----------
    story : list
        The platypus story list to append to.
    doc : SimpleDocTemplate
        The document (needed for spacing).
    project_data : dict
        Project metadata.
    cover_info : dict, optional
        Override fields: project_name, author, date, version, organization.
    styles : dict
        Paragraph styles.
    """
    if styles is None:
        styles = _build_styles()

    if cover_info is None:
        cover_info = {}

    # Add vertical spacing to center content on cover
    story.append(Spacer(1, 80))

    # Rekarisk logo/text
    story.append(Paragraph("REKARISK", styles["CoverTitle"]))
    story.append(Paragraph(
        "Consequence &amp; Risk Analysis for Safety Engineers",
        styles["CoverSubtitle"]
    ))
    story.append(Spacer(1, 40))

    # Horizontal line
    story.append(HorizontalLine(color=_COLOR_ACCENT))
    story.append(Spacer(1, 30))

    # Project name
    proj_name = cover_info.get("project_name") or project_data.get("name", "Untitled Project")
    story.append(Paragraph(
        f"<b>Project:</b> {proj_name}",
        ParagraphStyle("CoverProj", parent=styles["CoverSubtitle"],
                       fontSize=16, leading=22)
    ))
    story.append(Spacer(1, 20))

    # Info fields
    info_fields = [
        ("Author", cover_info.get("author", project_data.get("author", ""))),
        ("Date", cover_info.get("date", project_data.get("created_at",
                                 datetime.now().strftime("%d %B %Y")))),
        ("Version", cover_info.get("version", project_data.get("format_version", "1.0"))),
        ("Organization", cover_info.get("organization", project_data.get("organization", ""))),
    ]
    for label, value in info_fields:
        if value:
            story.append(Paragraph(f"<b>{label}:</b> {value}", styles["CoverInfo"]))

    story.append(Spacer(1, 30))
    story.append(HorizontalLine(color=_COLOR_ACCENT))
    story.append(Spacer(1, 20))

    # Disclaimer
    story.append(Paragraph(
        "<i>This report is generated by Rekarisk. Results should be reviewed "
        "by a qualified engineer before use in decision-making.</i>",
        ParagraphStyle("Disclaimer", parent=styles["CoverInfo"], fontSize=9)
    ))


# ══════════════════════════════════════════════════════════════════════════════
# Results section
# ══════════════════════════════════════════════════════════════════════════════

def add_results_section(
    story: List[Flowable],
    doc: Any,
    result: Dict[str, Any],
    scenario_num: int = 1,
    styles: Optional[dict] = None,
) -> None:
    """Add a complete results section for one scenario.

    Includes summary table, contour plots, and threshold distances table.
    """
    if styles is None:
        styles = _build_styles()

    scenario_name = result.get("name", f"Scenario {scenario_num}")
    scenario_type = result.get("type", "general").replace("_", " ").title()

    story.append(Paragraph(
        f"3.{scenario_num} {scenario_name} ({scenario_type})",
        styles["SectionH2"]
    ))

    # Summary table
    summary_data = result.get("summary", {})
    if summary_data:
        story.append(Paragraph("Summary", styles["SectionH3"]))
        rows = [[k, str(v)] for k, v in summary_data.items()]
        add_table(story, doc, ["Parameter", "Value"], rows, styles=styles)

    # Contour plots
    plots = result.get("plots", [])
    if plots:
        story.append(Paragraph("Visualization", styles["SectionH3"]))
        for plot in plots:
            if isinstance(plot, dict):
                caption = plot.get("caption", "")
                path = plot.get("path", "")
                fig = plot.get("figure", None)
                if path or fig is not None:
                    add_plot(story, doc, path or fig, caption, styles)

    # Threshold distances table
    thresholds = result.get("thresholds", {})
    if thresholds:
        story.append(Paragraph("Threshold Distances", styles["SectionH3"]))
        rows = []
        for thresh_label, distance in thresholds.items():
            rows.append([thresh_label, f"{distance:.1f} m" if isinstance(distance, (int, float)) else str(distance)])
        add_table(story, doc, ["Threshold", "Distance (m)"], rows, styles=styles)

    # Detailed results table
    detail_rows = result.get("table_rows", [])
    detail_headers = result.get("table_headers", [])
    if detail_rows and detail_headers:
        story.append(Paragraph("Detailed Results", styles["SectionH3"]))
        add_table(story, doc, detail_headers, detail_rows, styles=styles)

    story.append(Spacer(1, 12))


# ══════════════════════════════════════════════════════════════════════════════
# Table helper
# ══════════════════════════════════════════════════════════════════════════════

def add_table(
    story: List[Flowable],
    doc: Any,
    headers: List[str],
    rows: List[List[Any]],
    title: str = "",
    styles: Optional[dict] = None,
    col_widths: Optional[List[float]] = None,
) -> None:
    """Add a formatted table to the story.

    Parameters
    ----------
    story : list
        The platypus story.
    doc : SimpleDocTemplate
        The document (needed for page width).
    headers : list[str]
        Column headers.
    rows : list[list]
        Data rows (list of lists).
    title : str
        Optional table caption.
    styles : dict
        Paragraph styles.
    col_widths : list[float], optional
        Explicit column widths. If None, auto-calculated.
    """
    if styles is None:
        styles = _build_styles()

    if title:
        story.append(Paragraph(title, styles["Caption"]))

    # Wrap all cells in Paragraph for flowable support
    table_data = [[Paragraph(h, styles["TableHeader"]) for h in headers]]
    for row in rows:
        table_data.append([
            Paragraph(str(cell), styles["TableCell"])
            for cell in row
        ])

    avail_width = doc.width  # _PAGE_W - 2 * _MARGIN
    if col_widths is None:
        ncols = len(headers)
        if ncols > 0:
            col_widths = [avail_width / ncols] * ncols
        else:
            col_widths = [avail_width]

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)

    # Style: banded rows, header background
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), _COLOR_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), _COLOR_HEADER_FG),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, _COLOR_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]

    # Banded rows
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), _COLOR_ROW_ALT))

    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)
    story.append(Spacer(1, 6))


# ══════════════════════════════════════════════════════════════════════════════
# Plot helper
# ══════════════════════════════════════════════════════════════════════════════

def add_plot(
    story: List[Flowable],
    doc: Any,
    figure_or_path: Any,
    caption: str = "",
    styles: Optional[dict] = None,
    max_width: float = 480,
    max_height: float = 360,
) -> None:
    """Add a matplotlib figure or image file as an embedded plot.

    Parameters
    ----------
    story : list
        The platypus story.
    doc : SimpleDocTemplate
        The document.
    figure_or_path : matplotlib.figure.Figure or str or Path
        A matplotlib figure or a path to an image file.
    caption : str
        Figure caption.
    styles : dict
        Paragraph styles.
    max_width, max_height : float
        Maximum dimensions in points.
    """
    if styles is None:
        styles = _build_styles()

    try:
        img_path = None
        tmp_file = None

        if isinstance(figure_or_path, (str, Path)):
            img_path = str(figure_or_path)
        else:
            # Assume matplotlib Figure
            tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_file.close()
            figure_or_path.savefig(tmp_file.name, dpi=120, bbox_inches="tight",
                                   facecolor="white", edgecolor="none")
            img_path = tmp_file.name

        if img_path and os.path.exists(img_path):
            # Auto-fit
            img = Image(img_path)
            aspect = img.imageWidth / max(img.imageHeight, 1)
            if aspect > max_width / max_height:
                w = min(img.imageWidth, max_width)
                h = w / aspect
            else:
                h = min(img.imageHeight, max_height)
                w = h * aspect
            img = Image(img_path, width=w, height=h)
            story.append(img)

            if caption:
                story.append(Paragraph(caption, styles["Caption"]))

    except Exception:
        story.append(Paragraph(
            f"<i>[Plot could not be rendered: {caption or 'unnamed'}]</i>",
            styles["Caption"]
        ))
    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_toc(project_data: Dict[str, Any], sections: Dict[str, bool]) -> List[str]:
    """Build a simple list-based table of contents."""
    items = []
    if sections.get("summary"):
        items.append("1. Executive Summary")
    if sections.get("input"):
        items.append("2. Input Data")
        items.append("   2.1 Weather Conditions")
        items.append("   2.2 Substances")
        items.append("   2.3 Scenario Parameters")
    if sections.get("results"):
        items.append("3. Results")
    if sections.get("qra"):
        items.append("4. Quantitative Risk Assessment")
    if sections.get("conclusion"):
        items.append("5. Conclusion")
    if sections.get("appendix"):
        items.append("6. Appendix")
        items.append("   6.1 Methodology")
        items.append("   6.2 Assumptions & Limitations")
    return items


def _format_weather(weather: Dict[str, Any]) -> str:
    """Format weather data as HTML for reportlab Paragraph."""
    parts = []
    if "wind_speed" in weather:
        parts.append(f"Wind Speed: {weather['wind_speed']} m/s")
    if "wind_direction" in weather:
        parts.append(f"Direction: {weather['wind_direction']}°")
    if "temperature" in weather or "ambient_temp" in weather:
        t = weather.get("temperature", weather.get("ambient_temp"))
        parts.append(f"Temperature: {t} °C")
    if "stability_class" in weather:
        parts.append(f"Stability: Pasquill {weather['stability_class']}")
    if "humidity" in weather:
        parts.append(f"Humidity: {weather['humidity']}%")
    if "inversion_height" in weather:
        parts.append(f"Inversion Height: {weather['inversion_height']} m")
    if "roughness_length" in weather:
        parts.append(f"Roughness Length: {weather['roughness_length']} m")
    return "&nbsp;&nbsp;&nbsp;".join(parts)


def _add_qra_section(
    story: List[Flowable],
    qra_result: Dict[str, Any],
    num: int,
    styles: dict,
) -> None:
    """Add a QRA subsection with IR contours, FN curve, risk matrix."""
    story.append(Paragraph(
        f"4.{num} {qra_result.get('name', f'QRA Case {num}')}",
        styles["SectionH2"]
    ))

    # IR contours
    ir_grid = qra_result.get("ir_grid")
    ir_thresholds = qra_result.get("ir_thresholds", {})
    if ir_thresholds:
        story.append(Paragraph("Individual Risk Contours", styles["SectionH3"]))
        ir_rows = [[label, f"{dist:.1f} m" if isinstance(dist, (int, float)) else str(dist)]
                   for label, dist in ir_thresholds.items()]
        add_table(story, None, ["Risk Level", "Distance (m)"], ir_rows, styles=styles)

    # FN curve
    fn_data = qra_result.get("fn_data")
    if fn_data:
        story.append(Paragraph("Societal Risk (FN Curve)", styles["SectionH3"]))
        fn_rows = [["N (fatalities)", "F (frequency/year)"]]
        if isinstance(fn_data, dict):
            n_vals = fn_data.get("n", [])
            f_vals = fn_data.get("f", [])
            for n, f in zip(n_vals, f_vals):
                fn_rows.append([f"{n:.0f}", f"{f:.2e}"])
        elif isinstance(fn_data, list):
            for row in fn_data:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    fn_rows.append([str(row[0]), str(row[1])])
        add_table(story, None, fn_rows[0], fn_rows[1:], styles=styles)

    # Risk matrix
    risk_matrix = qra_result.get("risk_matrix")
    if risk_matrix:
        story.append(Paragraph("Risk Matrix", styles["SectionH3"]))
        if isinstance(risk_matrix, list) and len(risk_matrix) > 0:
            if isinstance(risk_matrix[0], list):
                add_table(story, None,
                          ["Likelihood \\ Consequence"] + [str(c) for c in range(len(risk_matrix[0]))],
                          risk_matrix, styles=styles)
            else:
                add_table(story, None, ["Entry"], [[str(r)] for r in risk_matrix], styles=styles)

    # Plots
    qra_plots = qra_result.get("plots", [])
    for plot in qra_plots:
        if isinstance(plot, dict):
            fig = plot.get("figure") or plot.get("path")
            caption = plot.get("caption", "")
            if fig:
                add_plot(story, None, fig, caption, styles)
