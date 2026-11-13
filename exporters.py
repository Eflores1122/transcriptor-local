"""Export helpers for the local transcription app."""

from __future__ import annotations

import io
import json
import os
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def to_json_bytes(result: Dict[str, Any]) -> bytes:
    return json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")


def _font_candidates() -> Tuple[Optional[Path], Optional[Path]]:
    candidates = [
        (
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ),
        (
            Path("/Library/Fonts/Arial Unicode.ttf"),
            Path("/Library/Fonts/Arial Bold.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if regular.exists():
            return regular, bold if bold.exists() else None
    return None, None


def _register_fonts() -> Tuple[str, str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    regular_path, bold_path = _font_candidates()
    if not regular_path:
        return "Helvetica", "Helvetica-Bold"

    regular_name = "TranscriptSans"
    bold_name = "TranscriptSansBold"
    try:
        pdfmetrics.registerFont(TTFont(regular_name, str(regular_path)))
        if bold_path:
            pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
        else:
            bold_name = regular_name
        return regular_name, bold_name
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def _format_seconds(seconds: Any) -> str:
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        total = 0
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def to_pdf_bytes(result: Dict[str, Any]) -> bytes:
    """Render a readable, paginated PDF with timestamps for every segment."""

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    regular_font, bold_font = _register_fonts()
    metadata = result.get("metadata", {})
    source_name = str(metadata.get("source_name", "archivo"))
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title=f"Transcripción — {source_name}",
        author="Transcriptor local",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TranscriptTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#102A43"),
        alignment=TA_LEFT,
        spaceAfter=5 * mm,
    )
    metadata_style = ParagraphStyle(
        "TranscriptMeta",
        parent=styles["Normal"],
        fontName=regular_font,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#486581"),
    )
    timestamp_style = ParagraphStyle(
        "TranscriptTimestamp",
        parent=styles["Normal"],
        fontName=bold_font,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#127C8A"),
        spaceAfter=1.5 * mm,
    )
    segment_style = ParagraphStyle(
        "TranscriptSegment",
        parent=styles["Normal"],
        fontName=regular_font,
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#243B53"),
        spaceAfter=4 * mm,
    )

    story = [Paragraph("Transcripción", title_style)]
    language = metadata.get("language") or "detección automática"
    duration = _format_seconds(metadata.get("duration_seconds"))
    model = metadata.get("model", "Whisper")
    story.append(
        Paragraph(
            f"<b>Fuente:</b> {escape(source_name)}<br/>"
            f"<b>Duración:</b> {duration} &nbsp; <b>Idioma:</b> {escape(str(language))}"
            f"<br/><b>Modelo:</b> {escape(str(model))} &nbsp; "
            f"<b>Marcas de tiempo:</b> {'sí' if metadata.get('word_timestamps') else 'segmentos'}",
            metadata_style,
        )
    )
    story.append(Spacer(1, 8 * mm))

    header = [
        Paragraph("<b>Tiempo</b>", metadata_style),
        Paragraph("<b>Texto</b>", metadata_style),
    ]
    table = Table([header], colWidths=[28 * mm, 146 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#D9F0F2")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#102A43")),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#9FB3C8")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#BCCCDC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 3 * mm))

    for segment in result.get("segments", []):
        start = _format_seconds(segment.get("start"))
        end = _format_seconds(segment.get("end"))
        text = escape(str(segment.get("text", ""))).replace("\n", "<br/>")
        story.append(Paragraph(f"{start} — {end}", timestamp_style))
        story.append(Paragraph(text or " ", segment_style))

    def draw_footer(canvas: Any, _doc: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, 12 * mm, 192 * mm, 12 * mm)
        canvas.setFont(regular_font, 7.5)
        canvas.setFillColor(colors.HexColor("#829AB1"))
        canvas.drawString(18 * mm, 7 * mm, "Transcriptor local · procesamiento en este equipo")
        canvas.drawRightString(192 * mm, 7 * mm, f"Página {canvas.getPageNumber()}")
        canvas.restoreState()

    document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return buffer.getvalue()
