"""A short, printable PDF summary of a batch prediction (built in memory, never saved)."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BAND_COLOURS = {
    "high": colors.HexColor("#d03b3b"),
    "moderate": colors.HexColor("#ec835a"),
    "low": colors.HexColor("#0ca30c"),
}


def batch_report_pdf(
    result: dict[str, Any], dataset_title: str, model_label: str, disclaimer: str, max_rows: int = 60
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="FairPath batch report",
    )
    ss = getSampleStyleSheet()
    small = ParagraphStyle(
        "small", parent=ss["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#52514e")
    )
    story = [
        Paragraph("FairPath - batch risk summary", ss["Title"]),
        Paragraph(
            f"{dataset_title} &middot; model: {model_label} &middot; "
            f"generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
            small,
        ),
        Spacer(1, 6 * mm),
    ]
    s = result["summary"]
    counts = s["bands"]
    story.append(
        Paragraph(
            f"<b>{result['n']}</b> students scored. High risk: <b>{counts.get('high', 0)}</b>, moderate: "
            f"<b>{counts.get('moderate', 0)}</b>, low: <b>{counts.get('low', 0)}</b>. "
            f"Flagged for outreach (risk &ge; {s['threshold']:.0%}): <b>{s['flagged']}</b>. "
            f"Uncertain predictions: <b>{s['uncertain']}</b>.",
            ss["Normal"],
        )
    )
    story.append(Spacer(1, 5 * mm))

    rows = sorted(result["rows"], key=lambda r: -r["risk"])[:max_rows]
    data = [["#", "Student", "Risk", "Band", "Confidence", "Main factors"]]
    for r in rows:
        data.append(
            [
                str(r["row"] + 1),
                r.get("student_id") or "-",
                f"{r['risk']:.0%}",
                r["band"],
                r["confidence"],
                Paragraph(", ".join(r["top_factors"][:3]), small),
            ]
        )
    t = Table(data, colWidths=[10 * mm, 24 * mm, 16 * mm, 20 * mm, 22 * mm, 82 * mm], repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#c3c2b7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f6f3")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    for i, r in enumerate(rows, start=1):
        style.append(("TEXTCOLOR", (3, i), (3, i), BAND_COLOURS.get(r["band"], colors.black)))
    t.setStyle(TableStyle(style))
    story.append(t)
    if result["n"] > max_rows:
        story.append(Paragraph(f"Showing the {max_rows} highest-risk rows of {result['n']}.", small))
    story += [Spacer(1, 6 * mm), Paragraph(disclaimer, small)]
    doc.build(story)
    return buf.getvalue()
