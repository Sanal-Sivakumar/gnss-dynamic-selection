"""Create the project report and viva guide PDFs from the checked-in source files.

The generated documents describe the ADAPTIVE-WEIGHT receiver-selection system:

    W_i = a*T_i + b*S_i + c*SNR_i + d*DOP_i      Best Source = argmax(W)

The heart of the system is an ML dynamic weight learner: a compact MLP maps
conditioning features X to softmax coefficients [a,b,c,d] = f(X), which the
reliability equation then uses to score each receiver. The predefined fixed
coefficients (0.25, 0.25, 0.30, 0.20) are only a bring-up/emergency fallback
and a comparison baseline. Model A (predefined fixed), Model B (grid-search
optimized fixed) and Model C (learned dynamic weights, with and without
hysteresis) are compared on held-out sessions.

The documents deliberately distinguish the implemented prototype (Phase 6,
live-integrated engine) from the offline simulation study. They are intended
for academic reporting and viva preparation, not as evidence of field-tested
GNSS performance.
"""
from __future__ import annotations

from pathlib import Path
import json

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.graphics.charts.barcharts import VerticalBarChart


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
METRICS = json.loads((ROOT / "ml" / "artifacts" / "metrics.json").read_text())
EVAL = json.loads((ROOT / "ml" / "artifacts" / "evaluation.json").read_text())

TEST = METRICS["test"]
SCEN = METRICS["test_by_scenario"]
LW = METRICS["learned_weights"]

NAVY = HexColor("#102A43")
BLUE = HexColor("#1769AA")
TEAL = HexColor("#0A7C86")
GOLD = HexColor("#B8860B")
GRAY = HexColor("#8A97A5")
INK = HexColor("#000000")
MUTED = HexColor("#3D3D3D")
PALE = HexColor("#F2F2F2")
GRID = HexColor("#B9B9B9")
WHITE = colors.white
PAGE_W, PAGE_H = A4

# Editable cover / certificate details (replace placeholders before submission).
STUDENT_NAME = "Sanal Sivakumar"
REGISTER_NO = "________________"
SEMESTER = "VIII Semester"
ACADEMIC_YEAR = "2026"
INST_LINE_1 = "DIVISION OF COMPUTER SCIENCE AND ENGINEERING"
INST_LINE_2 = "SCHOOL OF ENGINEERING"
INST_LINE_3 = "COCHIN UNIVERSITY OF SCIENCE AND TECHNOLOGY"
GUIDE_NAME = "________________"
GUIDE_DESIGNATION = "Project Guide"
GUIDE_CONTACT = "________________"
COVER_DATE = "September 2026"

# Model C results with and without the live-default hysteresis threshold.
MODEL_KEYS = [
    ("Model A - predefined fixed",    "A",   "original_fixed_weights_model_A"),
    ("Model B - optimized fixed",     "B",   "optimized_fixed_weights_model_B"),
    ("Model C - dynamic ML",          "C",   "dynamic_ml_weights_model_C"),
    ("Model C + hysteresis (0.05)",   "C+h", "dynamic_ml_with_hysteresis"),
]
SCENARIO_NAMES = ["open_sky", "partial_obstruction", "urban_multipath", "dropout", "mixed"]
SCENARIO_LABELS = ["Open sky", "Obstruction", "Multipath", "Dropout", "Mixed"]


def escape(text: str) -> str:
    return (text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def to_roman(n):
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"),
            (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"),
            (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for value, sym in vals:
        while n >= value:
            out += sym
            n -= value
    return out


def make_styles():
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Times-Roman", fontSize=11.5,
            leading=17.5, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=7,
        ),
        "body_small": ParagraphStyle(
            "BodySmall", parent=base["BodyText"], fontName="Times-Roman", fontSize=10.3,
            leading=14.8, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=5,
        ),
        "title": ParagraphStyle(
            "Title", fontName="Times-Bold", fontSize=22, leading=30,
            alignment=TA_CENTER, textColor=INK, spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", fontName="Times-Roman", fontSize=13, leading=20,
            alignment=TA_CENTER, textColor=INK, spaceAfter=6,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta", fontName="Times-Roman", fontSize=12, leading=19,
            alignment=TA_CENTER, textColor=INK, spaceAfter=3,
        ),
        "cover_label": ParagraphStyle(
            "CoverLabel", fontName="Times-Bold", fontSize=12, leading=19,
            alignment=TA_CENTER, textColor=INK, spaceAfter=3,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle", fontName="Times-Bold", fontSize=17, leading=24,
            alignment=TA_CENTER, textColor=INK, spaceAfter=6,
        ),
        "cover_head": ParagraphStyle(
            "CoverHead", fontName="Times-Bold", fontSize=13, leading=20,
            alignment=TA_CENTER, textColor=INK,
        ),
        "h1": ParagraphStyle(
            "Heading1", parent=base["Heading1"], fontName="Times-Bold", fontSize=14,
            leading=19, textColor=INK, spaceBefore=14, spaceAfter=7,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2", parent=base["Heading2"], fontName="Times-Bold", fontSize=11.8,
            leading=16, textColor=INK, spaceBefore=9, spaceAfter=4,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3", parent=base["Heading3"], fontName="Times-BoldItalic", fontSize=10.8,
            leading=15, textColor=INK, spaceBefore=7, spaceAfter=3,
            keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "Caption", fontName="Times-Italic", fontSize=9.3, leading=12,
            alignment=TA_CENTER, textColor=INK, spaceBefore=3, spaceAfter=8,
        ),
        "cert": ParagraphStyle(
            "Cert", parent=base["BodyText"], fontName="Times-Roman", fontSize=12,
            leading=20, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=8,
        ),
        "cert_sig": ParagraphStyle(
            "CertSig", parent=base["BodyText"], fontName="Times-Roman", fontSize=12,
            leading=16, textColor=INK,
        ),
        "toc": ParagraphStyle(
            "TOC", parent=base["BodyText"], fontName="Times-Roman", fontSize=11,
            leading=16.5, leftIndent=14, firstLineIndent=-14, textColor=INK,
        ),
        "toc_h1": ParagraphStyle(
            "TOCH1", parent=base["BodyText"], fontName="Times-Bold", fontSize=11,
            leading=16.5, leftIndent=14, firstLineIndent=-14, textColor=INK,
        ),
        "quote": ParagraphStyle(
            "Quote", parent=base["BodyText"], fontName="Times-Italic", fontSize=11,
            leading=16, leftIndent=12, rightIndent=12, textColor=INK, spaceBefore=6, spaceAfter=7,
        ),
        "table": ParagraphStyle(
            "Table", parent=base["BodyText"], fontName="Times-Roman", fontSize=9,
            leading=12, textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHead", parent=base["BodyText"], fontName="Times-Bold", fontSize=9,
            leading=12, textColor=WHITE, alignment=TA_CENTER,
        ),
        "code": ParagraphStyle(
            "Code", fontName="Courier-Bold", fontSize=8, leading=11, textColor=INK,
        ),
        "front_head": ParagraphStyle(
            "FrontHead", parent=base["Heading1"], fontName="Times-Bold", fontSize=14,
            leading=19, textColor=INK, spaceBefore=4, spaceAfter=7,
            keepWithNext=True, alignment=TA_LEFT,
        ),
        "figcap": ParagraphStyle(
            "FigCaption", fontName="Times-Roman", fontSize=9.5, leading=12.5,
            alignment=TA_CENTER, textColor=INK, spaceBefore=4, spaceAfter=8,
        ),
        "key": ParagraphStyle(
            "Key", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8.8,
            leading=11.5, textColor=NAVY,
        ),
    }


S = make_styles()


class AcademicDoc(BaseDocTemplate):
    def __init__(self, filename, label, **kwargs):
        super().__init__(filename, **kwargs)
        self.label = label
        self._body_start = None
        frame = Frame(23 * mm, 20 * mm, PAGE_W - 46 * mm, PAGE_H - 44 * mm,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([])
        from reportlab.platypus import PageTemplate
        self.addPageTemplates([PageTemplate(id="academic", frames=frame, onPage=self._pagenum)])

    def _pagenum(self, canvas, doc):
        page = doc.page
        if page == 1:
            return
        canvas.saveState()
        canvas.setFont("Times-Roman", 10.5)
        canvas.setFillColor(INK)
        if self._body_start is None or page < self._body_start:
            label = to_roman(page)
        else:
            label = str(page - self._body_start + 1)
        canvas.drawCentredString(PAGE_W / 2, 9.5 * mm, label)
        canvas.restoreState()

    def body_page_label(self, page):
        if self._body_start is None or page < self._body_start:
            return to_roman(page)
        return str(page - self._body_start + 1)

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style = flowable.style.name
            text = flowable.getPlainText()
            if style == "Heading1":
                import re as _re
                if self._body_start is None and _re.match(r"^\d+\.", text):
                    self._body_start = self.page
                key = "h1-" + str(doc_safe_id(text))
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, 0, False)
                self.notify("TOCEntry", (0, text, self.page, key))
            elif style == "Heading2":
                key = "h2-" + str(doc_safe_id(text))
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, 1, False)
                self.notify("TOCEntry", (1, text, self.page, key))
            elif style == "FigCaption":
                key = "fig-" + str(doc_safe_id(text))
                self.canv.bookmarkPage(key)
                self.notify("FigEntry", (0, text, self.page, key))


def doc_safe_id(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[:80]


def p(text: str, style: str = "body"):
    return Paragraph(text, S[style])


def h1(text: str):
    return Paragraph(text, S["h1"])


def h2(text: str):
    return Paragraph(text, S["h2"])


def h3(text: str):
    return Paragraph(text, S["h3"])


def caption(text: str):
    return Paragraph(text, S["caption"])


def figcaption(text: str):
    return Paragraph(text, S["figcap"])


def bullets(items, style="body_small"):
    return ListFlowable(
        [ListItem(p(item, style), leftIndent=10) for item in items],
        bulletType="bullet", start="circle", leftIndent=17, bulletFontName="Times-Roman",
        bulletFontSize=7, bulletOffsetY=2, spaceAfter=6,
    )


def numbered(items, style="body_small"):
    return ListFlowable(
        [ListItem(p(item, style), leftIndent=12) for item in items],
        bulletType="1", leftIndent=20, bulletFontName="Times-Roman",
        bulletFontSize=9, spaceAfter=6,
    )


def data_table(rows, widths, header=True, font_size=8.3):
    cooked = []
    for r, row in enumerate(rows):
        cooked.append([p(str(v), "table_head" if (header and r == 0) else "table") for v in row])
    table = Table(cooked, colWidths=widths, repeatRows=1 if header else 0, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY if header else WHITE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE if header else INK),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
    ]))
    return table


def box(d: Drawing, x, y, w, h, text, fill=PALE, stroke=BLUE, font_size=8.5):
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=fill, strokeColor=stroke, strokeWidth=1))
    lines = text.split("\n")
    line_h = font_size + 2
    y0 = y + h / 2 + ((len(lines) - 1) * line_h) / 2 - font_size * 0.3
    for i, line in enumerate(lines):
        d.add(String(x + w / 2, y0 - i * line_h, line, textAnchor="middle",
                     fontName="Helvetica-Bold", fontSize=font_size, fillColor=INK))


def arrow(d: Drawing, x1, y1, x2, y2, color=TEAL):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.5))
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        d.add(Line(x2, y2, x2 - 5 * direction, y2 + 3, strokeColor=color, strokeWidth=1.5))
        d.add(Line(x2, y2, x2 - 5 * direction, y2 - 3, strokeColor=color, strokeWidth=1.5))
    else:
        direction = 1 if y2 > y1 else -1
        d.add(Line(x2, y2, x2 + 3, y2 - 5 * direction, strokeColor=color, strokeWidth=1.5))
        d.add(Line(x2, y2, x2 - 3, y2 - 5 * direction, strokeColor=color, strokeWidth=1.5))


def system_drawing():
    d = Drawing(485, 225)
    box(d, 10, 158, 88, 45, "Receiver A\nGNSS readings")
    box(d, 10, 62, 88, 45, "Receiver B\nGNSS readings")
    box(d, 128, 110, 95, 45, "ESP32\nUART + C/N0 tracker", fill=HexColor("#E8F6F3"), stroke=TEAL)
    box(d, 253, 110, 130, 45, "Raspberry Pi\nDecisionEngine", fill=HexColor("#FDF3DE"), stroke=GOLD)
    box(d, 415, 110, 60, 45, "Web\nview", fill=HexColor("#EEF2FF"), stroke=BLUE)
    arrow(d, 98, 180, 128, 137)
    arrow(d, 98, 84, 128, 127)
    arrow(d, 223, 133, 253, 133)
    arrow(d, 383, 133, 415, 133)
    d.add(String(42, 213, "Hardware data path", fontName="Helvetica", fontSize=8.5, fillColor=MUTED))
    d.add(String(318, 168, "parse, reliability terms,\ndynamic weights, hysteresis,\nfallback, JSONL log",
                 fontName="Helvetica", fontSize=7.0, fillColor=MUTED))
    box_widgets_ml(d)
    return d


def box_widgets_ml(d):
    # Offline ML block feeding the live engine (this is the Phase 6 wiring).
    box(d, 128, 4, 110, 46, "Offline training\n(baselines A/B, Model C)", fill=HexColor("#F7EDF7"),
        stroke=HexColor("#8E3A8E"), font_size=7.2)
    box(d, 282, 4, 112, 46, "model.npz\n16 features -> a,b,c,d", fill=HexColor("#F7EDF7"),
        stroke=HexColor("#8E3A8E"), font_size=7.2)
    box(d, 390, 4, 85, 46, "DecisionEngine\nweights + fallback", fill=HexColor("#EEF2FF"),
        stroke=BLUE, font_size=7.2)
    arrow(d, 238, 27, 282, 27, color=HexColor("#8E3A8E"))
    arrow(d, 394, 50, 318, 106, color=HexColor("#8E3A8E"))
    d.add(String(245, 62, "Exported {mean, std, w1..b3} ride in model.npz; the engine loads them at runtime",
                 fontName="Helvetica", fontSize=7.6, fillColor=MUTED))


def model_drawing():
    d = Drawing(485, 235)
    box(d, 8, 128, 120, 66, "16 conditioning features\nT,S,SNR,DOP x both receivers\n+ delta + temporal", fill=PALE, stroke=BLUE, font_size=7.2)
    box(d, 148, 142, 92, 52, "Normalize\nz = (x - mean) / std", fill=PALE, stroke=BLUE, font_size=7.6)
    box(d, 260, 142, 92, 52, "Dense 16 -> 24\nReLU\n408 parameters", fill=PALE, stroke=BLUE, font_size=7.2)
    box(d, 372, 142, 95, 52, "Dense 24 -> 24\nReLU\n600 parameters", fill=PALE, stroke=BLUE, font_size=7.2)
    arrow(d, 128, 161, 148, 161)
    arrow(d, 240, 168, 260, 168)
    arrow(d, 352, 168, 372, 168)
    box(d, 300, 45, 110, 62, "Dense 24 -> 4\nsoftmax  a,b,c,d\nsum = 1\n100 parameters", fill=HexColor("#E8F6F3"), stroke=TEAL, font_size=7.2)
    box(d, 80, 45, 160, 62, "Reliability equation\nW = a*T + b*S + c*SNR + d*DOP\nper receiver; argmax(W) selects", fill=HexColor("#FDF3DE"), stroke=GOLD, font_size=7.2)
    arrow(d, 330, 130, 355, 107)
    arrow(d, 300, 76, 240, 76)
    d.add(String(8, 212, "Trainable parameters: 1,108 (408 + 600 + 100). Stored normalization values: 32 (mean and std for 16 inputs). Output coefficients are the adaptive weights used by the reliability equation, not a black-box position estimate.",
                 fontName="Helvetica", fontSize=7.4, fillColor=MUTED))
    return d


def accuracy_chart():
    d = Drawing(430, 218)
    chart = VerticalBarChart()
    chart.x = 52
    chart.y = 42
    chart.height = 140
    chart.width = 340
    series = []
    for _, _, key in MODEL_KEYS:
        series.append([TEST[key]["mean_error_m"], TEST[key]["median_error_m"], TEST[key]["p95_error_m"]])
    chart.data = series
    chart.categoryAxis.categoryNames = ["Mean error", "Median", "95th pct"]
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 10
    chart.valueAxis.valueStep = 2
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7.5
    chart.bars[0].fillColor = GRAY
    chart.bars[1].fillColor = BLUE
    chart.bars[2].fillColor = TEAL
    chart.bars[3].fillColor = GOLD
    chart.groupSpacing = 9
    chart.barSpacing = 2.5
    d.add(chart)
    d.add(String(52, 194, "Metres", fontName="Helvetica", fontSize=8, fillColor=MUTED))
    d.add(String(222, 15, "Gray: A   Blue: B   Teal: C   Gold: C+hysteresis",
                 textAnchor="middle", fontName="Helvetica", fontSize=8, fillColor=MUTED))
    return d


def scenario_chart():
    d = Drawing(445, 225)
    chart = VerticalBarChart()
    chart.x = 52
    chart.y = 45
    chart.height = 140
    chart.width = 345
    chart.data = [[SCEN[n]["model_A"]["mean_error_m"] for n in SCENARIO_NAMES],
                  [SCEN[n]["model_B"]["mean_error_m"] for n in SCENARIO_NAMES],
                  [SCEN[n]["model_C"]["mean_error_m"] for n in SCENARIO_NAMES]]
    chart.categoryAxis.categoryNames = SCENARIO_LABELS
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 6.8
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 6
    chart.valueAxis.valueStep = 1
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7.5
    chart.bars[0].fillColor = BLUE
    chart.bars[1].fillColor = GRAY
    chart.bars[2].fillColor = TEAL
    chart.groupSpacing = 7
    chart.barSpacing = 3
    d.add(chart)
    d.add(String(52, 202, "Mean selected error in metres", fontName="Helvetica", fontSize=8, fillColor=MUTED))
    d.add(String(230, 15, "Blue: A   Gray: B   Teal: C", textAnchor="middle", fontName="Helvetica", fontSize=8, fillColor=MUTED))
    return d


def confusion_matrix_drawing(key="dynamic_ml_with_hysteresis", width=440, height=255):
    """Heat-map of the 2x2 selector-vs-oracle counts for one selector.

    Rows are the receiver the oracle would have chosen (lower actual error);
    columns are the receiver the selector actually picked. The diagonal is
    agreement; the off-diagonal cells are the mistaken picks.
    """
    cm = EVAL["confusion"][key]
    m = cm["matrix"]
    n = cm["n"]
    acc = cm["accuracy_vs_oracle"]
    d = Drawing(width, height)
    cell_w = 92
    cell_h = 62
    ox, oy = 118, 46
    row_labels = ["Oracle: receiver 0\nhad the lower error",
                  "Oracle: receiver 1\nhad the lower error"]
    col_labels = ["Selected\nreceiver 0", "Selected\nreceiver 1"]
    for c, label in enumerate(col_labels):
        d.add(String(ox + c * cell_w + cell_w / 2, oy + 2 * cell_h + 30, label,
                     textAnchor="middle", fontName="Helvetica-Bold",
                     fontSize=7.6, fillColor=INK))
    for r, label in enumerate(row_labels):
        d.add(String(ox - 46, oy + (1 - r) * cell_h + cell_h / 2.45, label,
                     textAnchor="middle", fontName="Helvetica", fontSize=6.8,
                     fillColor=MUTED))
    for r in range(2):
        for c in range(2):
            val = m[r][c]
            pct = val / n * 100
            frac = val / n
            if r == c:
                intensity = round(0.10 + 0.85 * frac)
                fill = HexColor("#%02x%02x%02x" % (int(255 - 70 * frac),
                                                   int(255 - 20 * frac),
                                                   int(255 - 60 * frac)))
            else:
                fill = HexColor("#%02x%02x%02x" % (255, int(235 - 90 * frac),
                                                   int(230 - 90 * frac)))
            d.add(Rect(ox + c * cell_w, oy + (1 - r) * cell_h, cell_w, cell_h,
                       fillColor=fill, strokeColor=GRID, strokeWidth=0.6))
            d.add(String(ox + c * cell_w + cell_w / 2,
                         oy + (1 - r) * cell_h + cell_h / 2 + 6,
                         f"{val:,}", textAnchor="middle",
                         fontName="Helvetica-Bold", fontSize=13, fillColor=INK))
            d.add(String(ox + c * cell_w + cell_w / 2,
                         oy + (1 - r) * cell_h + cell_h / 2 - 10,
                         f"{pct:.1f}%", textAnchor="middle",
                         fontName="Helvetica", fontSize=7.5, fillColor=MUTED))
    d.add(String(width / 2, height - 8, f"n = {n:,} available test epochs"
                 f"      oracle agreement = {acc*100:.1f}%",
                 textAnchor="middle", fontName="Helvetica-Bold", fontSize=8,
                 fillColor=INK))
    d.add(String(width / 2, 16, "Diagonal: selector agrees with the oracle.  "
                 "Off diagonal: selector picked the other receiver.",
                 textAnchor="middle", fontName="Helvetica", fontSize=6.6,
                 fillColor=MUTED))
    return d


def cover_page(title, subtitle, kind, author="Sanal Sivakumar"):
    return [
        Paragraph(kind.upper(), S["cover_label"]),
        Paragraph("B.Tech in Computer Science and Engineering", S["cover_title"]),
        Spacer(1, 6 * mm),
        Paragraph(INST_LINE_1, S["cover_meta"]),
        Paragraph(INST_LINE_2, S["cover_meta"]),
        Paragraph(INST_LINE_3, S["cover_meta"]),
        Spacer(1, 18 * mm),
        Paragraph(author, S["cover_title"]),
        Paragraph(REGISTER_NO, S["cover_meta"]),
        Spacer(1, 12 * mm),
        Paragraph("Project Details", S["cover_label"]),
        Paragraph(title, S["cover_meta"]),
        Paragraph(subtitle, S["cover_meta"]),
        Spacer(1, 12 * mm),
        Paragraph("Guide Details", S["cover_label"]),
        Paragraph(GUIDE_NAME, S["cover_meta"]),
        Paragraph(GUIDE_DESIGNATION, S["cover_meta"]),
        Paragraph(GUIDE_CONTACT, S["cover_meta"]),
        Spacer(1, 16 * mm),
        Paragraph(COVER_DATE.upper(), S["cover_label"]),
        PageBreak(),
    ]


def certificate_page():
    return [
        Paragraph(INST_LINE_1, S["cover_label"]),
        Paragraph(INST_LINE_2, S["cover_label"]),
        Paragraph(INST_LINE_3, S["cover_label"]),
        Spacer(1, 8 * mm),
        Paragraph("CERTIFICATE", S["cover_title"]),
        Spacer(1, 8 * mm),
        p("Certified that this is a Bonafide record of the project work done by "
          + f"<b>{escape(STUDENT_NAME)}</b> ({escape(REGISTER_NO)}) of {SEMESTER}, Computer Science and Engineering "
          + f"in the year {ACADEMIC_YEAR} in partial fulfilment of the requirements for the award of the Degree of "
          + "Bachelor of Technology in Computer Science and Engineering of Cochin University of Science and "
          + "Technology with distinction.", "cert"),
        Spacer(1, 30 * mm),
        Table([[Paragraph(("\u2003" * 8) + "_______________________________", S["cert_sig"]),
                Paragraph(("\u2003" * 8) + "_______________________________", S["cert_sig"])],
               [Paragraph("Project Guide", S["cert_sig"]),
                Paragraph("Head of Division", S["cert_sig"])]],
              colWidths=[9.5 * cm, 9.5 * cm], hAlign="CENTER"),
        PageBreak(),
    ]


def front_matter(doc):
    story = [h1("Acknowledgment")]
    story += [p("I would like to convey my heartfelt gratitude to all those who supported and guided me "
                "throughout this project. I am especially thankful to my project guide for constant support, "
                "valuable feedback and encouragement at every stage of this work. His guidance helped shape a "
                "vague idea into a complete, well-tested system."),
              p("I am profoundly grateful to the Head of Division for the academic environment and for allowing "
                "the use of the facilities needed to build and test the hardware and software components of this "
                "project. I also thank the faculty members whose courses in embedded systems, machine learning and "
                "probability provided the foundation for the techniques used here."),
              p("I extend sincere thanks to my classmates and friends for reviewing the design, pointing out "
                "ambiguities and helping me debug the trickier parts of the pipeline. Finally, I thank my parents "
                "for their un- wavering support, patience and for everything they have done for me."),
              PageBreak()]

    story += [h1("Abstract")]
    story += [p("Low-cost GNSS receivers can disagree because satellite visibility, geometry, signal quality, "
                "multipath and stale observations vary over time. The project is built around a machine-learned "
                "weighting of the reliability indicators: the decision rule W = a*T + b*S + c*SNR + d*DOP scores "
                "each receiver with four normalized indicators, and a compact neural network learns the "
                "coefficients [a, b, c, d] from the current GNSS conditions, so the same self-explanatory formula "
                "is used everywhere while its weights adapt."),
              p("The hardware pipeline uses two GNSS data streams acquired by an ESP32 with a per-second C/N0 "
                "tracker, a Raspberry Pi decision engine that applies the learned weights with a switching "
                "deadband (hysteresis) and a fixed-weight fallback, and a Flask web view. The offline experiment "
                "conditions the weight network on 16 features derived from both receivers (raw reliability "
                "indicators, comparative deltas and short-term temporal statistics)."),
              p(f"On the 20 held-out simulated test sessions, the learned weights (Model C) reached a mean "
                f"selected error of {TEST['dynamic_ml_weights_model_C']['mean_error_m']:.3f} m, improving to "
                f"{TEST['dynamic_ml_with_hysteresis']['mean_error_m']:.3f} m with a 0.05 deadband and about 43 "
                f"percent fewer source switches, compared with "
                f"{TEST['original_fixed_weights_model_A']['mean_error_m']:.3f} m for the predefined fixed "
                f"coefficients. These figures are results of the controlled simulation, not field or RTK accuracy "
                f"claims."),
              PageBreak()]

    toc = TableOfContents(dotsMinLevel=0, formatter=doc.body_page_label)
    toe = ParagraphStyle("TOCTitle", parent=S["front_head"], alignment=TA_CENTER)
    story += [Paragraph("Table of Contents", toe), toc]
    story += [PageBreak()]

    loftoc = TableOfContents(dotsMinLevel=0, formatter=doc.body_page_label,
                             notifyKind="FigEntry")
    loftoc.levelStyles = [S["toc"]]
    story += [h1("List of Figures"), loftoc, PageBreak()]
    return story


def report_story(doc):
    story = cover_page("Multi GNSS Dynamic Receiver Selection System", "Adaptive reliability weights, live integration and simulation based evaluation", "Project Report")
    story += certificate_page()
    story += front_matter(doc)

    A = TEST["original_fixed_weights_model_A"]
    B = TEST["optimized_fixed_weights_model_B"]
    C = TEST["dynamic_ml_weights_model_C"]
    Cx = TEST["dynamic_ml_with_hysteresis"]

    story += [h1("1. Introduction"),
              p("Global Navigation Satellite Systems estimate a receiver position from radio signals transmitted by satellites. A low-cost GNSS module normally reports latitude, longitude, time, satellite count and quality indicators. In open sky, two receivers may appear similar. In partly blocked or reflective surroundings, their reported positions can differ even when both still have a valid fix. A selection system is useful when it continuously decides which receiver output is more likely to be accurate at the current instant."),
              p("The project's decision rule is a weighted reliability score: each receiver gets a score W = a*T + b*S + c*SNR + d*DOP built from timing accuracy, satellite count, signal-to-noise ratio and HDOP, and the receiver with the larger W is selected. The coefficients [a, b, c, d] are not hand-tuned; they are learned by a compact neural network f(X) that maps the current GNSS conditions to the weights. A predefined fixed set (0.25, 0.25, 0.30, 0.20) was used only during early bring-up so the acquisition and decision pipeline could be verified before the ML component was wired in; in the deployed system it survives solely as an emergency fallback and as a comparison baseline."),
              h2("1.1 Objectives"),
              bullets([
                  "Acquire and format readings, including a 1 s average C/N0, from two GNSS receivers through an ESP32 UART bridge.",
                  "Provide a Raspberry Pi decision engine and Flask path that applies the learned adaptive weights, with hysteresis and a fixed-weight fallback, and exposes a selected location.",
                  "Learn the coefficients of the reliability equation with a compact neural network so the decision rule stays transparent while its weights become context-dependent.",
                  "Evaluate the learned weights against the predefined fixed set and a globally optimized fixed set on held-out sessions, reporting mean, median, tail error, flip accuracy and switching behaviour."]),
              h2("1.2 Scope"),
              p("The repository contains a hardware-oriented prototype whose live path is implemented and a separate offline model experiment that produced the exported weights. The ESP32 firmware, Raspberry Pi engine, web dashboard and the ML training/evaluation code are all in the repository. The evaluation reported here is a controlled simulation study; the model and engine are integrated together, but they have not yet been validated against synchronized real receiver logs and a trusted reference trajectory."), PageBreak(),

              h1("2. GNSS Background"),
              h2("2.1 How a GNSS position is formed"),
              p("Each satellite broadcasts a time-stamped signal and orbital information. The receiver estimates the travel time of the signal and converts it to a pseudorange. With signals from at least four satellites, the receiver solves for three position coordinates and its clock bias. This is often called trilateration in introductory explanations, although the actual receiver solution uses pseudoranges, timing corrections and a least-squares or filter-based estimator."),
              h2("2.2 Why two receivers can disagree"),
              bullets([
                  "<b>Visibility:</b> buildings, trees and the human body can block satellites differently for each antenna.",
                  "<b>Geometry:</b> satellites clustered in one part of the sky produce a weaker geometric configuration than satellites spread widely across the sky.",
                  "<b>Multipath:</b> a reflected signal travels a longer path and can bias the time estimate.",
                  "<b>Receiver and antenna differences:</b> chipsets, antenna placement, update timing and firmware can change tracking quality.",
                  "<b>Staleness:</b> a position can be valid but old; a moving platform needs fresh readings.",
              ]),
              h2("2.3 Meaning of the main indicators"),
              data_table([
                  ["Indicator", "Meaning", "Use in this project", "Important caution"],
                  ["Satellite count", "Number of satellites contributing to or visible for a solution.", "Term S and a conditioning feature.", "More satellites often helps, but quality and geometry still matter."],
                  ["HDOP", "Horizontal Dilution of Precision. Lower is generally better geometry.", "Term DOP and a conditioning feature.", "HDOP is a geometry indicator; it is not the actual position error."],
                  ["C/N0", "Carrier-to-noise density ratio in dB-Hz, a signal-quality measure (1 s average).", "Term SNR and a conditioning feature.", "Per-satellite aggregation is not implemented; one average is used."],
                  ["Time accuracy", "Seconds between the reported time and the live UTC clock.", "Term T and an availability check.", "A fresh timestamp does not guarantee low position error."],
                  ["Position step", "Distance from the prior reported position.", "Conditioning feature and sanity indicator.", "Large step can be motion, noise or a jump; context matters."],
              ], [2.4*cm, 4.0*cm, 3.4*cm, 5.1*cm]),
              caption("Table 1. GNSS indicators used in the reliability score and the adaptive weight network."), PageBreak(),

              h1("3. System Architecture"),
              p("The data path begins with two GNSS modules. The ESP32 reads each NMEA stream through a separate hardware UART, parses it with TinyGPSPlus, tracks the per-satellite C/N0 from GSV sentences and emits one combined line through USB serial at about 1 Hz. The Raspberry Pi decision engine parses the line, evaluates the reliability terms, obtains adaptive weights [a,b,c,d] from the exported model (with a fixed-weight fallback), applies a hysteresis deadband and logs every decision to a JSONL file. A Flask endpoint serves the selected location to the browser dashboard."),
              system_drawing(),
              figcaption("Fig 3.1 The live-integrated pipeline: the exported model drives the Raspberry Pi decision engine."),
              h2("3.1 ESP32 message format"),
              p("The firmware emits a compact, comma-separated representation for each receiver; the two records are separated with a vertical bar. Each record is GPS,time,lat,lon,sat,hdop,snr or GNSS,time,lat,lon,sat,hdop,snr, where snr is the 1 s average C/N0. Latitude and longitude become the string NO_FIX when location data is invalid; satellite count defaults to zero; HDOP defaults to 99.99."),
              Preformatted("GPS,12:10:05,10.051602,76.331617,18,0.82,46|GNSS,12:10:05,10.051610,76.331601,15,1.24,44", S["code"]),
              h2("3.2 Processing components"),
              data_table([
                  ["Component", "Technology", "Responsibility", "Current status"],
                  ["GNSS modules", "NEO-M8N and Quectel L89", "Produce receiver-specific GNSS observations.", "Hardware path described in firmware; no field validation documented."],
                  ["ESP32", "Arduino C++ and TinyGPSPlus", "Read two UART streams, track C/N0, publish a combined serial line.", "Implements time, location, satellite count, HDOP and SNR output."],
                  ["Decision engine", "gnss_engine.py (NumPy inference)", "Parse frames, compute terms, apply adaptive weights with hysteresis and fallback, log decisions.", "Implemented and unit-tested; runs with a built-in simulated frame source."],
                  ["Web view", "Python, Flask and Leaflet", "Serve the selected location and dashboard.", "Implemented; /location returns live decision JSON."],
                  ["Offline experiment", "Python and NumPy", "Generate controlled sessions, train Model C, grid-search Model B, export weights and metrics.", "Implemented with a fixed seed; deterministic and CPU-only."],
              ], [2.5*cm, 3.1*cm, 5.3*cm, 4.0*cm]),
              caption("Table 2. Components and evidence-based implementation status."), PageBreak(),

              h1("4. The Weighted Reliability Score and Its Limitations"),
              p("The decision rule of the project is the weighted reliability equation. Each receiver is scored with four components normalized to 0 to 1:"),

              p("<font name='Courier'>T   = clip(1 - time_error / 5, 0, 1)\nS   = clip(satellites / 20, 0, 1)\nSNR = clip(cn0 / 50, 0, 1)\nDOP = clip(1 / HDOP, 0, 1)\nW = a*T + b*S + c*SNR + d*DOP\nbest_receiver = argmax(W)</font>", "body"),

              h2("4.1 Why a numerical score is not a reliability percentage"),
              p("A value such as 0.8 is only the result of a formula unless it has been calibrated against repeated observed outcomes. It does not mean an 80 percent probability of being correct or an 80 percent chance of being within a distance. The project therefore reports the selection in measurable terms: mean, median and tail position error of the chosen receiver against the generated reference trajectory. The weights themselves are compared as numbers learned from data, not as probabilities."),
              h2("4.2 Specific limitations of the predefined fixed set"),
              data_table([
                  ["Issue", "Effect", "Role in this project"],
                  ["Predefined 25/25/30/20 weights", "No experimental basis for the relative importance of features; weights cannot adapt when the environment changes.", "Model A keeps these as the fallback and comparison baseline; the learned model supplies the weights used in practice."],
                  ["Satellite count capped at 20", "Counts above 20 receive the same satellite term.", "The cap is retained in the normalized term; additional satellites still appear in the 16-dimension conditioning input."],
                  ["HDOP capped at 1", "All HDOP values at or below 1 receive the same geometry term.", "The cap remains in the term; raw HDOP enters the conditioning vector."],
                  ["Satellite count and HDOP overlap", "A hand formula can double-count closely related quality evidence.", "The learned network can weight the terms jointly and nonlinearly, although correlation remains a data issue."],
                  ["Invalid inputs inconsistently handled", "Negative or stale values can create meaningless scores.", "Eligibility rules reject invalid or stale receivers before scoring."],
                  ["Weights cannot react to context", "The relative value of timing versus geometry is assumed constant.", "The adaptive network changes [a,b,c,d] with current conditions while keeping the equation interpretable."],
              ], [3.2*cm, 6.0*cm, 5.8*cm]),
              caption("Table 3. Motivation for learning context-dependent weights rather than relying on a predefined set."), PageBreak(),

              h1("5. Adaptive Weight Learning"),
              p("The heart of the system is an adaptive weight learner. A compact MLP maps a conditioning feature vector X to softmax coefficients [a,b,c,d] = f(X), which the reliability equation then uses to score each receiver. Constraining the output with softmax keeps a,b,c,d non-negative and summing to one, so the weights stay interpretable as a normalized allocation of importance. The learned model is the primary source of the coefficients; the predefined set is used only when the network is unavailable."),
              model_drawing(),
              figcaption("Fig 5.1 Learned-weight architecture: conditioning features produce the coefficients used by the W = a*T + b*S + c*SNR + d*DOP rule."),
              h2("5.1 Conditioning features"),
              p("The network conditions on 16 features built from both receivers at each epoch: the eight raw reliability-relevant values (T-related timing error, satellites, C/N0 and HDOP for receiver 0 and receiver 1), four comparative deltas between the two receivers, and four temporal statistics (per-receiver satellite-count rate and a causal 5-sample C/N0 moving average). All features are observable at decision time. The model respects the project's raw channel order: satellites, HDOP, C/N0, age, step, time error and receiver id."),
              h2("5.2 Network and parameter count"),
              data_table([
                  ["Layer", "Shape", "Activation", "Parameters"],
                  ["Input", "16", "-", "0"],
                  ["Dense 1", "16 to 24", "ReLU", "16 x 24 + 24 = 408"],
                  ["Dense 2", "24 to 24", "ReLU", "24 x 24 + 24 = 600"],
                  ["Output", "24 to 4", "softmax", "24 x 4 + 4 = 100"],
                  ["Total trainable", "-", "-", "1,108"],
                  ["Stored normalization", "mean and std for 16 features", "-", "32 non-trainable values"],
              ], [2.9*cm, 3.4*cm, 3.2*cm, 5.5*cm]),
              caption("Table 4. Model C architecture and parameter accounting."),
              h2("5.3 Why learn weights instead of predicting error directly"),
              p("Predicting each receiver's error and choosing the smaller one is a valid alternative, but it changes the system's behaviour into a per-receiver regression on synthetic labels and requires trusting each network's error scale. Learning weights keeps the decision rule transparent and localized: the examiner can see that the formula is fixed and the network's whole role is to set its four coefficients."),
              h2("5.4 Deployment behaviour"),
              p("At inference the engine computes W for both receivers, restricts selection to receivers that pass the availability checks (valid fix, at least four satellites, HDOP inside range, data younger than 2 s), and applies a deadband: if the absolute difference |W0 - W1| is below the hysteresis threshold, the previously selected receiver is retained to avoid rapid oscillation. If the learned model fails to load or returns abnormal output, the engine falls back to the predefined fixed coefficients (0.25, 0.25, 0.30, 0.20)."), PageBreak(),

              h1("6. Training and Evaluation Design"),
              h2("6.1 Training objective"),
              p("The network is trained with a soft, differentiable selection objective. Define the soft probability P1 = sigmoid(tau * (W1 - W2)); the loss is the expected positioning error E[P1*E1 + P2*E2] using the receivers' actual generated errors. As training proceeds the temperature tau anneals from 2.0 toward 0.5, making the soft selection progressively sharper toward the deterministic argmax used at deployment."),
              h2("6.2 Controlled scenario generation"),
              p("The generator creates 120 independent sessions of 300 seconds each and two receiver readings per second, giving 72,000 receiver readings. Scenarios include open sky, partial obstruction, urban multipath, dropout and mixed conditions. Each session includes a reference trajectory, reported observations, feature values, stale intervals, invalid fixes, shared drift, receiver-specific bias and occasional spikes. These are controlled assumptions used to study the decision logic, not measurements from a deployed GNSS system."),
              h2("6.3 Data split and leakage control"),
              p("Whole sessions are separated into 80 training, 20 validation and 20 test sessions, stratified per scenario, before any fitting. The feature mean and standard deviation are calculated from eligible training readings only, and the test split is untouched until the final evaluation."),
              data_table([
                  ["Partition", "Sessions", "Purpose", "Leakage rule"],
                  ["Training", "80", "Fit network parameters.", "No session appears in validation or test."],
                  ["Validation", "20", "Select the best epoch; grid-search Model B; stop early.", "Whole held-out sessions."],
                  ["Test", "20", "Report final independent results.", "Never used to choose weights or epochs."],
              ], [3.0*cm, 2.3*cm, 5.6*cm, 4.1*cm]),
              caption("Table 5. Session-level partitioning used by the experiment."),
              h2("6.4 Optimisation and baselines"),
              p("Model C is trained with Adam (learning rate 0.001, batch 512, early stopping after 12 epochs without improvement, maximum 60 epochs). The best validation checkpoint was found at epoch 6. Two fixed-weight baselines accompany it: Model A reproduces the predefined fallback coefficients (0.25, 0.25, 0.30, 0.20); Model B is a fixed set found by an exhaustive grid search over the simplex at step 0.05 (1,771 candidates) on validation epochs, which selected alpha 0.10, beta 0.20, gamma 0.00 and delta 0.70."),
              h2("6.5 Metrics"),
              bullets([
                  "<b>Mean, median, RMSE and 95th-percentile selected error:</b> distribution of the horizontal error of the receiver selected at each available epoch.",
                  "<b>Flip accuracy vs oracle:</b> fraction of available epochs at which the selector agrees with the receiver that actually had the lower error (the oracle cannot be known live; it is an upper bound on decision quality).",
                  "<b>Switches and mean duration:</b> number of source changes and average run length; with hysteresis the system trades a small amount of responsiveness for stability.",
                  "<b>Availability:</b> number of epochs with at least one eligible receiver.",
              ]), PageBreak(),

              h1("7. Results"),
              p("The following results compare the predefined fixed set, the optimized fixed set and the learned adaptive model on the 20 held-out test sessions. All selection decisions use the same reliability equation; only the coefficients differ."),
              accuracy_chart(),
              figcaption("Fig 7.1 Held-out comparison. Lower error is better."),
              data_table([
                  ["Selector", "Accuracy vs oracle", "Mean m", "Median m", "RMSE m", "95th m", "Switches"],
                  *[[name, f"{TEST[key]['selection_accuracy_vs_oracle']*100:.1f}%",
                     f"{TEST[key]['mean_error_m']:.3f}", f"{TEST[key]['median_error_m']:.3f}",
                     f"{TEST[key]['rmse_m']:.3f}", f"{TEST[key]['p95_error_m']:.3f}",
                     f"{TEST[key]['switches']:,}"] for name, _, key in MODEL_KEYS],
              ], [4.2*cm, 2.5*cm, 2.0*cm, 2.2*cm, 2.0*cm, 2.0*cm, 2.2*cm]),
              caption("Table 6. Held-out results (5,986 available epochs). The oracle lower bound is not a deployable selector."),
              p(f"The row 'Accuracy vs oracle' counts, at each available epoch, whether the selector picked the "
                f"receiver that actually had the lower error ({Cx['selection_accuracy_vs_oracle']*100:.1f} percent "
                f"for the deployed selector). The confusion matrix below breaks those decisions into full counts, "
                f"with the lower-error receiver as the reference:"),
              confusion_matrix_drawing("dynamic_ml_with_hysteresis"),
              figcaption("Fig 7.2 Confusion matrix of the deployed selector (Model C + hysteresis 0.05) against the "
                         "oracle on the 5,986 available test epochs."),
              data_table([
                  ["Selector", "Oracle R0, sel R0", "Oracle R0, sel R1", "Oracle R1, sel R0", "Oracle R1, sel R1", "Oracle agreement"],
                  *[[name,
                     f"{EVAL['confusion'][key]['matrix'][0][0]:,}",
                     f"{EVAL['confusion'][key]['matrix'][0][1]:,}",
                     f"{EVAL['confusion'][key]['matrix'][1][0]:,}",
                     f"{EVAL['confusion'][key]['matrix'][1][1]:,}",
                     f"{EVAL['confusion'][key]['accuracy_vs_oracle']*100:.1f}%"]
                    for name, _, key in MODEL_KEYS],
              ], [3.4*cm, 2.1*cm, 2.1*cm, 2.1*cm, 2.1*cm, 2.2*cm]),
              caption("Table 7. Selector-vs-oracle confusion counts over the held-out test split. 'Oracle R0' means "
                      "receiver 0 had the lower actual error at that epoch; 'sel R1' means the rule picked receiver 1. "
                      "Diagonal columns are agreement with the oracle."),
              p(f"In absolute terms the deployed selector agrees with the oracle at "
                f"{EVAL['confusion']['dynamic_ml_with_hysteresis']['accuracy_vs_oracle']*100:.1f} percent of epochs. "
                f"It is conservative in a specific way: it agrees on more than half of the epochs where the oracle "
                f"prefers receiver 1, while a fixed baseline struggles with the same set. The off-diagonal imbalance "
                f"also shows why the reliability equation needs a hysteresis deadband: many of the residual disagreements "
                f"occur when the two scores are close, which is exactly the regime hysteresis suppresses."),
              p(f"Model C reduced mean selected error by {(1 - C['mean_error_m'] / A['mean_error_m'])*100:.1f} percent relative to the predefined fixed Model A and by {(1 - C['mean_error_m'] / B['mean_error_m'])*100:.1f} percent relative to the optimized fixed Model B. With the 0.05 hysteresis deadband, mean error fell to {Cx['mean_error_m']:.3f} m, 95th-percentile error to {Cx['p95_error_m']:.3f} m, flip accuracy rose from {A['selection_accuracy_vs_oracle']*100:.1f} percent (A) to {Cx['selection_accuracy_vs_oracle']*100:.1f} percent, and source switches dropped by {(1 - Cx['switches']/A['switches'])*100:.1f} percent compared with Model A."),
              p("The honest reading is that the accuracy gain of dynamic weighting over even the optimized fixed set is small. Most of the practical improvement comes from the hysteresis deadband, which stabilizes the output track rather than dramatically changing accuracy. These figures only support the claim that the adaptive weights learned the patterns built into this controlled study."), PageBreak(),

              h1("8. Results by Scenario and Learned-Weight Analysis"),
              scenario_chart(),
              figcaption("Fig 8.1 Mean selected error by generated condition. Lower error is better."),
              data_table([
                  ["Scenario", "Model A mean m", "Model B mean m", "Model C mean m"],
                  *[[label, f"{SCEN[n]['model_A']['mean_error_m']:.3f}",
                     f"{SCEN[n]['model_B']['mean_error_m']:.3f}",
                     f"{SCEN[n]['model_C']['mean_error_m']:.3f}"] for label, n in zip(SCENARIO_LABELS, SCENARIO_NAMES)],
              ], [4.6*cm, 3.5*cm, 3.5*cm, 3.5*cm]),
              caption("Table 8. Scenario comparison on the held-out test split."),
              p("The largest margin occurs in the urban-multipath condition, where Model C can combine several degraded indicators. The adaptive weights are also interpretable, which is a core claim of the project:"),
              data_table([
                  ["Condition (test, both receivers available)", "alpha (timing)", "beta (sats)", "gamma (SNR)", "delta (DOP)"],
                  *[[f"{label} ({int(LW['by_condition'][name]['n'])} epochs)",
                     f"{LW['by_condition'][name]['weights']['alpha']:.3f}",
                     f"{LW['by_condition'][name]['weights']['beta']:.3f}",
                     f"{LW['by_condition'][name]['weights']['gamma']:.3f}",
                     f"{LW['by_condition'][name]['weights']['delta']:.3f}"]
                    for label, name in [("Open sky", "high_satellite_count"), ("Urban multipath", "low_satellite_count"),
                                        ("Good geometry", "good_geometry"), ("Poor geometry", "poor_geometry")]],
              ], [4.3*cm, 2.7*cm, 2.4*cm, 2.2*cm, 2.4*cm]),
              caption("Table 9. Learned weights by proxy condition, aggregated over test epochs (see metrics.json learned_weights for the full set)."),
              p(f"Overall, the model allocates about {LW['overall']['weights']['delta']*100:.0f} percent of its weight to DOP and {LW['overall']['weights']['alpha']*100:.0f} percent to timing, but in open-sky epochs the DOP weight approaches {LW['by_scenario']['open_sky']['weights']['delta']:.2f} while in urban multipath the timing and satellite weights dominate (alpha + beta ~0.96). The gamma (SNR) term carries little extra discriminative power in this synthetic data. These weight shifts are correlational observations about the generated study, not causal claims."),
              h2("8.1 Feature ablation"),
              data_table([
                  ["Conditioning input", "Mean error m", "Flip accuracy", "Switches"],
                  *[[name, f"{ab['mean_error_m']:.3f}", f"{ab['selection_accuracy']*100:.1f}%",
                     f"{ab['switches']:,}"] for name, ab in EVAL["ablation"].items()],
              ], [4.6*cm, 2.5*cm, 2.7*cm, 2.5*cm]),
              caption("Table 10. Ablation of conditioning feature groups (30-epoch budget, test split). Adding comparative and temporal features progressively improves selection."),
              h2("8.2 Hysteresis sweep"),
              data_table([
                  ["Threshold", "Accuracy vs oracle", "Mean m", "Switches", "Mean run s"],
                  *[[f"{float(t):.2f}", f"{s['selection_accuracy_vs_oracle']*100:.1f}%",
                     f"{s['mean_error_m']:.3f}", f"{s['switches']:,}", f"{s['mean_duration_s']:.2f}"]
                    for t, s in EVAL["hysteresis_sweep"].items()],
              ], [2.7*cm, 3.5*cm, 2.4*cm, 2.4*cm, 2.6*cm]),
              caption("Table 11. Hysteresis sweep for Model C. Threshold 0.05 was chosen for balance: better accuracy and error with about 42 percent fewer switches than no deadband."), PageBreak(),

              h1("9. Implementation Status and Remaining Gaps"),
              p("A defensible project report states both what is implemented and what remains before a field-ready system. The live path is now implemented end to end: firmware with C/N0, a unit-tested decision engine with adaptive weights, hysteresis and logging, and a simulated frame source for hardware-free verification. The offline experiment is complete and deterministic. The remaining work is empirical validation."),
              data_table([
                  ["Area", "Status in current source", "Remaining work"],
                  ["ESP32 firmware", "Two UART parsers, HDOP and 1 s average C/N0 output; new 7-field frame.", "Field check on real receivers; confirm timing synchronization and C/N0 calibration."],
                  ["Decision engine", "gnss_engine.py: parsing, reliability terms, adaptive weights, hysteresis, fixed-weight fallback, JSONL logging; 8 unit tests pass.", "Tune the hysteresis threshold and fallback behaviour from real data."],
                  ["Web service", "server.py runs the engine with --simulate or serial; /location serves decision JSON.", "Serial reconnection and health reporting need field testing."],
                  ["ML experiment", "Deterministic NumPy training (Model A/B/C), grid search, ablation, hysteresis sweep; 9 tests pass.", "Retrain on synchronized real sessions separated by route or time."],
                  ["Validation", "Simulated reference trajectories and generated errors only.", "Collect real reference positions (RTK or surveyed), remove faulty reference intervals, account for antenna offsets."],
              ], [3.0*cm, 6.7*cm, 5.3*cm]),
              caption("Table 12. Implementation status and evidence boundaries.")
              ,
              h2("9.1 Recommended sequence to a field claim"),
              numbered([
                  "Collect synchronized raw messages, parsed features, selected source, weights, fix state and switches into the existing JSONL log.",
                  "Build a synchronized real reference trajectory (RTK or surveyed points) and remove faulty reference intervals.",
                  "Retrain and retest by whole real sessions, separated by route or time period.",
                  "Re-tune the hysteresis threshold and fallback behaviour from the real sessions.",
                  "Report mean, median, tail error, availability and switching by environment, not only overall numbers.",
              ]), PageBreak(),

              h1("10. Conclusion"),
              p("This project is a dynamic receiver-selection system built around a transparent decision rule - W = a*T + b*S + c*SNR + d*DOP with argmax selection - whose weights are learned adaptively by a compact neural network. The offline experiment shows that the network learns interpretable, context-dependent weights: DOP dominates under good conditions, while timing and satellite terms take over under degraded ones. The gains over even an optimized fixed set are modest, and most of the practical stability comes from the hysteresis deadband."),
              p("The live pipeline integrates all pieces: the ESP32 emits C/N0, the Raspberry Pi engine applies learned weights with fallback and hysteresis and logs every decision, and the dashboard visualizes results. The next milestone is not more modelling: it is collecting synchronized real receiver and reference data so that the same pipeline can be retrained and honestly evaluated outside the controlled simulation."),
              h1("References"),
              numbered([
                  "Project source repository: ESP32 firmware, Raspberry Pi decision engine and Flask service, offline model code and exported metrics, inspected September 2026.",
                  "E. D. Kaplan and C. J. Hegarty, Understanding GPS GNSS Principles and Applications, 3rd ed., Artech House, 2017.",
                  "P. Misra and P. Enge, Global Positioning System Signals Measurements and Performance, 2nd ed., Ganga-Jamuna Press, 2011.",
                  "NumPy project documentation and the repository-pinned NumPy " + METRICS["numpy_version"] + " runtime used for the experiment.",
                  "M. Hart, TinyGPSPlus Arduino library documentation, used for the ESP32 parsing interface.",
              ]),
              h1("Appendix A Key Reproducibility Facts"),
              data_table([
                  ["Item", "Recorded value"],
                  ["Training backend", METRICS["backend"] + " (NumPy " + METRICS["numpy_version"] + ")"],
                  ["Random seed", str(METRICS["seed"])],
                  ["Sessions and readings", "120 sessions; 300 seconds each; 72,000 receiver readings"],
                  ["Split", "80 train / 20 validation / 20 held-out test sessions (stratified per scenario)"],
                  ["Decision-relevant training epochs (both available)", str(METRICS["decision_relevant_training_epochs_both_available"])],
                  ["Best validation epoch", "Epoch " + str(METRICS["best_epoch"]) + " of 60 max; patience 12"],
                  ["Temperature schedule", "annealed " + str(METRICS["temperature_schedule"][0]) + " toward " + str(METRICS["temperature_schedule"][-1])],
                  ["Model B search", str(METRICS["grid_search"]["candidates"]) + " candidates; validation mean error " + f"{METRICS['grid_search']['best_validation_mean_error_m']:.4f} m"],
                  ["Hysteresis used", str(METRICS["hysteresis_threshold_used"])],
                  ["Training CSV", "readable rows of the 80 training sessions"],
                  ["Testing CSV", "readable rows of the 20 test sessions"],
              ], [6.0*cm, 9.1*cm]),
              caption("Table A1. Information needed to reproduce the recorded offline experiment."),
              h1("Appendix B Selection Pseudocode"),
              Preformatted("for each epoch, for each receiver i:\n    check eligibility (fix valid, sat >= 4, HDOP in range, age <= 2)\n    T_i   = clip(1 - time_err_i/5,  0, 1)\n    S_i   = clip(sat_i/20,           0, 1)\n    SNR_i = clip(cn0_i/50,           0, 1)\n    DOP_i = clip(1/hdop_i,           0, 1)\n\n[a, b, c, d] = f(X)               # learned, softmax, sum = 1\nW_i = a*T_i + b*S_i + c*SNR_i + d*DOP_i\nif |W0 - W1| < hysteresis: keep the previous receiver\nselected = argmax(W) over available receivers (else none)\npublish selected source, coordinates, weights and reason", S["code"]),
              p("The result is a source-selection decision. The model does not correct a coordinate and does not fuse two positions; it chooses one receiver's reported position."),
    ]
    return story


def qa(title, answer):
    return [h3(title), p(answer, "body_small")]


def viva_story(doc):
    test = TEST
    story = cover_page("Multi GNSS Dynamic Receiver Selection System", "Technical viva preparation guide from fundamentals to implementation", "Viva Guide")
    story += [h1("How to Use This Guide"),
              p("Read Sections 1 to 6 first if you are new to GNSS. They build the mental model needed to answer a teacher without memorising jargon. Sections 7 to 10 connect those concepts to the exact repository and its current limitations. Section 11 contains short answers to likely questions. Do not claim a result that this guide labels as a simulation result."),
              h2("Thirty second project explanation"),
              p("Our project compares two GNSS receiver readings at each update and selects the one with the larger reliability score W = a*T + b*S + c*SNR + d*DOP, where T is timing accuracy, S is satellite count, SNR is signal quality and DOP is inverted HDOP. The coefficients come from a small neural network that learns [a, b, c, d] from the current conditions, so the same transparent formula is always used while its weights adapt. A predefined fixed set is kept only as a bring-up fallback and a comparison baseline. We compare the learned model against the predefined fixed set and an optimized fixed set on held-out simulated sessions, and we integrated the adaptive weights into a Raspberry Pi decision engine with hysteresis and a fixed-weight fallback."),
              h2("One sentence answer for the main contribution"),
              p("We built a receiver-selection system whose reliability-equation coefficients are learned context-dependently by a compact network, and integrated the adaptive weights into the live engine with hysteresis and a safe predefined-weight fallback."),
              PageBreak(),

              h1("1. GNSS From First Principles"),
              h2("1.1 What does GNSS mean"),
              p("GNSS means Global Navigation Satellite System. It is the broad name for satellite navigation constellations such as GPS, Galileo, GLONASS and BeiDou. GPS is one constellation; a GNSS receiver may use more than one constellation."),
              h2("1.2 What is a GNSS receiver actually measuring"),
              p("A satellite broadcasts a known signal with timing information. The receiver compares the expected code timing with the received code timing. Multiplying the measured delay by the speed of light gives an approximate distance called a pseudorange. It is called pseudo because the receiver clock and other sources add error. With at least four satellites, the receiver estimates x, y, z and clock bias together."),
              h2("1.3 Why four satellites"),
              p("A 3D position has three unknown coordinates. The receiver clock offset is a fourth unknown because its clock is not perfectly synchronized with satellite time. Four independent pseudorange equations provide enough information to estimate all four unknowns. More than four satellites allow an overdetermined solution, which can improve robustness."),
              h2("1.4 Main error sources"),
              data_table([
                  ["Source", "Plain-language explanation", "Effect"],
                  ["Satellite clock and orbit", "Broadcast timing and orbital data are not perfect.", "Bias in pseudorange."],
                  ["Ionosphere and troposphere", "Signal speed changes slightly while passing through the atmosphere.", "Distance error that depends on path and conditions."],
                  ["Multipath", "The receiver sees reflected as well as direct signals.", "Can cause metre-level bias and spikes."],
                  ["Obstruction", "Buildings or trees block satellites or weaken signals.", "Fewer usable satellites and poorer geometry."],
                  ["Receiver noise", "Low-cost electronics and antenna limits add tracking noise.", "Jitter in time and position."],
                  ["Reference mismatch", "A test reference may itself be wrong or not aligned to the antenna.", "Can make measured model error misleading."],
              ], [3.0*cm, 7.0*cm, 5.0*cm]),
              caption("Table 1. Sources of GNSS position error."), PageBreak(),

              h1("2. Quality Indicators You Must Understand"),
              h2("2.1 Satellite count"),
              p("Satellite count is a useful clue because more tracked satellites can provide redundancy. It is not a guarantee: ten strong, well spread satellites can be better than twenty weak or clustered ones. The normalized term uses satellites / 20 capped at 1, and the raw count also feeds the conditioning vector."),
              h2("2.2 HDOP"),
              p("HDOP stands for Horizontal Dilution of Precision. It describes how the satellite geometry amplifies measurement errors into horizontal position errors. Smaller HDOP is generally better. HDOP does not include every error source and is not equal to metre error. The score uses 1/HDOP capped at 1; the raw HDOP is also a conditioning feature."),
              h2("2.3 C/N0"),
              p("C/N0 is carrier-to-noise density ratio in dB-Hz. Higher values usually mean the receiver tracks the signal more comfortably above noise. It is more precise terminology than a generic SNR label in GNSS contexts. Our firmware averages the per-satellite C/N0 over one second and sends it in the frame. A production system should decide how to aggregate per-satellite values and log the method."),
              h2("2.4 Time accuracy and position step"),
              p("Time accuracy answers whether a reading is fresh by comparing the reported receiver time with the live UTC clock. A position can be valid but stale, which is dangerous during movement. Position step is the distance between the current and previous reported positions. A large step can be real motion, an outlier or a receiver jump, so it is useful only in combination with the other features."),
              h2("2.5 HDOP and satellite count are related but not identical"),
              p("They often move together because fewer visible satellites can worsen geometry. They are still different: two groups of satellites can have the same count but very different sky distribution and HDOP. A model can receive both, but it needs properly separated data and careful validation so it does not learn a fragile accidental correlation."), PageBreak(),

              h1("3. The Weighted Score and the Predefined Fallback"),
              Preformatted("T   = clip(1 - time_error/5, 0, 1)\nS   = clip(satellites/20,    0, 1)\nSNR = clip(cn0/50,           0, 1)\nDOP = clip(1/HDOP,           0, 1)\n\nW = 0.25*T + 0.25*S + 0.30*SNR + 0.20*DOP   (fallback set)\nselect the receiver with the larger W", S["code"]),
              h2("3.1 Why the predefined set is useful as a fallback"),
              p("The formula uses understandable signals and produces one number per receiver. It is cheap to calculate and easy to explain. A predefined coefficient set is useful during bring-up to verify that acquisition and decision logic work before the learned weights are enabled, and as a safe fallback when the model is unavailable."),
              h2("3.2 Why the predefined set is not the decision method"),
              bullets([
                   "The numbers 0.25, 0.25, 0.30 and 0.20 are convenient starting values, not values learned from labelled position-error observations.",
                   "The weights never change, so the formula cannot react when the GNSS environment changes.",
                   "The terms clip high satellite counts and good HDOP values, losing distinctions in those ranges.",
                   "It can double-count quality evidence when inputs are correlated.",
                   "A score has no direct physical unit. Without calibration, 0.8 does not mean 80 percent reliable.",
                   "Invalid inputs must be explicitly rejected; otherwise a missing value may accidentally look favourable.",
              ]),
              h2("3.3 Difference between predefined and learned weights"),
              data_table([
                   ["Question", "Predefined fallback", "Learned weights"],
                   ["Where do weights come from", "Human choice.", "Optimisation on labelled training examples (or grid search for Model B)."],
                   ["Can weights change with context", "No.", "Yes: [a,b,c,d] = f(X) changes with conditions."],
                   ["What is fixed", "The formula and the coefficients.", "The formula; the coefficients change."],
                   ["What can go wrong", "Poor fixed formula or thresholds.", "Bad or unrepresentative data, leakage, overfitting, or unsafe outputs without a fallback."],
              ], [4.0*cm, 5.6*cm, 5.4*cm]), PageBreak(),

              h1("4. The Adaptive Weight Model"),
              model_drawing(),
              figcaption("Fig 4.1 Learned weights: conditioning features produce softmax coefficients for the reliability equation."),
              h2("4.1 Exact architecture and parameter count"),
              Preformatted("Input X: 16 conditioning features\nNormalize: z = (x - training_mean) / training_std\nHidden 1: ReLU(z @ W1 + b1)      16 -> 24\nHidden 2: ReLU(h1 @ W2 + b2)     24 -> 24\nOutput:   softmax(h2 @ W3 + b3)  24 -> 4  [a, b, c, d]\n\nW1: 16 x 24 = 384      b1: 24\nW2: 24 x 24 = 576      b2: 24\nW3: 24 x 4 = 96        b3: 4\nTotal trainable parameters = 1,108", S["code"]),
              h2("4.2 What does ReLU do"),
              p("ReLU means Rectified Linear Unit. It applies max(0, value). Negative hidden activations become zero while positive activations pass through. Stacking ReLU layers lets a small network learn nonlinear patterns, such as a case where HDOP matters differently when C/N0 is weak and a reading is stale."),
              h2("4.3 Why softmax and why sum to one"),
              p("Softmax converts the four output logits into non-negative values that sum to one. This makes [a,b,c,d] interpretable as a normalized allocation of importance across timing, satellites, signal and geometry. It is a modelling choice, not an empirical law; the network could in principle learn weights with different constraints."),
              h2("4.4 Why normalize inputs"),
              p("Features have very different scales. Without normalization, large-scale inputs can dominate optimisation just because of their units. The model stores the training mean and standard deviation and applies the same transformation at inference."),
              h2("4.5 What the model does not do"),
              bullets([
                  "It does not calculate a new latitude or longitude.",
                  "It does not fuse both receiver positions.",
                  "It does not output a calibrated probability of reliability.",
                  "It does not guarantee a maximum error.",
                  "It never replaces the decision rule; it only provides the coefficients.",
              ]), PageBreak(),

              h1("5. Data, Labels and Training"),
              h2("5.1 What is one training example"),
              p("One training example is one epoch (one time step) where both receivers are available. Its input is the 16-dimension conditioning vector assembled from both receivers. Its targets for the loss are the two horizontal errors, in metres, between each receiver's reported position and the reference trajectory."),
              h2("5.2 Why the reference position is not an input"),
              p("If the reference position were given to the model as an input, it would leak the answer. In a real system, the reference position is not available while selecting a receiver. It is used only during training and evaluation to compute the errors in the loss."),
              h2("5.3 The training objective"),
              p(f"The network outputs weights, faces the reliability scores W1 and W2, and is trained so that soft selection expected error is small: loss = P1*E1 + P2*E2 with P1 = sigmoid(tau*(W1-W2)). The temperature tau anneals from {METRICS['temperature_schedule'][0]:.2f} to {METRICS['temperature_schedule'][-1]:.2f} so training starts smooth and ends close to the hard argmax deployed at runtime."),
              h2("5.4 Controlled data design"),
              p("The generator produces five conditions: open sky, partial obstruction, urban multipath, dropout and mixed. It includes quality changes, reported position noise, persistent bias, shared drift, occasional large errors, stale output intervals and invalid fixes. The goal is to test the pipeline under varied controlled cases. It is not a substitute for collecting real receiver logs with a surveyed reference."),
              h2("5.5 Split, loss and early stopping"),
              data_table([
                  ["Term", "Meaning in this project"],
                  ["Train split", "80 full sessions used to adjust network parameters."],
                  ["Validation split", "20 separate sessions used to choose the best epoch, grid-search Model B and stop early."],
                  ["Test split", "20 separate sessions used once for final reported metrics."],
                  ["Loss", "Expected positioning error under soft selection, with temperature annealing."],
                  ["Optimizer", "Adam, learning rate 0.001, batch 512, patience 12, max 60 epochs."],
                  ["Best checkpoint", f"Epoch {METRICS['best_epoch']} based on lowest validation expected error."],
              ], [4.0*cm, 11.0*cm]), PageBreak(),

              h1("6. How Inference Chooses a Receiver"),
              Preformatted("for each receiver: check eligibility, compute T, S, SNR, DOP\n[a, b, c, d] = f(X)             # learned or fixed fallback\nW_i = a*T_i + b*S_i + c*SNR_i + d*DOP_i\n\nif |W0 - W1| < hysteresis threshold:\n    keep the previously selected receiver\nelse:\n    select argmax(W) among available receivers (else none)", S["code"]),
              h2("6.1 Eligibility rules"),
              data_table([
                  ["Input", "Accepted condition", "Reason"],
                  ["Fix", "Valid latitude and longitude", "A receiver without a fix cannot provide a position."],
                  ["Satellite count", "At least 4", "A normal 3D fix needs enough satellites for position plus clock bias."],
                  ["HDOP", "In (0, 99.99)", "Zero, negative or the invalid default is rejected."],
                  ["Age", "0 to 2 seconds", "Old samples should not win a real-time decision."],
                  ["C/N0", "Positive (when present)", "For ML use both receivers must report a positive 1 s average."],
              ], [3.2*cm, 4.3*cm, 7.5*cm]),
              h2("6.2 Hysteresis and why it matters"),
              p("Suppose receiver A scores 0.70 and receiver B scores 0.69 at one second, then their order reverses the next second. Immediate switching can oscillate, changing the selected source several times per minute. Hysteresis keeps the previous selection unless the difference crosses a threshold. In this study threshold 0.05 was the best balance, cutting switches by about 42 percent with unchanged or slightly better accuracy. The correct threshold must be confirmed on real data."),
              h2("6.3 Fallback behaviour"),
              p("If the model fails to load or returns abnormal weights, the engine falls back to the predefined fixed coefficients 0.25, 0.25, 0.30, 0.20 (Model A). The fallback keeps the live service functional and is also used for the 7-field frames that lack a C/N0 value."), PageBreak(),

              h1("7. Results You Can Explain Precisely"),
              accuracy_chart(),
              figcaption("Fig 7.1 Overall held-out result from the controlled experiment."),
              data_table([
                  ["Selector", "Accuracy vs oracle", "Mean m", "Median m", "95th m", "Switches"],
                  *[[name, f"{test[key]['selection_accuracy_vs_oracle']*100:.1f}%",
                     f"{test[key]['mean_error_m']:.3f}", f"{test[key]['median_error_m']:.3f}",
                     f"{test[key]['p95_error_m']:.3f}", f"{test[key]['switches']:,}"]
                    for name, _, key in MODEL_KEYS],
              ], [4.6*cm, 2.3*cm, 2.0*cm, 2.2*cm, 2.2*cm, 2.0*cm]),
              h2("7.1 A correct interpretation"),
              p(f"On the held-out sessions, Model C achieved {test['dynamic_ml_weights_model_C']['mean_error_m']:.3f} m mean selected error versus {test['original_fixed_weights_model_A']['mean_error_m']:.3f} m for the predefined fixed fallback and {test['optimized_fixed_weights_model_B']['mean_error_m']:.3f} m for the optimized fixed set. With the 0.05 deadband the mean falls to {test['dynamic_ml_with_hysteresis']['mean_error_m']:.3f} m. The gains are real but modest; the bigger practical win is the switching reduction. It does not prove that the model will achieve these values on a road, farm or city street."),
              h2("7.2 Learned weights are interpretable"),
              p(f"Overall the model allocates about {LW['overall']['weights']['delta']*100:.0f} percent to DOP, but in open-sky epochs the DOP weight reaches {LW['by_scenario']['open_sky']['weights']['delta']:.2f} while in urban multipath the timing and satellite weights dominate (alpha + beta ~0.96). This is the key evidence that the network learned context-dependent weighting rather than a single fixed allocation."),
              h2("7.3 The flip-oracle is not a competitor"),
              p("The oracle compares the actual reference errors after the fact and represents the maximum agreement any selector could achieve. It is shown only to calibrate how much room for improvement remains in choosing between these two outputs. It must never be described as an implementable method."), PageBreak(),

              h1("8. Exact Code Flow and Current Limits"),
              h2("8.1 ESP32 firmware"),
              p("The sketch creates two TinyGPSPlus parsers and two ESP32 hardware serial ports (UART1 RX1 18 / TX1 5, UART2 RX2 26 / TX2 25). Each loop feeds bytes into the parsers and a small C/N0 tracker that consumes GSV sentences. When a location updates, it emits GPS or GNSS records with time, latitude, longitude, satellite count, HDOP and the 1 s average C/N0."),
              h2("8.2 Raspberry Pi decision engine and server"),
              p("gnss_engine.py parses the frame, builds receiver records with the raw channels (satellites, HDOP, C/N0, age, step, time error, receiver id), computes the reliability terms and the 16-dimension conditioning vector, obtains weights from DynamicWeights with a fixed-weight fallback, applies the hysteresis deadband and logs every decision to decisions.jsonl. server.py runs the engine in a background thread and serves /location, with a --simulate mode that feeds synthetic frames when no hardware is attached."),
              h2("8.3 Offline model files"),
              data_table([
                  ["File", "Purpose"],
                  ["ml/generate.py", "Creates controlled sessions, reference trajectory, observations, features and errors."],
                  ["ml/features.py", "Builds the 16-dimension conditioning features and the reliability terms."],
                  ["ml/baselines.py", "Model A fixed weights and the Model B grid search over the simplex."],
                  ["ml/train.py", "Splits sessions, trains Model C, exports weights and metrics."],
                  ["ml/predict.py", "Portable NumPy inference: DynamicWeights, hysteresis and selection."],
                  ["ml/metrics.py", "Evaluation, switching statistics and hysteresis application."],
                  ["ml/evaluate.py", "Reproduces the full table, hysteresis sweep and ablation."],
                  ["ml/artifacts/model.npz", "Saved weights, biases and 32 normalization values."],
                  ["ml/tests/test_pipeline.py", "Checks determinism, features, training, inference and session isolation."],
                  ["raspberrypi/gnss_engine.py", "Live engine that loads model.npz and applies weights with fallback."],
                  ["raspberrypi/tests/test_engine.py", "Live engine tests: parsing, fallback, hysteresis, logging."],
              ], [5.1*cm, 10.1*cm]),
              h2("8.4 Hard question: is this live machine learning on the Raspberry Pi"),
              p("Yes, in prototype form. The exported model.npz is loaded at runtime by gnss_engine.py and the engine is unit-tested. The remaining honesty point is that 'live' here means the software pipeline is connected end to end; it has not been validated against real receiver logs with a trusted reference. Real-field validation and retraining are the next step."), PageBreak(),

              h1("9. Practical Defence Checklist"),
              h2("9.1 Statements you can safely make"),
              bullets([
                  "The architecture supports two receiver inputs, embedded acquisition with C/N0, and a live decision engine.",
                  "The reliability equation W = a*T + b*S + c*SNR + d*DOP is the fixed decision rule; the coefficients are learned and context-dependent.",
                  "The model is compact: 1,108 trainable parameters and 32 stored normalization values.",
                  "Invalid, stale or unsupported inputs are rejected by explicit eligibility rules.",
                  "Whole sessions were kept separate between training, validation and testing.",
                  "The learned weights are interpretable and shift with GNSS conditions (DOP-dominated in open sky, timing and satellite-dominated in urban multipath).",
                  "The live engine is integrated with hysteresis and a fixed-weight fallback and is covered by unit tests.",
                  "In the recorded controlled study the learned selector improved over the fixed baselines, and hysteresis cut switching by about 43 percent.",
              ]),
              h2("9.2 Statements you must not make"),
              bullets([
                  "Do not say a score of 0.8 proves 80 percent reliability.",
                  "Do not present the recorded metres as RTK, surveyed-point or field-test accuracy.",
                  "Do not say the engine has been validated with real receiver logs.",
                  "Do not say the model corrects GNSS coordinates or fuses two positions.",
                  "Do not call the flip-oracle a deployable selector.",
                  "Do not claim the learned weights are causal; they are aggregated observations from the controlled study.",
              ]),
              h2("9.3 A good answer when challenged about data"),
              p("This stage is a controlled model-development experiment. It verifies the training pipeline, session split, invalid-input handling, model export, fallback behaviour and comparison against fixed baselines. For deployment-grade claims, we would collect synchronized readings and a trusted reference trajectory, retrain with held-out real routes, and report error, availability and switch behaviour by environment."),
              h2("9.4 Suggested demonstration sequence"),
              numbered([
                  "Show the architecture diagram and explain the flow from two receivers through ESP32 and Raspberry Pi.",
                  "Show one combined serial-line example and identify each field including the C/N0.",
                  "Show the reliability equation and the predefined Model A fallback coefficients.",
                  "Show the adaptive model: 16 features, 1,108 parameters, softmax weights feeding the equation.",
                  "Show the train, validation and test session split, then the held-out chart.",
                  "Show the eligibility checks, the hysteresis deadband and the fixed-weight fallback.",
                  "Run server.py --simulate and show /location changing as the simulated conditions change, plus the JSONL log.",
                  "End with the field-validation plan and the evidence boundary.",
              ]), PageBreak(),

              h1("10. Possible Viva Questions and Answers"),
    ]

    qas = [
        ("1. What problem does your project solve?", "It selects the more reliable output from two GNSS receivers at each update. The aim is to reduce selected-position error when receiver quality changes due to geometry, blockage, multipath, signal strength or stale data."),
        ("2. Why use two receivers instead of one?", "A single low-cost receiver can degrade unexpectedly. Two receivers provide alternative observations. The system does not create an ideal position from nothing; it chooses the receiver that looks more reliable at that moment."),
        ("3. What is GNSS and how is it different from GPS?", "GNSS is the general term for satellite navigation systems. GPS is one constellation. A GNSS receiver may also use Galileo, GLONASS or BeiDou depending on its hardware and configuration."),
        ("4. Why do you need at least four satellites?", "The receiver must solve three position coordinates and its own clock bias. Four unknowns require at least four independent pseudorange observations."),
        ("5. What is HDOP?", "Horizontal Dilution of Precision measures the effect of satellite geometry on horizontal position uncertainty. Lower is generally better. It is not the same thing as actual horizontal error."),
        ("6. What is C/N0?", "Carrier-to-noise density ratio in dB-Hz. It describes how strong and trackable a satellite signal is relative to noise. Our ESP32 averages the per-satellite GSV values over one second into the snr field."),
        ("7. Why can a receiver with more satellites still be worse?", "Satellite count does not tell us geometry, multipath level, signal integrity or receiver-specific bias. A larger count can still be poor if satellites are clustered or reflected signals dominate."),
        ("8. What is multipath?", "Multipath occurs when the receiver receives reflected copies of a satellite signal in addition to the direct path. The reflected path is longer, so the timing estimate can be biased and the position can jump."),
        ("9. State the reliability equation.", "W = a*T + b*S + c*SNR + d*DOP, where T is timing accuracy, S is the satellite-count term, SNR is the signal-quality term and DOP is inverted HDOP. We select the receiver with larger W."),
        ("10. What does each term in the equation mean?", "T = clip(1 - time_err/5, 0, 1), S = clip(sat/20, 0, 1), SNR = clip(cn0/50, 0, 1), DOP = clip(1/hdop, 0, 1). Each is normalized to 0 to 1."),
        ("11. What are the predefined fallback weights?", "a=0.25, b=0.25, c=0.30, d=0.20. This is Model A; the live engine keeps them as an emergency fallback and they serve as a comparison baseline in the study."),
        ("12. Why are predefined weights a weak primary method?", "They were selected by design choice rather than fitted to labelled receiver errors. There was no experiment showing signal quality deserves exactly 30 percent and geometry exactly 20 percent, and they never change with conditions."),
        ("13. Why is capping satellite count or HDOP a concern?", "It treats all counts above 20 (or all HDOP values below 1) as equal for that term. The system loses information that may still matter."),
        ("14. What is Model B?", "A fixed coefficient set found by grid search over the simplex at step 0.05 - 1,771 candidates evaluated on validation epochs. It selected a=0.10, b=0.20, c=0.00, d=0.70."),
        ("15. What does the learned model actually output?", "It outputs the four weights [a, b, c, d]. It does not predict a position or an error directly; the coefficients are plugged into the same reliability equation as before."),
        ("16. Why softmax the output?", "Softmax forces the four coefficients to be non-negative and sum to one, so they read as a normalized allocation of importance. It also guarantees bounded weights at inference."),
        ("17. What are the model inputs?", "A 16-dimension conditioning vector: for both receivers the four reliability-relevant values (timing error, satellites, C/N0, HDOP), plus four comparative deltas and four temporal statistics (satellite rate and a 5-sample C/N0 moving average per receiver)."),
        ("18. What is normalization?", "For each feature, subtract the training mean and divide by the training standard deviation. It puts different units on comparable scales for stable optimisation."),
        ("19. How many parameters does the model have?", "1,108 trainable: 408 in the first dense layer (16 to 24), 600 in the second (24 to 24), 100 in the output layer (24 to 4). It also stores 32 non-trainable normalization values."),
        ("20. Why did you choose such a small network?", "The input is only 16 values and the system targets lightweight inference on the Raspberry Pi. A small model is easier to debug, inspect and validate."),
        ("21. What is the training loss?", "The expected positioning error under soft selection: loss = P1*E1 + P2*E2, where P1 = sigmoid(tau*(W1-W2)) and E1, E2 are the actual generated errors of the two receivers."),
        ("22. What is the temperature tau?", "A sharpness parameter: low tau makes selection soft and differentiable; high tau approaches the hard argmax. It is annealed from about 1.98 to 0.55 across training so the final behaviour matches deployment."),
        ("23. What is data leakage?", "Leakage happens when information from the evaluation set influences training or model selection. We prevent a common time-series form by separating whole sessions, not random neighbouring rows."),
        ("24. Why split by session rather than random rows?", "Adjacent seconds in one journey share conditions and are highly correlated. Random row splitting can put almost identical points into train and test."),
        ("25. What is validation data used for?", "Validation data selects the training settings, the best epoch and the Model B grid search. The final results come from separately held-out test sessions."),
        ("26. What is early stopping?", "It stops training when validation performance no longer improves for a chosen number of epochs (patience 12 here). The best checkpoint was epoch 6."),
        ("27. What is flip accuracy vs the oracle?", "The fraction of available epochs where the selector picks the receiver that actually had the lower error. The oracle is computed after the fact and is an unattainable upper bound, used only to calibrate."),
        ("28. What are the held-out results?", f"Model A mean error {test['original_fixed_weights_model_A']['mean_error_m']:.3f} m at {test['original_fixed_weights_model_A']['selection_accuracy_vs_oracle']*100:.1f}% agreement; Model B {test['optimized_fixed_weights_model_B']['mean_error_m']:.3f} m; Model C {test['dynamic_ml_weights_model_C']['mean_error_m']:.3f} m; Model C with 0.05 hysteresis {test['dynamic_ml_with_hysteresis']['mean_error_m']:.3f} m at {test['dynamic_ml_with_hysteresis']['selection_accuracy_vs_oracle']*100:.1f}%."),
        ("29. How large is the improvement?", "Real but modest: Model C is about two percent better in mean error than Model A in this controlled study. The larger practical gain is stability - the hysteresis deadband cuts switches by about 43 percent."),
        ("30. Can you claim this accuracy in the real world?", "No. The recorded figures are from a controlled simulation study. Real deployment requires synchronized real receiver logs and a trusted reference system, then retraining and held-out field evaluation."),
        ("31. What happens if an input is negative or missing?", "The eligibility checks reject invalid or stale receivers, so they cannot win selection. If neither receiver passes, the engine reports no selection rather than a fabricated location."),
        ("32. What happens when there is no GNSS fix?", "That receiver is marked unavailable. The engine uses the other eligible receiver if available, otherwise it reports no selection."),
        ("33. Why do you need hysteresis?", "Without it, very close scores can cause the selected source to flip every update, producing a noisy output track and more position jumps. Hysteresis keeps the previous selection until the difference crosses a threshold."),
        ("34. What hysteresis threshold is used live?", "0.05 by default, matching the best value in the simulation sweep. It is configurable and should be re-tuned on real data."),
        ("35. Why is there a fixed-weight fallback?", "So the live system never depends on a single model file. If the model fails to load or returns abnormal weights, the engine uses the predefined fixed coefficients and keeps working."),
        ("36. Does your system fuse the positions?", "No. It is a selector: it picks one receiver position. Sensor fusion would combine measurements using another estimator such as a Kalman filter, which is future work."),
        ("37. What is the role of ESP32?", "The ESP32 acquires two GNSS UART streams, parses them with TinyGPSPlus, tracks C/N0 from GSV sentences and sends a compact combined message over serial."),
        ("38. What serial format does the ESP32 use?", "Two receiver records separated by a vertical bar: GPS,time,lat,lon,sat,hdop,snr | GNSS,time,lat,lon,sat,hdop,snr. Latitude/longitude are NO_FIX when there is no fix."),
        ("39. Is the model connected to the live server?", "Yes. gnss_engine.py loads model.npz at runtime, applies the weights with hysteresis and fallback, and logs every decision. The simulated frame mode lets you demo it without hardware."),
        ("40. How was the engine tested?", "With unit tests (raspberrypi/tests/test_engine.py) covering frame parsing, availability, ML versus fallback selection, the hysteresis deadband and JSONL logging, plus an end-to-end simulated run against /location."),
        ("41. Which features are computed live?", "The engine computes satellites, HDOP, C/N0, age, step and time error per receiver from the frame and its own state, forming the same 16-dimension conditioning vector used in training."),
        ("42. Why not just use the receiver's own accuracy field?", "Receiver-reported accuracy can be useful, but its availability and calibration depend on the chipset. The project explores selection using a feature vector that can be logged, tested and extended."),
        ("43. What is RTK and why is it useful?", "Real-Time Kinematic positioning uses carrier-phase corrections from a base or network reference to obtain much more accurate relative positions under suitable conditions. It can provide a strong reference for evaluating ordinary GNSS receivers, but it must itself be checked for solution quality and antenna alignment."),
        ("44. What is the difference between a surveyed stationary point and RTK reference?", "A surveyed stationary point is a known fixed coordinate used while the antenna stays still. RTK provides a moving reference trajectory, useful for dynamic testing. Both need correct antenna reference point handling."),
        ("45. What real data would you collect next?", "Synchronized raw observations and reported features from both devices, a trustworthy timestamp, reference coordinates, antenna offsets, route labels, fix status and quality flags; the existing JSONL log already records decisions and weights."),
        ("46. How would you evaluate a real deployment?", "Split by complete real routes or days, train on some environments and test on unseen ones. Report mean, median, 95th percentile and worst tail behaviour, availability, switch rate and performance by environment."),
        ("47. How do you avoid overfitting?", "Session-level held-out validation and test partitions, a small network, early stopping, and finally real routes that were not used for model development."),
        ("48. Why not use a larger deep network?", "More parameters are not automatically better. The conditioning input is modest and the deployment needs simple inference. Architecture choice should be justified by held-out real-data comparisons."),
        ("49. What are limitations of the present study?", "All results come from controlled generated trajectories, not field data. C/N0 is a single average without per-satellite aggregation; the same softmax-constrained weight form is a modelling choice; and the engine has not been validated against real receiver logs with a trusted reference."),
        ("50. What is your future work?", "Collect synchronized reference-labelled field data, retrain by real receiver and environment, re-tune hysteresis, extend per-satellite C/N0 telemetry, and consider calibrated uncertainty or sensor fusion as separate extensions."),
    ]
    for title, answer in qas:
        story += qa(title, answer)
    story += [PageBreak(), h1("11 Last Minute Revision Sheet"),
              data_table([
                  ["Fact", "Answer"],
                  ["Project goal", "Choose the receiver with the larger reliability score and learn the score's weights adaptively."],
                  ["Decision rule", "W = a*T + b*S + c*SNR + d*DOP; argmax(W) among available receivers."],
                  ["Predefined fallback", "Predefined coefficients (0.25/0.25/0.30/0.20) that never adapt; used only as a fallback and baseline."],
                  ["Conditioning input", "16 features from both receivers: reliability indicators, deltas and temporal statistics."],
                  ["Architecture", "16 -> 24 ReLU -> 24 ReLU -> 4 softmax [a,b,c,d]."],
                  ["Trainable parameters", "1,108"],
                  ["Loss", "Expected error under soft selection with temperature annealing."],
                  ["Data split", "80 train / 20 validation / 20 test whole sessions (stratified per scenario)."],
                  ["Best epoch", str(METRICS["best_epoch"])],
                  ["Test results", f"A {test['original_fixed_weights_model_A']['mean_error_m']:.3f} m, B {test['optimized_fixed_weights_model_B']['mean_error_m']:.3f} m, C {test['dynamic_ml_weights_model_C']['mean_error_m']:.3f} m, C+hyst {test['dynamic_ml_with_hysteresis']['mean_error_m']:.3f} m."],
                  ["Learned behaviour", "DOP dominates in open sky; timing and satellite terms dominate in urban multipath."],
                  ["Live status", "Engine integrated and tested; field validation with real reference data remains."],
                  ["Main evidence boundary", "Controlled simulation result, not RTK or field accuracy."],
              ], [5.1*cm, 10.1*cm]),
              h2("Remember"),
              p("A strong viva answer gives the mechanism, the reason for the design decision, and the boundary of the evidence. For example: HDOP measures geometry, so we keep it in the DOP term; the learned network re-weights the terms because conditions change; the model is integrated and tested, but only real labelled field data can establish performance for a real deployment."),
              h1("Glossary"),
              data_table([
                  ["Term", "Meaning"],
                  ["GNSS", "Global Navigation Satellite System, a general term for satellite navigation constellations."],
                  ["Pseudorange", "Distance-like measurement from signal travel time that includes clock and other errors."],
                  ["HDOP", "Horizontal Dilution of Precision, a satellite-geometry indicator."],
                  ["C/N0", "Carrier-to-noise density ratio in dB-Hz, a GNSS signal-quality indicator."],
                  ["Multipath", "Reflected satellite signals that add timing and position bias."],
                  ["RTK", "Real-Time Kinematic carrier-phase correction technique for high-accuracy positioning."],
                  ["Feature", "Input value available to the model at decision time."],
                  ["Hysteresis", "A rule that prevents rapid switching when two scores are nearly equal."],
                  ["Softmax", "A transform that maps values to non-negative numbers summing to one."],
              ], [4.0*cm, 11.2*cm]),
    ]
    return story


def build(filename: Path, label: str, story_fn):
    doc = AcademicDoc(str(filename), label=label, pagesize=A4,
                      title=label, author="Sanal Sivakumar")
    doc.multiBuild(story_fn(doc))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    build(OUT / "GNSS_Dynamic_Receiver_Selection_Project_Report.pdf",
          "Multi GNSS Dynamic Receiver Selection System Project Report", report_story)
    build(OUT / "GNSS_Dynamic_Receiver_Selection_Viva_Guide.pdf",
          "Multi GNSS Dynamic Receiver Selection System Viva Guide", viva_story)
    print("Created", OUT / "GNSS_Dynamic_Receiver_Selection_Project_Report.pdf")
    print("Created", OUT / "GNSS_Dynamic_Receiver_Selection_Viva_Guide.pdf")


if __name__ == "__main__":
    main()