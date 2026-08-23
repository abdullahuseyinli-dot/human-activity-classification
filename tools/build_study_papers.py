"""Render a scholarly PDF from a single Markdown source file.

The renderer intentionally supports a focused, predictable Markdown subset used by
the study reports in this repository. Document prose, figures, tables, citations,
and metadata remain in the Markdown file; this module contains presentation logic
only.
"""

from __future__ import annotations

import argparse
import html
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    LongTable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "POLAR_COMPREHENSIVE_WORKING_PAPER.md"

PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = 22 * mm
RIGHT_MARGIN = 22 * mm
TOP_MARGIN = 22 * mm
BOTTOM_MARGIN = 19 * mm
BODY_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

INK = colors.HexColor("#172033")
NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1D5D8F")
TEAL = colors.HexColor("#177E89")
SLATE = colors.HexColor("#526171")
MID = colors.HexColor("#C7D2DE")
PALE = colors.HexColor("#F2F6F9")
PALE_BLUE = colors.HexColor("#EAF2F8")
WHITE = colors.white

HEADING_LEVELS = {
    "StudyHeading1": 0,
    "StudyHeading2": 1,
    "StudyHeading3": 2,
    "StudyHeading4": 3,
}


@dataclass(frozen=True)
class ParsedMarkdown:
    metadata: dict[str, str]
    body_lines: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Markdown source (default: docs/POLAR_COMPREHENSIVE_WORKING_PAPER.md)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PDF (default: output/pdf/<input-stem>.pdf)",
    )
    parser.add_argument("--title", help="Override the front-matter title")
    parser.add_argument("--author", help="Override the front-matter author")
    parser.add_argument("--date", help="Override the front-matter date")
    parser.add_argument("--status", help="Override the front-matter status")
    parser.add_argument(
        "--no-cover",
        action="store_true",
        help="Start directly with the Markdown body instead of a cover page",
    )
    return parser.parse_args()


def _read_front_matter(path: Path) -> ParsedMarkdown:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return ParsedMarkdown(metadata={}, body_lines=lines)

    metadata: dict[str, str] = {}
    closing_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(
                f"Invalid front matter in {path} at line {index + 1}: expected key: value"
            )
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower().replace("-", "_")
        normalized_value = value.strip()
        if len(normalized_value) >= 2 and normalized_value[0] == normalized_value[-1]:
            if normalized_value[0] in {'"', "'"}:
                normalized_value = normalized_value[1:-1]
        metadata[normalized_key] = normalized_value

    if closing_index is None:
        raise ValueError(f"Unclosed front matter in {path}")
    return ParsedMarkdown(metadata=metadata, body_lines=lines[closing_index + 1 :])


def _first_heading(lines: list[str]) -> str | None:
    for line in lines:
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return None


def _slug(text: str) -> str:
    plain = re.sub(r"[*_`]", "", text).lower()
    plain = re.sub(r"[^a-z0-9]+", "-", plain).strip("-")
    return plain or "section"


def _inline_markup(text: str) -> str:
    """Translate safe inline Markdown to the subset accepted by Paragraph."""

    escaped = html.escape(text.strip(), quote=True)
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda match: (
            f'<link href="{match.group(2).strip()}" color="{BLUE.hexval()}">'
            f"{match.group(1)}</link>"
        ),
        escaped,
    )
    escaped = re.sub(
        r"&lt;(https?://.+?)&gt;",
        lambda match: (
            f'<link href="{match.group(1)}" color="{BLUE.hexval()}">'
            f"{match.group(1)}</link>"
        ),
        escaped,
    )
    escaped = re.sub(
        r"`([^`]+)`",
        r'<font name="Courier" color="#31475B">\1</font>',
        escaped,
    )
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "StudyBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.35,
            leading=13.25,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=7.5,
            allowWidows=0,
            allowOrphans=0,
        ),
        "lead": ParagraphStyle(
            "StudyLead",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=14.6,
            textColor=SLATE,
            spaceAfter=10,
        ),
        "small": ParagraphStyle(
            "StudySmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.4,
            textColor=SLATE,
        ),
        "caption": ParagraphStyle(
            "StudyCaption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.8,
            leading=10.5,
            textColor=SLATE,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=9,
        ),
        "quote": ParagraphStyle(
            "StudyQuote",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=13,
            textColor=NAVY,
            leftIndent=9,
            rightIndent=6,
            spaceBefore=3,
            spaceAfter=3,
        ),
        "list": ParagraphStyle(
            "StudyList",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.8,
            textColor=INK,
            spaceAfter=2.5,
        ),
        "code": ParagraphStyle(
            "StudyCode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.4,
            leading=10,
            textColor=NAVY,
            backColor=PALE,
            borderColor=MID,
            borderWidth=0.4,
            borderPadding=7,
            spaceBefore=4,
            spaceAfter=9,
        ),
        "h1": ParagraphStyle(
            "StudyHeading1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17.2,
            leading=21,
            textColor=NAVY,
            spaceBefore=9,
            spaceAfter=9,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "StudyHeading2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.6,
            leading=15.5,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=5.5,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "StudyHeading3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.4,
            leading=13.3,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "h4": ParagraphStyle(
            "StudyHeading4",
            parent=base["Heading4"],
            fontName="Helvetica-BoldOblique",
            fontSize=9.3,
            leading=12.2,
            textColor=SLATE,
            spaceBefore=6,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "cover_kicker": ParagraphStyle(
            "CoverKicker",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=10,
            textColor=TEAL,
            tracking=1.4,
            spaceAfter=11,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=31.5,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=13,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=13.4,
            leading=18.5,
            textColor=SLATE,
            spaceAfter=18,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12.3,
            textColor=INK,
        ),
        "toc_title": ParagraphStyle(
            "ContentsTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=NAVY,
            spaceAfter=12,
        ),
        "table_header": ParagraphStyle(
            "StudyTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.3,
            leading=9.4,
            textColor=WHITE,
        ),
        "table_cell": ParagraphStyle(
            "StudyTableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.25,
            leading=9.4,
            textColor=INK,
        ),
    }


class StudyDocTemplate(BaseDocTemplate):
    """Document template with stable navigation and multi-pass TOC entries."""

    def __init__(
        self,
        filename: str,
        *,
        metadata: dict[str, str],
        include_cover: bool,
    ) -> None:
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT_MARGIN,
            rightMargin=RIGHT_MARGIN,
            topMargin=TOP_MARGIN,
            bottomMargin=BOTTOM_MARGIN,
            title=metadata.get("title", "Study report"),
            author=metadata.get("author", ""),
            subject=metadata.get("subject", ""),
            keywords=metadata.get("keywords", ""),
            pageCompression=1,
            invariant=1,
        )
        self.study_metadata = metadata
        self.include_cover = include_cover
        self.heading_counts: dict[str, int] = {}
        self._outline_keys_seen: set[str] = set()

        frame = Frame(
            LEFT_MARGIN,
            BOTTOM_MARGIN,
            BODY_WIDTH,
            PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="study-body",
        )
        cover = PageTemplate(id="Cover", frames=[frame], onPage=self._draw_cover_page)
        body = PageTemplate(id="Body", frames=[frame], onPage=self._draw_body_page)
        self.addPageTemplates([cover, body] if include_cover else [body])

    def beforeDocument(self) -> None:  # noqa: N802 - ReportLab API
        self.heading_counts.clear()
        self._outline_keys_seen.clear()

    def afterFlowable(self, flowable: Flowable) -> None:  # noqa: N802 - ReportLab API
        if not isinstance(flowable, Paragraph):
            return
        level = HEADING_LEVELS.get(flowable.style.name)
        if level is None:
            return

        title = flowable.getPlainText()
        key = getattr(flowable, "_study_bookmark", None)
        if not key:
            base_key = _slug(title)
            count = self.heading_counts.get(base_key, 0) + 1
            self.heading_counts[base_key] = count
            key = f"{base_key}-{count}"

        logical_page = self.page - (1 if self.include_cover else 0)
        self.canv.bookmarkPage(key)
        if key not in self._outline_keys_seen:
            self.canv.addOutlineEntry(title, key, level=level, closed=level > 0)
            self._outline_keys_seen.add(key)
        self.notify("TOCEntry", (level, title, logical_page, key))

    def _set_pdf_metadata(self, canvas) -> None:
        metadata = self.study_metadata
        canvas.setTitle(metadata.get("title", "Study report"))
        canvas.setAuthor(metadata.get("author", ""))
        canvas.setSubject(metadata.get("subject", ""))
        canvas.setKeywords(metadata.get("keywords", ""))
        canvas.setCreator("ReportLab scholarly report renderer")

    def _draw_cover_page(self, canvas, _doc) -> None:
        self._set_pdf_metadata(canvas)
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_HEIGHT - 12 * mm, PAGE_WIDTH, 12 * mm, fill=1, stroke=0)
        canvas.setFillColor(TEAL)
        canvas.rect(0, 0, 7 * mm, PAGE_HEIGHT, fill=1, stroke=0)
        canvas.setStrokeColor(MID)
        canvas.setLineWidth(0.6)
        canvas.line(LEFT_MARGIN, 15 * mm, PAGE_WIDTH - RIGHT_MARGIN, 15 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(SLATE)
        canvas.drawString(LEFT_MARGIN, 9.5 * mm, "RESEARCH REPORT")
        canvas.drawRightString(
            PAGE_WIDTH - RIGHT_MARGIN,
            9.5 * mm,
            self.study_metadata.get("version", "Review draft"),
        )
        canvas.restoreState()

    def _draw_body_page(self, canvas, _doc) -> None:
        self._set_pdf_metadata(canvas)
        canvas.saveState()
        short_title = self.study_metadata.get(
            "short_title", self.study_metadata.get("title", "Study report")
        )
        if stringWidth(short_title, "Helvetica", 7.2) > BODY_WIDTH - 25 * mm:
            short_title = textwrap.shorten(short_title, width=72, placeholder="...")
        status = self.study_metadata.get("status", "Review draft")
        logical_page = canvas.getPageNumber() - (1 if self.include_cover else 0)

        canvas.setStrokeColor(MID)
        canvas.setLineWidth(0.45)
        canvas.line(LEFT_MARGIN, PAGE_HEIGHT - 14.5 * mm, PAGE_WIDTH - RIGHT_MARGIN, PAGE_HEIGHT - 14.5 * mm)
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(SLATE)
        canvas.drawString(LEFT_MARGIN, PAGE_HEIGHT - 11.2 * mm, short_title)
        canvas.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, PAGE_HEIGHT - 11.2 * mm, status)

        canvas.line(LEFT_MARGIN, 13.5 * mm, PAGE_WIDTH - RIGHT_MARGIN, 13.5 * mm)
        canvas.drawString(LEFT_MARGIN, 8.6 * mm, self.study_metadata.get("document_type", "Technical report"))
        canvas.setFont("Helvetica-Bold", 7.2)
        canvas.setFillColor(NAVY)
        canvas.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, 8.6 * mm, str(logical_page))
        canvas.restoreState()


def _cover_story(metadata: dict[str, str], styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    title = metadata.get("title", "Study report")
    subtitle = metadata.get("subtitle", "")
    document_type = metadata.get("document_type", "Research report")
    status = metadata.get("status", "Review draft")

    story: list[Flowable] = [
        Spacer(1, 38 * mm),
        Paragraph(_inline_markup(document_type.upper()), styles["cover_kicker"]),
        Paragraph(_inline_markup(title), styles["cover_title"]),
    ]
    if subtitle:
        story.append(Paragraph(_inline_markup(subtitle), styles["cover_subtitle"]))
    story.extend(
        [
            HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceBefore=3, spaceAfter=12),
            Table(
                [
                    [
                        Paragraph("<b>Author</b>", styles["cover_meta"]),
                        Paragraph(_inline_markup(metadata.get("author", "Not specified")), styles["cover_meta"]),
                    ],
                    [
                        Paragraph("<b>Date</b>", styles["cover_meta"]),
                        Paragraph(_inline_markup(metadata.get("date", "Undated")), styles["cover_meta"]),
                    ],
                    [
                        Paragraph("<b>Version</b>", styles["cover_meta"]),
                        Paragraph(_inline_markup(metadata.get("version", "Review draft")), styles["cover_meta"]),
                    ],
                    [
                        Paragraph("<b>Status</b>", styles["cover_meta"]),
                        Paragraph(_inline_markup(status), styles["cover_meta"]),
                    ],
                ],
                colWidths=[25 * mm, BODY_WIDTH - 25 * mm],
                style=TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEBELOW", (0, 0), (-1, -2), 0.3, MID),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                ),
            ),
        ]
    )
    repository = metadata.get("repository", "")
    if repository:
        story.extend(
            [
                Spacer(1, 9 * mm),
                Paragraph(
                    f"<b>Repository</b><br/>{_inline_markup(repository)}",
                    styles["cover_meta"],
                ),
            ]
        )
    story.extend([NextPageTemplate("Body"), PageBreak()])
    return story


def _toc_flowables(styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    toc = TableOfContents()
    toc.dotsMinLevel = 0
    toc.levelStyles = [
        ParagraphStyle(
            "TOCLevel1",
            fontName="Helvetica-Bold",
            fontSize=9.2,
            leading=13,
            leftIndent=0,
            firstLineIndent=0,
            textColor=NAVY,
            spaceBefore=4,
        ),
        ParagraphStyle(
            "TOCLevel2",
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            leftIndent=11,
            firstLineIndent=0,
            textColor=INK,
        ),
        ParagraphStyle(
            "TOCLevel3",
            fontName="Helvetica",
            fontSize=8,
            leading=10.5,
            leftIndent=22,
            firstLineIndent=0,
            textColor=SLATE,
        ),
        ParagraphStyle(
            "TOCLevel4",
            fontName="Helvetica-Oblique",
            fontSize=7.6,
            leading=10,
            leftIndent=33,
            firstLineIndent=0,
            textColor=SLATE,
        ),
    ]
    return [Paragraph("Contents", styles["toc_title"]), toc, PageBreak()]


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = re.split(r"(?<!\\)\|", stripped)
    return [cell.replace(r"\|", "|").strip() for cell in cells]


def _is_table_separator(line: str) -> bool:
    cells = _split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def _table_widths(rows: list[list[str]]) -> list[float]:
    column_count = len(rows[0])
    weights: list[float] = []
    for column in range(column_count):
        longest = max(
            len(re.sub(r"[*_`]", "", row[column])) if column < len(row) else 0
            for row in rows
        )
        weights.append(max(8.0, min(float(longest), 34.0)))

    total = sum(weights)
    widths = [BODY_WIDTH * weight / total for weight in weights]
    minimum = 21 * mm if column_count <= 6 else 14 * mm
    widths = [max(minimum, width) for width in widths]
    if sum(widths) > BODY_WIDTH:
        scale = BODY_WIDTH / sum(widths)
        widths = [width * scale for width in widths]
    return widths


def _make_table(
    raw_rows: list[list[str]],
    separator: list[str],
    styles: dict[str, ParagraphStyle],
) -> LongTable:
    column_count = len(raw_rows[0])
    normalized_rows = [row + [""] * (column_count - len(row)) for row in raw_rows]
    normalized_rows = [row[:column_count] for row in normalized_rows]
    alignments: list[tuple[str, int]] = []
    for marker in separator[:column_count]:
        if marker.startswith(":") and marker.endswith(":"):
            alignments.append(("CENTER", TA_CENTER))
        elif marker.endswith(":"):
            alignments.append(("RIGHT", TA_RIGHT))
        else:
            alignments.append(("LEFT", TA_LEFT))
    header_styles = [
        ParagraphStyle(
            f"StudyTableHeader{column}",
            parent=styles["table_header"],
            alignment=alignments[column][1],
        )
        for column in range(column_count)
    ]
    cell_styles = [
        ParagraphStyle(
            f"StudyTableCell{column}",
            parent=styles["table_cell"],
            alignment=alignments[column][1],
        )
        for column in range(column_count)
    ]
    rendered = [
        [
            Paragraph(
                _inline_markup(cell),
                header_styles[column] if row_index == 0 else cell_styles[column],
            )
            for column, cell in enumerate(row)
        ]
        for row_index, row in enumerate(normalized_rows)
    ]
    widths = _table_widths(normalized_rows)
    commands: list[tuple] = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BOX", (0, 0), (-1, -1), 0.55, MID),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, MID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in range(1, len(rendered)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE))
    for column, (alignment, _paragraph_alignment) in enumerate(alignments):
        commands.append(("ALIGN", (column, 0), (column, -1), alignment))

    table = LongTable(
        rendered,
        colWidths=widths,
        repeatRows=1,
        splitByRow=1,
        hAlign="LEFT",
        spaceBefore=4,
        spaceAfter=10,
    )
    table.setStyle(TableStyle(commands))
    return table


def _parse_image_width(attributes: str, natural_width: float) -> float:
    match = re.search(r"\bwidth\s*=\s*([0-9.]+)(%|mm|cm|in|pt)?", attributes)
    if not match:
        return min(BODY_WIDTH, natural_width)
    value = float(match.group(1))
    unit = match.group(2) or "%"
    if unit == "%":
        return min(BODY_WIDTH, BODY_WIDTH * value / 100)
    factors = {"mm": mm, "cm": 10 * mm, "in": 25.4 * mm, "pt": 1.0}
    return min(BODY_WIDTH, value * factors[unit])


def _image_flowables(
    line: str,
    source_dir: Path,
    styles: dict[str, ParagraphStyle],
) -> list[Flowable] | None:
    match = re.match(r"^!\[(.*?)\]\((.+?)\)\s*(?:\{([^}]*)\})?\s*$", line.strip())
    if not match:
        return None
    caption, raw_target, attributes = match.groups()
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    target = unquote(target)
    if re.match(r"^[a-z]+://", target, flags=re.IGNORECASE):
        raise ValueError(f"Remote images are not supported in PDF sources: {target}")
    path = (source_dir / target).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image referenced by Markdown does not exist: {path}")
    if path.suffix.lower() == ".svg":
        raise ValueError(f"Use the PNG companion for PDF rendering instead of SVG: {path}")

    image = Image(str(path))
    requested_width = _parse_image_width(attributes or "", image.drawWidth)
    scale = requested_width / image.drawWidth
    requested_height = image.drawHeight * scale
    maximum_height = 158 * mm
    if requested_height > maximum_height:
        scale *= maximum_height / requested_height
    image.drawWidth *= scale
    image.drawHeight *= scale
    image.hAlign = "CENTER"

    result: list[Flowable] = [image]
    if caption:
        result.append(Paragraph(_inline_markup(caption), styles["caption"]))
    else:
        result.append(Spacer(1, 7))
    return [KeepTogether(result)]


def _list_flowable(
    entries: list[tuple[int, str, str]],
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    rendered: list[Flowable] = []
    group: list[tuple[int, str, str]] = []

    def flush() -> None:
        if not group:
            return
        indent, marker, _ = group[0]
        ordered = marker[0].isdigit()
        start = int(re.match(r"\d+", marker).group()) if ordered else None
        items = [
            ListItem(Paragraph(_inline_markup(text), styles["list"]))
            for _, _, text in group
        ]
        rendered.append(
            ListFlowable(
                items,
                bulletType="1" if ordered else "bullet",
                start=start,
                leftIndent=15 + indent * 8,
                bulletFontName="Helvetica-Bold",
                bulletFontSize=7.4,
                bulletColor=BLUE,
                bulletOffsetY=1,
                spaceBefore=1,
                spaceAfter=6,
            )
        )
        group.clear()

    for entry in entries:
        if group:
            same_indent = entry[0] == group[0][0]
            same_kind = entry[1][0].isdigit() == group[0][1][0].isdigit()
            if not (same_indent and same_kind):
                flush()
        group.append(entry)
    flush()
    return rendered


def _markdown_story(
    lines: list[str],
    *,
    source_dir: Path,
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    story: list[Flowable] = []
    heading_occurrences: dict[str, int] = {}
    source_heading_levels = [
        len(match.group(1))
        for line in lines
        if (match := re.match(r"^(#{1,4})\s+", line))
    ]
    heading_offset = max(0, min(source_heading_levels, default=1) - 1)
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        if stripped == "<!-- toc -->":
            story.extend(_toc_flowables(styles))
            index += 1
            continue
        if stripped == "<!-- pagebreak -->":
            story.append(PageBreak())
            index += 1
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            index += 1
            continue

        if stripped.startswith("```"):
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if index >= len(lines):
                raise ValueError("Unclosed fenced code block")
            code = "\n".join(code_lines)
            story.append(Preformatted(code, styles["code"], maxLineLength=96))
            index += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.+?)\s*#*\s*$", line)
        if heading:
            level = max(1, len(heading.group(1)) - heading_offset)
            title = heading.group(2).strip()
            base_key = _slug(title)
            occurrence = heading_occurrences.get(base_key, 0) + 1
            heading_occurrences[base_key] = occurrence
            key = f"{base_key}-{occurrence}"
            paragraph = Paragraph(_inline_markup(title), styles[f"h{level}"])
            paragraph._study_bookmark = key  # type: ignore[attr-defined]
            story.append(paragraph)
            index += 1
            continue

        image_flowables = _image_flowables(line, source_dir, styles)
        if image_flowables is not None:
            story.extend(image_flowables)
            index += 1
            continue

        if (
            "|" in line
            and index + 1 < len(lines)
            and "|" in lines[index + 1]
            and _is_table_separator(lines[index + 1])
        ):
            rows = [_split_table_row(line)]
            separator = _split_table_row(lines[index + 1])
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(_split_table_row(lines[index]))
                index += 1
            if len(rows[0]) != len(separator):
                raise ValueError("Markdown table header and separator have different column counts")
            story.append(_make_table(rows, separator, styles))
            continue

        list_match = re.match(r"^(\s*)([-+*]|\d+[.)])\s+(.+)$", line)
        if list_match:
            entries: list[tuple[int, str, str]] = []
            while index < len(lines):
                current = re.match(r"^(\s*)([-+*]|\d+[.)])\s+(.+)$", lines[index])
                if not current:
                    break
                entries.append((len(current.group(1)), current.group(2), current.group(3)))
                index += 1
            story.extend(_list_flowable(entries, styles))
            continue

        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip()[1:].strip())
                index += 1
            quote = Paragraph(_inline_markup(" ".join(quote_lines)), styles["quote"])
            box = Table([[quote]], colWidths=[BODY_WIDTH], hAlign="LEFT")
            box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                        ("LINEBEFORE", (0, 0), (0, -1), 2.2, TEAL),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.extend([box, Spacer(1, 7)])
            continue

        if re.fullmatch(r"[-*_]{3,}", stripped):
            story.append(HRFlowable(width="100%", thickness=0.5, color=MID, spaceBefore=4, spaceAfter=8))
            index += 1
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index]
            candidate_stripped = candidate.strip()
            if not candidate_stripped:
                break
            if (
                candidate_stripped in {"<!-- toc -->", "<!-- pagebreak -->"}
                or candidate_stripped.startswith("<!--")
                or candidate_stripped.startswith("```")
                or re.match(r"^#{1,4}\s+", candidate)
                or re.match(r"^(\s*)([-+*]|\d+[.)])\s+", candidate)
                or candidate_stripped.startswith(">")
                or re.match(r"^!\[.*?\]\(.+?\)", candidate_stripped)
                or re.fullmatch(r"[-*_]{3,}", candidate_stripped)
                or (
                    "|" in candidate
                    and index + 1 < len(lines)
                    and _is_table_separator(lines[index + 1])
                )
            ):
                break
            paragraph_lines.append(candidate_stripped)
            index += 1
        story.append(Paragraph(_inline_markup(" ".join(paragraph_lines)), styles["body"]))

    return story


def build_pdf(
    input_path: Path,
    output_path: Path,
    *,
    overrides: dict[str, str | None] | None = None,
    include_cover: bool = True,
) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Markdown source does not exist: {input_path}")

    parsed = _read_front_matter(input_path)
    metadata = dict(parsed.metadata)
    for key, value in (overrides or {}).items():
        if value is not None:
            metadata[key] = value
    metadata.setdefault("title", _first_heading(parsed.body_lines) or input_path.stem.replace("_", " "))
    metadata.setdefault("document_type", "Technical report")
    metadata.setdefault("status", "Review draft")
    metadata.setdefault("version", metadata["status"])

    styles = _styles()
    story: list[Flowable] = []
    if include_cover:
        story.extend(_cover_story(metadata, styles))
    story.extend(
        _markdown_story(
            parsed.body_lines,
            source_dir=input_path.parent,
            styles=styles,
        )
    )
    if not story:
        raise ValueError(f"Markdown source contains no renderable content: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = StudyDocTemplate(
        str(output_path),
        metadata=metadata,
        include_cover=include_cover,
    )
    document.multiBuild(story)


def main() -> None:
    args = parse_args()
    output = args.output or ROOT / "output" / "pdf" / f"{args.input.stem.lower()}.pdf"
    build_pdf(
        args.input,
        output,
        overrides={
            "title": args.title,
            "author": args.author,
            "date": args.date,
            "status": args.status,
        },
        include_cover=not args.no_cover,
    )
    print(f"Built {output.resolve()}")


if __name__ == "__main__":
    main()
