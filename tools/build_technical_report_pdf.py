"""Build the polished POLAR technical report PDF from locked tracked evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor("#0F172A")
BLUE = colors.HexColor("#2563EB")
PURPLE = colors.HexColor("#7C3AED")
TEAL = colors.HexColor("#059669")
RED = colors.HexColor("#DC2626")
SLATE = colors.HexColor("#475569")
LIGHT = colors.HexColor("#F1F5F9")
MID = colors.HexColor("#CBD5E1")
WHITE = colors.white


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "pdf" / "polar_technical_report.pdf",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.2,
            textColor=NAVY,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=10.2,
            textColor=SLATE,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.8,
            leading=10.4,
            textColor=SLATE,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=NAVY,
            spaceBefore=3,
            spaceAfter=9,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.2,
            leading=15,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=31,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=14,
            leading=19,
            textColor=SLATE,
            spaceAfter=22,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=SLATE,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12.7,
            leftIndent=12,
            firstLineIndent=-7,
            bulletIndent=4,
            textColor=NAVY,
            spaceAfter=4,
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.4,
            leftIndent=12,
            firstLineIndent=-12,
            textColor=NAVY,
            spaceAfter=4,
        ),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullet(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(f"- {text}", style)


def styled_table(
    rows: list[list[str]],
    widths: list[float],
    text_style: ParagraphStyle,
    *,
    header: bool = True,
    font_size: float = 7.7,
) -> Table:
    header_style = ParagraphStyle(
        "TableHeader",
        parent=text_style,
        fontName="Helvetica-Bold",
        textColor=WHITE,
    )
    cells = [
        [Paragraph(str(value), header_style if header and index == 0 else text_style) for value in row]
        for index, row in enumerate(rows)
    ]
    table = LongTable(cells, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.3),
        ("GRID", (0, 0), (-1, -1), 0.35, MID),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [WHITE, LIGHT]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def report_image(path: Path, max_width: float, max_height: float) -> Image:
    image = Image(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return image


def cover(canvas, document) -> None:
    canvas.saveState()
    canvas.setTitle("Leakage-Safe Transfer Learning for Still-Image Posture Recognition")
    canvas.setAuthor("Abdulla Huseyinli")
    canvas.setSubject("Locked POLAR study with external-transfer and attribution audits")
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 18 * mm, width, 18 * mm, stroke=0, fill=1)
    canvas.setFillColor(PURPLE)
    canvas.rect(0, height - 20.5 * mm, width, 2.5 * mm, stroke=0, fill=1)
    canvas.restoreState()


def page(canvas, document) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(SLATE)
    canvas.drawRightString(width - 22 * mm, 10.5 * mm, f"{document.page}")
    canvas.restoreState()


class ReportDocTemplate(SimpleDocTemplate):
    """Draw page numbers after flowables so figures cannot cover them."""

    def afterPage(self) -> None:
        if self.page == 1:
            cover(self.canv, self)
        else:
            page(self.canv, self)


def build_report(repository: Path, output: Path) -> None:
    result_dir = repository / "results"
    asset_dir = repository / "assets"
    test = read_json(result_dir / "polar_test_summary.json")
    uncertainty = read_json(result_dir / "polar_test_uncertainty.json")
    external = read_json(result_dir / "polar_external_summary.json")
    faith = read_json(result_dir / "polar_faithfulness_summary.json")
    fault = read_json(result_dir / "polar_fault_summary.json")
    audit = read_json(result_dir / "polar_data_audit.json")
    fits = read_json(result_dir / "polar_final_fit_manifest.json")
    overlap = read_json(result_dir / "polar_external_overlap_audit.json")
    selection = read_json(result_dir / "polar_final_selection_lock.json")
    extension = read_json(result_dir / "polar_extension_summary.json")
    if {
        test.get("status"),
        external.get("status"),
        faith.get("status"),
        fault.get("status"),
    } != {
        "LOCKED_FINAL_TEST_COMPLETE",
        "LOCKED_EXTERNAL_EVALUATION_COMPLETE",
        "LOCKED_POLAR_FAITHFULNESS_COMPLETE",
        "LOCKED_POLAR_FAULT_ROBUSTNESS_COMPLETE",
    }:
        raise RuntimeError("The report requires complete locked evidence")

    output.parent.mkdir(parents=True, exist_ok=True)
    style = styles()
    document = ReportDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=21 * mm,
        bottomMargin=18 * mm,
        title="Leakage-Safe Transfer Learning for Still-Image Posture Recognition",
        author="Abdulla Huseyinli",
    )
    story = []
    primary = test["primary_metrics"]
    primary_ci = uncertainty["locked_ensemble"]

    story.extend(
        [
            Spacer(1, 21 * mm),
            para("Leakage-Safe Transfer Learning for Still-Image Posture Recognition", style["cover_title"]),
            para(
                "A locked POLAR study with external-transfer and attribution audits",
                style["cover_subtitle"],
            ),
            para("Abdulla Huseyinli", style["cover_meta"]),
            para("Technical report - version 2.0.0 - 23 August 2026", style["cover_meta"]),
            Spacer(1, 12 * mm),
        ]
    )
    key_table = Table(
        [
            [
                para(f"<font size='18'>{primary['macro_f1']:.3f}</font><br/>Macro-F1", style["callout"]),
                para(f"<font size='18'>{primary['accuracy']:.3f}</font><br/>Accuracy", style["callout"]),
                para(
                    f"<font size='18'>{test['test_rows_read']:,}</font><br/>Held-out images",
                    style["callout"],
                ),
            ]
        ],
        colWidths=[53 * mm, 53 * mm, 53 * mm],
        rowHeights=[27 * mm],
    )
    key_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.7, MID),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, MID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.extend([key_table, Spacer(1, 10 * mm)])
    story.append(para("Abstract", style["h2"]))
    story.append(
        para(
            "This study evaluates ConvNeXt and DINOv2 transfer learning on a cleaned four-class "
            "POLAR subset. A source-aware audit quarantined 125 cross-split related images before "
            "fitting. All model, classifier, seed, epoch, and blend decisions were locked on "
            "13,285 development images. Nine neural fits and three probes were hash-verified before "
            f"one-time evaluation on 3,329 held-out images. The predeclared ensemble achieved "
            f"{primary['macro_f1']:.4f} macro-F1 (95% CI [{primary_ci['ci_95_low']:.4f}, "
            f"{primary_ci['ci_95_high']:.4f}]). External V-COCO transfer and attribution sanity "
            "show that this strong in-domain result is not equivalent to deployment readiness.",
            style["body"],
        )
    )
    story.append(
        para(
            "Scope: reproducible benchmark, not an exact state-of-the-art claim.", style["small"]
        )
    )
    story.append(PageBreak())

    story.extend([para("1. Study design and evidence boundary", style["h1"])])
    story.append(
        para(
            "The central design constraint is that the official test partition cannot influence "
            "model selection. The official train split fitted candidates; validation selected "
            "views, adaptation depth, regularization, classifier settings, epoch rules, and "
            "ensemble weights. Train and validation were combined only after the final lock.",
            style["body"],
        )
    )
    for text in (
        "Primary task: sitting, standing, walking, and running; macro-F1 is primary.",
        "Secondary task: walking and running collapsed into one class.",
        "Uncertainty: 10,000 class-stratified bootstrap resamples, seed 20260822.",
        "Comparison: paired resampling on the same test rows.",
        "External and explanation audits have selection role 'none'.",
    ):
        story.append(bullet(text, style["bullet"]))

    story.append(para("1.1 Clean POLAR split", style["h2"]))
    counts = audit["clean_target_counts"]
    data_rows = [["Split", "Sitting", "Standing", "Walking", "Running", "Total"]]
    for split, label in (("train", "Train"), ("val", "Validation"), ("test", "Test")):
        values = counts[split]
        data_rows.append(
            [
                label,
                f"{values['sitting']:,}",
                f"{values['standing']:,}",
                f"{values['walking']:,}",
                f"{values['running']:,}",
                f"{sum(values.values()):,}",
            ]
        )
    story.append(styled_table(data_rows, [32 * mm, 24 * mm, 25 * mm, 24 * mm, 24 * mm, 24 * mm], style["small"]))
    story.append(para("1.2 Source-related quarantine", style["h2"]))
    story.append(
        para(
            "Byte hashes found no exact cross-split duplicates, but perceptual retrieval identified "
            "alternate crops, color variants, and nearby source frames. Confirmation required a "
            "64-bit perceptual-hash distance at most six and normalized grayscale correlation at "
            "least 0.90, or equivalent independent source evidence. The audit quarantined 125 "
            "images in 61 connected components before supervised fitting. No test image moved into "
            "development.",
            style["body"],
        )
    )
    control_rows = [
        ["Control", "Recorded value"],
        ["Clean development rows", f"{fits['development_rows']:,}"],
        ["Verified neural fits", "9 (3 families x 3 seeds)"],
        ["Verified frozen-feature probes", "3"],
        ["Final-fit test rows read", str(fits["test_rows_read"])],
        ["Official test-manifest opens", str(test["official_test_manifest_open_count"])],
        ["Selection-lock SHA-256", test["selection_lock_sha256"]],
    ]
    story.append(styled_table(control_rows, [58 * mm, 101 * mm], style["small"]))
    story.append(PageBreak())

    story.append(para("2. Models and final-stage classifiers", style["h1"]))
    model_rows = [
        ["Component", "View", "Adaptation", "Augmentation", "Dropout", "Epochs"],
        ["ConvNeXt-S", "Full frame", "Full; layer decay 0.70", "Mild", "0.10", "12"],
        ["DINOv2-S", "Person + 25% context", "Full; layer decay 0.75", "Moderate", "0.10", "12"],
        ["DINOv2-B", "Person + 25% context", "Top four blocks", "Mild", "0.10", "7"],
    ]
    story.append(
        styled_table(
            model_rows,
            [27 * mm, 33 * mm, 39 * mm, 25 * mm, 18 * mm, 17 * mm],
            style["small"],
            font_size=7.2,
        )
    )
    story.append(
        para(
            "Neural components average seeds 42, 52, and 62. Fixed epoch counts are the median "
            "best epochs from corresponding development confirmation runs. The strongest frozen "
            "representation concatenates normalized CLS tokens from the last four DINOv2-B layers "
            "and the final mean patch token across full-frame and person-plus-10%-context views, "
            "yielding 7,680 features.",
            style["body"],
        )
    )
    story.append(para("2.1 Classifier probes", style["h2"]))
    story.append(
        para(
            "The linear endpoint is standardized multinomial logistic regression with C=0.001. "
            "The nonlinear endpoint is an RBF SVM with C=10 and gamma=1/7680, followed by "
            "five-fold sigmoid calibration. RBF settings were transferred from the declared "
            "development screen rather than retuned on final features.",
            style["body"],
        )
    )
    story.append(para("2.2 Locked probability blend", style["h2"]))
    weight_rows = [["Component", "Weight"]]
    display = {
        "locked_ensemble": "Locked ensemble",
        "convnext_small_full": "ConvNeXt-S full",
        "dinov2_small_moderate": "DINOv2-S full",
        "dinov2_base_top4": "DINOv2-B top four",
        "dinov2_base_multilayer_logistic": "DINOv2-B logistic",
        "dinov2_base_multilayer_rbf": "DINOv2-B RBF SVM",
        "direct_three_class_probe": "Direct three-class probe",
    }
    for name, value in selection["ensemble"]["weights"].items():
        weight_rows.append([display[name], f"{value:.0%}"])
    story.append(styled_table(weight_rows, [110 * mm, 35 * mm], style["small"]))
    story.append(
        para(
            f"All final fits completed in {fits['total_fit_runtime_seconds'] / 3600:.2f} hours. "
            "Checkpoint and pipeline hashes were verified before the test gate opened.",
            style["body"],
        )
    )
    story.append(PageBreak())

    story.append(para("3. Locked held-out results", style["h1"]))
    metrics = pd.read_csv(result_dir / "polar_test_metrics.csv")
    metric_rows = [["Candidate", "Macro-F1", "95% CI", "Accuracy", "Log loss", "ECE"]]
    for row in metrics.itertuples(index=False):
        interval = uncertainty[row.candidate]
        metric_rows.append(
            [
                display.get(row.candidate, "Locked ensemble"),
                f"{row.macro_f1:.4f}",
                f"[{interval['ci_95_low']:.4f}, {interval['ci_95_high']:.4f}]",
                f"{row.accuracy:.4f}",
                f"{row.log_loss:.4f}",
                f"{row.ece:.4f}",
            ]
        )
    story.append(
        styled_table(
            metric_rows,
            [50 * mm, 22 * mm, 36 * mm, 20 * mm, 20 * mm, 17 * mm],
            style["small"],
            font_size=6.9,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(report_image(asset_dir / "polar_test_comparison.png", 160 * mm, 88 * mm))
    story.append(
        para(
            "Figure 1. Every candidate was declared before test access. The ensemble is the primary "
            "candidate regardless of the observed ordering.",
            style["caption"],
        )
    )
    paired = uncertainty["locked_ensemble_paired_deltas"]
    smallest = min(paired.values(), key=lambda item: item["point_estimate"])
    story.append(
        para(
            "The ensemble's paired 95% macro-F1 interval is positive against every component. "
            f"The smallest gain is +{smallest['point_estimate']:.4f}, with interval "
            f"[{smallest['ci_95_low']:.4f}, {smallest['ci_95_high']:.4f}]. The collapsed "
            "three-class ensemble reaches 0.9611 macro-F1 and 0.9622 accuracy.",
            style["body"],
        )
    )
    story.append(PageBreak())

    story.append(para("4. Class behavior", style["h1"]))
    per_class = pd.read_csv(result_dir / "polar_test_per_class.csv")
    per_class = per_class[per_class["candidate"].eq("locked_ensemble")]
    class_rows = [["Class", "Precision", "Recall", "F1", "Support"]] + [
        [
            str(record["class"]).title(),
            f"{record['precision']:.3f}",
            f"{record['recall']:.3f}",
            f"{record['f1']:.3f}",
            f"{int(record['support']):,}",
        ]
        for record in per_class.to_dict("records")
    ]
    story.append(styled_table(class_rows, [45 * mm, 28 * mm, 28 * mm, 28 * mm, 28 * mm], style["small"]))
    story.append(Spacer(1, 4 * mm))
    story.append(report_image(asset_dir / "polar_confusion_matrix.png", 142 * mm, 110 * mm))
    story.append(
        para(
            "Figure 2. Row-normalized confusion for the locked ensemble. Walking is hardest: 8.2% "
            "of walking rows are predicted as standing and 4.4% as running.",
            style["caption"],
        )
    )
    story.append(PageBreak())

    story.append(para("5. Where the gain came from", style["h1"]))
    scale = sorted(extension["scale_curve"], key=lambda item: item["actual_train_size"])
    story.append(
        para(
            f"The frozen DINOv2-B curve rises from {scale[0]['macro_f1_mean']:.4f} macro-F1 at "
            f"{scale[0]['actual_train_size']:,} training images to {scale[-1]['macro_f1_mean']:.4f} "
            f"at {scale[-1]['actual_train_size']:,}. This 6.64-point increase is larger than any "
            "single seed-42 regularization intervention.",
            style["body"],
        )
    )
    story.append(report_image(asset_dir / "polar_scale_curve.png", 158 * mm, 94 * mm))
    story.append(para("Figure 3. Frozen DINOv2-B validation learning curve.", style["caption"]))
    story.append(para("5.1 Regularization screen", style["h2"]))
    reg_rows = [
        ["Seed-42 DINOv2-S intervention", "Validation macro-F1"],
        ["Dropout 0.20", "0.9191"],
        ["Dropout 0.10 + mild augmentation", "0.9184"],
        ["Moderate augmentation", "0.9182"],
        ["Label smoothing 0.05", "0.9142"],
        ["No random erasing", "0.9124"],
        ["MixUp 0.20", "0.9120"],
        ["No dropout", "0.9113"],
        ["Inverse-frequency weights", "0.9082"],
    ]
    story.append(styled_table(reg_rows, [110 * mm, 40 * mm], style["small"]))
    story.append(
        para(
            "The evidence supports retaining dropout and image augmentation, not stacking every "
            "available regularizer. Multi-seed stability and complementarity favored the moderate "
            "DINOv2-S variant for the blend.",
            style["body"],
        )
    )
    story.append(PageBreak())

    story.append(para("6. SVM tradeoff and external transfer", style["h1"]))
    logistic = fits["probes"]["dinov2_base_multilayer_logistic"]
    rbf = fits["probes"]["dinov2_base_multilayer_rbf"]
    probe_rows = [
        ["Probe", "Test F1", "Log loss", "External F1", "Fit time", "Size"],
        [
            "Logistic",
            "0.9258",
            "0.1764",
            "0.6392",
            f"{logistic['fit_seconds']:.1f} s",
            f"{logistic['pipeline_bytes'] / 1_000_000:.1f} MB",
        ],
        [
            "Calibrated RBF SVM",
            "0.9274",
            "0.2280",
            "0.6504",
            f"{rbf['fit_seconds'] / 60:.1f} min",
            f"{rbf['pipeline_bytes'] / 1_000_000:.1f} MB",
        ],
    ]
    story.append(styled_table(probe_rows, [40 * mm, 22 * mm, 25 * mm, 26 * mm, 25 * mm, 25 * mm], style["small"]))
    story.append(
        para(
            "The nonlinear boundary adds a small accuracy gain and useful ensemble diversity, but "
            "logistic regression is the practical endpoint for compact, calibrated serving. The "
            f"RBF model averages {rbf['mean_support_vectors']:.0f} support vectors per calibration "
            "fold and is more than 2,000 times larger.",
            style["body"],
        )
    )
    story.append(para("6.1 V-COCO no-retuning evaluation", style["h2"]))
    story.append(
        para(
            f"The overlap audit compared {overlap['polar_clean_rows']:,} clean POLAR records with "
            f"{overlap['vcoco_unique_images']:,} V-COCO images and found zero exact matches, zero "
            "perceptual candidates, and zero confirmed source-related pairs. The locked models "
            "were evaluated without retuning.",
            style["body"],
        )
    )
    ext_frame = pd.read_csv(result_dir / "polar_external_image_metrics.csv")
    ext_rows = [["Candidate", "Macro-F1", "Accuracy", "Log loss"]]
    for row in ext_frame.itertuples(index=False):
        ext_rows.append(
            [
                display.get(row.candidate.replace("locked_ensemble_collapsed", "locked_ensemble"), row.candidate),
                f"{row.macro_f1:.4f}",
                f"{row.accuracy:.4f}",
                f"{row.log_loss:.4f}",
            ]
        )
    story.append(styled_table(ext_rows, [73 * mm, 28 * mm, 28 * mm, 28 * mm], style["small"], font_size=7.0))
    story.append(PageBreak())

    story.append(para("7. External result and interpretation", style["h1"]))
    story.append(report_image(asset_dir / "polar_external_validation.png", 160 * mm, 98 * mm))
    story.append(
        para(
            "Figure 4. Image-level V-COCO macro-F1 for fixed three-class mappings. DINOv2-B top-four "
            "adaptation transfers best descriptively; the locked ensemble remains the predeclared "
            "primary candidate.",
            style["caption"],
        )
    )
    story.append(
        para(
            "The collapsed ensemble falls from 0.9611 in-domain macro-F1 to 0.6669 externally, a "
            "29.4-point gap. External standing recall is 0.412, while walking/running recall is "
            "0.929 but precision is 0.349. Image composition and annotation semantics therefore "
            "change the decision problem. The result argues for domain-generalization work rather "
            "than another POLAR-only tuning loop.",
            style["body"],
        )
    )
    for text in (
        "In-domain selection and external robustness are not aligned objectives.",
        "The adapted DINOv2-B component transfers better than the in-domain-optimal blend.",
        "External results diagnose generalization; they do not reopen model selection.",
    ):
        story.append(bullet(text, style["bullet"]))
    story.append(PageBreak())

    story.append(para("8. Attribution faithfulness", style["h1"]))
    story.append(
        para(
            "The fixed cohort contains 256 images balanced by class and person-box-area quartile. "
            "ConvNeXt uses Grad-CAM and DINOv2-B uses 16-step integrated gradients. The audit "
            "combines deletion/insertion, nested random deletion, bbox localization, equal-area "
            "person/context occlusion, full/crop consistency, alternative targets, and two levels "
            "of parameter randomization.",
            style["body"],
        )
    )
    conv = faith["aggregate"]["convnext_small_full"]
    dino = faith["aggregate"]["dinov2_base_top4"]
    faith_rows = [
        ["Metric", "ConvNeXt-S", "DINOv2-B"],
        ["Deletion AUC (lower)", f"{conv['deletion_auc']['mean']:.3f}", f"{dino['deletion_auc']['mean']:.3f}"],
        ["Random-minus-targeted gap", f"{conv['deletion_selectivity_gap']['mean']:.3f}", f"{dino['deletion_selectivity_gap']['mean']:.3f}"],
        ["Insertion AUC", f"{conv['insertion_auc']['mean']:.3f}", f"{dino['insertion_auc']['mean']:.3f}"],
        ["Person mass / bbox area", f"{conv['person_attribution_mass_lift']['mean']:.3f}", f"{dino['person_attribution_mass_lift']['mean']:.3f}"],
        ["Person-minus-context drop", f"{conv['person_minus_context_occlusion_drop']['mean']:.3f}", f"{dino['person_minus_context_occlusion_drop']['mean']:.3f}"],
        ["Alternative-target Spearman", f"{conv['target_vs_alternative_attribution_spearman']['mean']:.3f}", f"{dino['target_vs_alternative_attribution_spearman']['mean']:.3f}"],
        ["Randomized-head Spearman", f"{conv['randomized_head_spearman']['mean']:.3f}", f"{dino['randomized_head_spearman']['mean']:.3f}"],
        ["Randomized-layers Spearman", f"{conv['randomized_adapted_cascade_spearman']['mean']:.3f}", f"{dino['randomized_adapted_cascade_spearman']['mean']:.3f}"],
    ]
    story.append(styled_table(faith_rows, [91 * mm, 34 * mm, 34 * mm], style["small"]))
    story.append(Spacer(1, 4 * mm))
    story.append(report_image(asset_dir / "polar_faithfulness.png", 163 * mm, 119 * mm))
    story.append(para("Figure 5. Perturbation curves and bbox-aware attribution diagnostics.", style["caption"]))
    story.append(PageBreak())

    story.append(para("9. Attribution sanity result", style["h1"]))
    story.append(report_image(asset_dir / "polar_attribution_sanity.png", 160 * mm, 93 * mm))
    story.append(
        para(
            "Figure 6. Lower correlation after changing the target or learned parameters indicates "
            "greater explanation sensitivity.",
            style["caption"],
        )
    )
    story.append(
        para(
            "ConvNeXt provides the stronger evidence: targeted deletion separates from random "
            "deletion, person attribution has 2.37x area-normalized lift, and maps change after "
            "target and parameter randomization. DINOv2-B integrated gradients place 84.8% of "
            "positive mass inside person boxes, but area-normalized lift is only 1.10 and maps "
            "remain highly correlated after both sanity interventions. DINO integrated gradients "
            "are therefore retained as coarse localization diagnostics, not promoted as faithful "
            "causal explanations.",
            style["body"],
        )
    )
    story.append(
        para(
            "Two rows per family had full-frame projected person boxes and no available equal-area "
            "context. They remain in every other metric and are explicitly excluded only from the "
            "matched-context statistic. Maximum probability replay error was 0.00176.",
            style["body"],
        )
    )
    story.append(PageBreak())

    story.append(para("10. Bounded bit-flip robustness", style["h1"]))
    story.append(
        para(
            "Input faults flip exact bits in post-resize uint8 RGB tensors before normalization. "
            "Parameter faults symmetrically quantize each classifier weight matrix to int8 and "
            "flip exact bits. Three declared fault seeds are averaged. This audit is reported "
            "separately from attribution faithfulness.",
            style["body"],
        )
    )
    fault_frame = pd.DataFrame(fault["aggregate_results"])
    fault_frame = fault_frame[fault_frame["fault_seed"].astype(str).eq("aggregate")]
    fault_rows = [["Condition", "Level", "ConvNeXt agreement", "DINOv2 agreement"]]
    for condition, level, label in (
        ("uint8_input_bit_flip_rate", 1e-5, "Input bit rate"),
        ("uint8_input_bit_flip_rate", 1e-4, "Input bit rate"),
        ("uint8_input_bit_flip_rate", 1e-3, "Input bit rate"),
        ("symmetric_int8_head_weight_bit_flips", 16.0, "Head bit flips"),
    ):
        selected = fault_frame[
            fault_frame["condition"].eq(condition) & fault_frame["level"].eq(level)
        ].set_index("family")
        fault_rows.append(
            [
                label,
                f"{level:g}",
                f"{selected.loc['convnext_small_full', 'prediction_agreement_with_clean']:.3f}",
                f"{selected.loc['dinov2_base_top4', 'prediction_agreement_with_clean']:.3f}",
            ]
        )
    story.append(styled_table(fault_rows, [56 * mm, 28 * mm, 38 * mm, 38 * mm], style["small"]))
    story.append(Spacer(1, 5 * mm))
    story.append(report_image(asset_dir / "polar_fault_robustness.png", 162 * mm, 76 * mm))
    story.append(para("Figure 7. Agreement with clean predictions under declared fault levels.", style["caption"]))
    story.append(
        para(
            "The result shows local prediction stability on the fixed cohort. It does not test "
            "persistent memory faults, random backbone corruption, or hardware failure, and it is "
            "not a safety certification.",
            style["body"],
        )
    )
    story.append(PageBreak())

    story.append(para("11. Conclusions and limitations", style["h1"]))
    for text in (
        "Data scale is the largest isolated performance amplifier in the study.",
        "DINOv2-B representations support strong linear and nonlinear final-stage classifiers.",
        "The development-locked blend has a statistically supported gain over every component.",
        "The external-domain gap is large; POLAR performance is not a deployment guarantee.",
        "ConvNeXt Grad-CAM passes declared sanity checks more convincingly than DINOv2-B integrated gradients.",
    ):
        story.append(bullet(text, style["bullet"]))
    story.append(para("11.1 Claim boundary", style["h2"]))
    story.append(
        para(
            "No exact state-of-the-art claim is made. The task is a cleaned four-class subset rather "
            "than the full nine-label POLAR benchmark, and subject/session identifiers are not "
            "available. V-COCO uses a different source and annotation policy. A formal preprint "
            "would be stronger after replication on additional labels or an independent dataset "
            "with subject/session boundaries and a predeclared domain-generalization intervention.",
            style["body"],
        )
    )
    story.append(para("11.2 Reproducibility", style["h2"]))
    story.append(
        para(
            "The repository publishes code, protocols, selection locks, exclusion records, metrics, "
            "uncertainty, figure builders, and per-file hashes. Raw images, checkpoints, fitted "
            "classifier binaries, local paths, and dense predictions remain local. GitHub Actions "
            "runs lint, compilation, tests, and the portable evidence validator on Linux.",
            style["body"],
        )
    )
    story.append(
        para(
            "Recommended publication form: technical report plus reproducible portfolio article. "
            "The contribution is the evidence discipline around a strong benchmark, including the "
            "negative external and attribution findings.",
            style["body"],
        )
    )
    story.append(PageBreak())

    story.append(para("References", style["h1"]))
    references = [
        "1. Ma, W. and Liang, S. POLAR: Posture-level Action Recognition Dataset. Mendeley Data, V1. https://doi.org/10.17632/hvnsh7rwz7.1",
        "2. Ma, W. and Liang, S. POLAR: Posture-level Action Recognition Dataset. ICSAI 2019. https://doi.org/10.1109/ICSAI48974.2019.9010160",
        "3. Oquab, M. et al. DINOv2: Learning Robust Visual Features without Supervision. https://arxiv.org/abs/2304.07193",
        "4. Liu, Z. et al. A ConvNet for the 2020s. https://arxiv.org/abs/2201.03545",
        "5. Gupta, S. and Malik, J. Visual Semantic Role Labeling. https://arxiv.org/abs/1505.04474",
        "6. Sundararajan, M., Taly, A., and Yan, Q. Axiomatic Attribution for Deep Networks. https://arxiv.org/abs/1703.01365",
        "7. Selvaraju, R. R. et al. Grad-CAM. https://arxiv.org/abs/1610.02391",
        "8. Adebayo, J. et al. Sanity Checks for Saliency Maps. https://arxiv.org/abs/1810.03292",
        "9. Petsiuk, V., Das, A., and Saenko, K. RISE. https://arxiv.org/abs/1806.07421",
        "10. Rong, Y. et al. Consistent and Efficient Evaluation of Feature Attribution Methods. https://proceedings.mlr.press/v162/rong22a.html",
    ]
    for reference in references:
        story.append(para(reference, style["reference"]))
    story.append(Spacer(1, 8 * mm))
    story.append(para("Artifact availability", style["h2"]))
    story.append(
        para(
            "Repository: https://github.com/abdullahuseyinli-dot/human-activity-classification<br/>"
            "Code and documentation license: MIT. Dataset images, annotations, and pretrained "
            "weights retain upstream terms.",
            style["body"],
        )
    )

    document.build(story)


def main() -> None:
    args = parse_args()
    build_report(ROOT, args.output.resolve())
    print(f"Wrote technical report PDF: {args.output.resolve()}")


if __name__ == "__main__":
    main()
