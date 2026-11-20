from pathlib import Path
import re
import pdfplumber

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "paper" / "APMGSRN_arXiv_2308.02494.pdf"
OUT = Path(__file__).resolve().parent / "C_ORIGINAL_PAPER_PAGE_COLUMN_TRANSCRIPT.md"

PAGE_INDEX = {
    1: "Title, authors, abstract, index terms, Section 1 Introduction, contribution bullets",
    2: "Figure 1, Figure 2, end of Introduction, Section 2 Related Works, Section 2.1",
    3: "Figure 3, end of Related Works, Section 3 Overview, Section 4 and 4.1, Equation 1",
    4: "Section 4.1.1 and 4.1.2, Equations 2-3, Section 4.2, Equations 4-9, start of Section 4.3",
    5: "Section 4.3 training routine and initialization, Section 5 Domain Decomposition",
    6: "Sections 5.1-5.3 data partitioning, training and inference, Section 6 renderer",
    7: "Table 1 datasets and quantitative results, Section 7 experiments, Section 7.1, Figure 4",
    8: "End of Section 7.1, Figure 5 compression, Section 7.2, Figure 6 ghost cells",
    9: "Section 7.3 rendering, Figures 7-9, Section 8 Limitations, Section 9 Conclusion",
    10: "End of Conclusion, acknowledgments and complete references",
}

VISUAL_INDEX = {
    1: "No numbered figure. Title, abstract and the opening two-column text.",
    2: "Figure 1 shows adaptive grid positions over training for Asteroid, Plume, Supernova and Nyx. Figure 2 compares APMGSRN, fVSRN and NGP architectures.",
    3: "Figure 3 diagrams global-to-local transforms, per-grid interpolation, feature concatenation and MLP decoding.",
    4: "Equations 2 through 9, including encoding, feature density, flat-top Gaussian, target density and KL loss.",
    5: "Training-routine equations and prose; no numbered figure.",
    6: "Domain-decomposition geometry, hash formula and renderer prose; no numbered figure.",
    7: "Table 1 contains all datasets, storage sizes, models, decomposition sizes, PSNRs and training times. Figure 4 compares three grid initializations.",
    8: "Figure 5 compares APMGSRN compression with TTHRESH and SZ3. Figure 6 compares one and sixteen ghost cells.",
    9: "Figure 7 compares Supernova renders. Figure 8 compares the Rotstrat model with downsampled raw data. Figure 9 shows the renderer UI.",
    10: "Acknowledgments and the two-column reference list.",
}

def clean(text):
    if not text:
        return "[No extractable text in this region.]"
    repl = {
        "\ufb01": "fi", "\ufb02": "fl", "\u2212": "-", "\u2010": "-",
        "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "--",
        "\u00a0": " ", "\u00ad": "", "\u0000": "",
    }
    for a, b in repl.items():
        text = text.replace(a, b)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract(page, box):
    return clean(page.crop(box).extract_text(x_tolerance=2, y_tolerance=3, layout=False) or "")

lines = [
    "# Original APMGSRN paper: page-by-page, column-by-column transcript",
    "",
    "**Paper:** Skylar W. Wurster, Tianyu Xiong, Han-Wei Shen, Hanqi Guo, and Tom Peterka, "
    "*Adaptively Placed Multi-Grid Scene Representation Networks for Large-Scale Data Visualization*, IEEE TVCG 30(1), 2024.",
    "",
    "**Source:** `paper/APMGSRN_arXiv_2308.02494.pdf` (10 pages).",
    "",
    "> Purpose: this file preserves physical PDF location. An assistant can answer a question and then cite a precise place such as **page 8, column 2, Effect of ghost cells**.",
    ">",
    "> Text is extracted from the PDF region itself. Line wrapping follows the printed column. Equations, symbols, tables and figure labels can lose some typography during extraction, so use the stated page and column to inspect the PDF visually when exact notation matters.",
    "",
    "## Quick location index",
    "",
]
for p, desc in PAGE_INDEX.items():
    lines.append(f"- **Page {p}:** {desc}")

with pdfplumber.open(PDF) as doc:
    for number, page in enumerate(doc.pages, start=1):
        w, h = float(page.width), float(page.height)
        if number == 1:
            top_end = 300
        elif number == 2:
            top_end = 225
        elif number == 3:
            top_end = 252
        else:
            top_end = 28
        lines += ["", "---", "", f"# PAGE {number}", "", f"**Contents:** {PAGE_INDEX[number]}", "", f"**Visual elements:** {VISUAL_INDEX[number]}", ""]
        if number <= 3:
            lines += [
                "## Full-width content above the columns",
                "",
                "```text",
                extract(page, (38, 24, w - 38, top_end)),
                "```",
                "",
            ]
        lines += [
            "## Column 1 (left)",
            "",
            "```text",
            extract(page, (40, top_end, w / 2 - 3, h - 24)),
            "```",
            "",
            "## Column 2 (right)",
            "",
            "```text",
            extract(page, (w / 2 + 3, top_end, w - 40, h - 24)),
            "```",
        ]

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
