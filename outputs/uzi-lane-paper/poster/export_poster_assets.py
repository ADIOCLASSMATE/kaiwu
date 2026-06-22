#!/usr/bin/env python3
"""Export reusable poster assets as high-resolution PNG/PDF/SVG files."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
AI = ROOT / "generated-components"
ASSET = ROOT / "poster" / "assets"

CORE_FIGURES = [
    "feature_pipeline",
    "architecture",
    "training_curriculum",
]
DATA_FIGURES = [
    "training_progress",
    "lane_pressure",
    "leaderboard_card",
    "recall_diagnostics",
    "historical_probe_matrix",
]
AI_COMPONENTS = [
    "feature_processing_ai",
    "policy_architecture_ai",
    "training_curriculum_ai",
    "uzi_mascot_ai",
]


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "ai": ASSET / "ai_components_png",
        "fig_png": ASSET / "labeled_figures_png",
        "pdf": ASSET / "vector_pdf",
        "svg": ASSET / "vector_svg",
        "icons": ASSET / "icons_png_transparent",
        "icons_pdf": ASSET / "icons_pdf_vector",
        "ai_pdf": ASSET / "ai_components_pdf",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT.parents[1])


def copy_ai_components(dst: Path) -> None:
    for name in AI_COMPONENTS:
        src = AI / f"{name}.png"
        if not src.exists():
            continue
        shutil.copy2(src, dst / f"{name}_original.png")
        with Image.open(src) as im:
            im = im.convert("RGBA")
            target_w = 3840 if name != "uzi_mascot_ai" else 2048
            scale = target_w / im.width
            target_h = round(im.height * scale)
            up = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
            up.save(dst / f"{name}_highres.png")


def export_ai_component_pdfs(dst: Path) -> None:
    """Wrap raster AI components in PDF containers.

    These are useful for PPT workflows that prefer PDF import, but they are not
    true vector artwork because the source was generated as raster images.
    """
    for name in AI_COMPONENTS:
        src = AI / f"{name}.png"
        if not src.exists():
            continue
        with Image.open(src) as im:
            w_px, h_px = im.size
        width = 10.0 * 72
        height = width * h_px / w_px
        out = dst / f"{name}_raster_embedded.pdf"
        c = canvas.Canvas(str(out), pagesize=(width, height))
        c.drawImage(ImageReader(str(src)), 0, 0, width=width, height=height, mask="auto")
        c.save()


def export_pdf_assets(pdf_dst: Path, svg_dst: Path, png_dst: Path) -> None:
    for name in CORE_FIGURES + DATA_FIGURES:
        src = FIG / f"{name}.pdf"
        if not src.exists():
            continue
        shutil.copy2(src, pdf_dst / f"{name}.pdf")
        run([
            "pdftoppm",
            "-png",
            "-r",
            "600",
            "-singlefile",
            str(src),
            str(png_dst / f"{name}_600dpi"),
        ])
        try:
            run(["pdftocairo", "-svg", str(src), str(svg_dst / f"{name}.svg")])
        except subprocess.CalledProcessError:
            pass


def draw_icon(kind: str, size: int = 1024) -> Image.Image:
    s = size
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def xy(vals):
        return [round(v * s) for v in vals]

    navy = (24, 70, 118, 255)
    teal = (20, 139, 125, 255)
    red = (196, 78, 82, 255)
    gold = (218, 163, 26, 255)
    green = (76, 154, 42, 255)
    blue = (60, 148, 226, 255)
    grey = (228, 234, 238, 255)
    dark = (31, 36, 42, 255)
    white = (255, 255, 255, 255)

    enemy = kind.endswith("_enemy")
    base = red if enemy else teal

    if kind.startswith("hero"):
        d.ellipse(xy([0.30, 0.12, 0.70, 0.52]), fill=white, outline=base, width=18)
        d.polygon([tuple(xy([0.31, 0.22])), tuple(xy([0.50, 0.06])), tuple(xy([0.69, 0.22])), tuple(xy([0.60, 0.18])), tuple(xy([0.50, 0.27])), tuple(xy([0.40, 0.18]))], fill=navy if not enemy else dark)
        d.ellipse(xy([0.40, 0.30, 0.44, 0.34]), fill=dark)
        d.ellipse(xy([0.56, 0.30, 0.60, 0.34]), fill=dark)
        d.arc(xy([0.42, 0.34, 0.58, 0.45]), 10, 170, fill=base, width=8)
        d.rounded_rectangle(xy([0.32, 0.52, 0.68, 0.82]), radius=55, fill=base, outline=dark, width=8)
        d.rectangle(xy([0.23, 0.56, 0.35, 0.66]), fill=base)
        d.rectangle(xy([0.65, 0.56, 0.77, 0.66]), fill=base)
        sword = gold if not enemy else red
        d.polygon([tuple(xy([0.73, 0.28])), tuple(xy([0.90, 0.20])), tuple(xy([0.82, 0.38]))], fill=sword, outline=dark)
        d.line(xy([0.67, 0.55, 0.82, 0.34]), fill=dark, width=10)
    elif kind.startswith("minion"):
        d.ellipse(xy([0.32, 0.16, 0.68, 0.52]), fill=white, outline=base, width=16)
        d.rectangle(xy([0.25, 0.20, 0.75, 0.32]), fill=base)
        d.ellipse(xy([0.42, 0.34, 0.46, 0.38]), fill=dark)
        d.ellipse(xy([0.54, 0.34, 0.58, 0.38]), fill=dark)
        d.rounded_rectangle(xy([0.30, 0.50, 0.70, 0.82]), radius=48, fill=base, outline=dark, width=8)
        d.rectangle(xy([0.22, 0.58, 0.31, 0.70]), fill=base)
        d.rectangle(xy([0.69, 0.58, 0.78, 0.70]), fill=base)
    elif kind.startswith("tower"):
        color = red if enemy else navy
        d.polygon([tuple(xy([0.22, 0.84])), tuple(xy([0.78, 0.84])), tuple(xy([0.68, 0.24])), tuple(xy([0.32, 0.24]))], fill=grey, outline=color)
        for x in (0.28, 0.44, 0.60):
            d.rectangle(xy([x, 0.16, x + 0.12, 0.29]), fill=grey, outline=color, width=10)
        d.polygon([tuple(xy([0.50, 0.02])), tuple(xy([0.62, 0.17])), tuple(xy([0.50, 0.31])), tuple(xy([0.38, 0.17]))], fill=(83, 183, 255, 255) if not enemy else red, outline=color)
        d.rounded_rectangle(xy([0.40, 0.60, 0.60, 0.84]), radius=22, fill=color)
    elif kind == "bullet":
        d.polygon([tuple(xy([0.82, 0.50])), tuple(xy([0.36, 0.25])), tuple(xy([0.18, 0.50])), tuple(xy([0.36, 0.75]))], fill=gold, outline=dark)
        d.line(xy([0.18, 0.50, 0.04, 0.50]), fill=gold, width=26)
        d.line(xy([0.30, 0.37, 0.09, 0.25]), fill=(255, 205, 90, 210), width=18)
        d.line(xy([0.30, 0.63, 0.09, 0.75]), fill=(255, 205, 90, 210), width=18)
    elif kind == "cake":
        d.ellipse(xy([0.18, 0.54, 0.82, 0.84]), fill=(229, 196, 122, 255), outline=gold, width=14)
        d.rounded_rectangle(xy([0.20, 0.36, 0.80, 0.68]), radius=55, fill=(250, 235, 198, 255), outline=gold, width=12)
        d.ellipse(xy([0.36, 0.12, 0.64, 0.40]), fill=(122, 211, 102, 255), outline=green, width=10)
        d.rectangle(xy([0.47, 0.18, 0.53, 0.33]), fill=white)
        d.rectangle(xy([0.42, 0.235, 0.58, 0.295]), fill=white)
    elif kind == "monster":
        d.rounded_rectangle(xy([0.18, 0.28, 0.82, 0.76]), radius=120, fill=(118, 139, 156, 255), outline=navy, width=14)
        d.ellipse(xy([0.35, 0.18, 0.65, 0.48]), fill=blue, outline=navy, width=10)
        d.ellipse(xy([0.39, 0.23, 0.61, 0.43]), fill=(85, 205, 255, 255))
        d.ellipse(xy([0.30, 0.50, 0.36, 0.56]), fill=white)
        d.ellipse(xy([0.64, 0.50, 0.70, 0.56]), fill=white)
        for x in (0.10, 0.80):
            d.rounded_rectangle(xy([x, 0.50, x + 0.16, 0.78]), radius=50, fill=(90, 110, 126, 255), outline=navy, width=8)
    elif kind == "target_pointer":
        d.ellipse(xy([0.20, 0.20, 0.80, 0.80]), outline=gold, width=28)
        d.ellipse(xy([0.43, 0.43, 0.57, 0.57]), fill=gold)
        d.line(xy([0.50, 0.05, 0.50, 0.28]), fill=gold, width=22)
        d.line(xy([0.50, 0.72, 0.50, 0.95]), fill=gold, width=22)
        d.line(xy([0.05, 0.50, 0.28, 0.50]), fill=gold, width=22)
        d.line(xy([0.72, 0.50, 0.95, 0.50]), fill=gold, width=22)
    elif kind == "token_card":
        d.rounded_rectangle(xy([0.12, 0.18, 0.88, 0.82]), radius=70, fill=(245, 251, 252, 235), outline=teal, width=16)
        for i, color in enumerate([teal, blue, red, gold]):
            x = 0.24 + i * 0.13
            d.rounded_rectangle(xy([x, 0.43, x + 0.08, 0.55]), radius=14, fill=color)
    else:
        d.ellipse(xy([0.2, 0.2, 0.8, 0.8]), fill=base)

    return im


def export_icons(dst: Path) -> None:
    names = [
        "hero_friendly",
        "hero_enemy",
        "minion_friendly",
        "minion_enemy",
        "tower_friendly",
        "tower_enemy",
        "bullet",
        "cake",
        "monster",
        "target_pointer",
        "token_card",
    ]
    icons = []
    for name in names:
        icon = draw_icon(name, 1024)
        path = dst / f"{name}.png"
        icon.save(path)
        icons.append((name, icon))

    sheet = Image.new("RGBA", (4096, 3072), (255, 255, 255, 0))
    for idx, (_, icon) in enumerate(icons):
        x = (idx % 4) * 1024
        y = (idx // 4) * 1024
        sheet.alpha_composite(icon, (x, y))
    sheet.save(dst / "icon_sheet_4096_transparent.png")


def _pdf_color(rgb):
    r, g, b = rgb[:3]
    return colors.Color(r / 255, g / 255, b / 255)


def _path_polygon(c: canvas.Canvas, pts, fill, stroke=None, width=2):
    p = c.beginPath()
    p.moveTo(*pts[0])
    for pt in pts[1:]:
        p.lineTo(*pt)
    p.close()
    c.setFillColor(fill)
    if stroke is not None:
        c.setStrokeColor(stroke)
        c.setLineWidth(width)
        c.drawPath(p, fill=1, stroke=1)
    else:
        c.drawPath(p, fill=1, stroke=0)


def draw_vector_icon(c: canvas.Canvas, kind: str, s: float = 144.0) -> None:
    """Draw one icon as vector primitives on an s x s PDF page."""
    navy = _pdf_color((24, 70, 118))
    teal = _pdf_color((20, 139, 125))
    red = _pdf_color((196, 78, 82))
    gold = _pdf_color((218, 163, 26))
    green = _pdf_color((76, 154, 42))
    blue = _pdf_color((60, 148, 226))
    grey = _pdf_color((228, 234, 238))
    dark = _pdf_color((31, 36, 42))
    white = colors.white
    enemy = kind.endswith("_enemy")
    base = red if enemy else teal

    def r(vals):
        return [v * s for v in vals]

    def ellipse(vals, fill, stroke=None, width=2):
        x1, y1, x2, y2 = r(vals)
        c.setFillColor(fill)
        c.setStrokeColor(stroke or fill)
        c.setLineWidth(width)
        c.ellipse(x1, y1, x2, y2, fill=1, stroke=1 if stroke else 0)

    def rect(vals, fill, stroke=None, width=2):
        x1, y1, x2, y2 = r(vals)
        c.setFillColor(fill)
        c.setStrokeColor(stroke or fill)
        c.setLineWidth(width)
        c.rect(x1, y1, x2 - x1, y2 - y1, fill=1, stroke=1 if stroke else 0)

    def round_rect(vals, radius, fill, stroke=None, width=2):
        x1, y1, x2, y2 = r(vals)
        c.setFillColor(fill)
        c.setStrokeColor(stroke or fill)
        c.setLineWidth(width)
        c.roundRect(x1, y1, x2 - x1, y2 - y1, radius * s, fill=1, stroke=1 if stroke else 0)

    if kind.startswith("hero"):
        ellipse([0.30, 0.48, 0.70, 0.88], white, base, 3)
        _path_polygon(c, [(0.31*s, 0.78*s), (0.50*s, 0.94*s), (0.69*s, 0.78*s), (0.60*s, 0.82*s), (0.50*s, 0.73*s), (0.40*s, 0.82*s)], navy if not enemy else dark)
        ellipse([0.40, 0.66, 0.44, 0.70], dark)
        ellipse([0.56, 0.66, 0.60, 0.70], dark)
        c.setStrokeColor(base); c.setLineWidth(3); c.arc(*r([0.42, 0.55, 0.58, 0.66]), 190, 160)
        round_rect([0.32, 0.18, 0.68, 0.48], 0.05, base, dark, 2)
        rect([0.23, 0.34, 0.35, 0.44], base)
        rect([0.65, 0.34, 0.77, 0.44], base)
        sword = gold if not enemy else red
        _path_polygon(c, [(0.73*s, 0.72*s), (0.90*s, 0.80*s), (0.82*s, 0.62*s)], sword, dark, 1.5)
        c.setStrokeColor(dark); c.setLineWidth(2.5); c.line(0.67*s, 0.45*s, 0.82*s, 0.66*s)
    elif kind.startswith("minion"):
        ellipse([0.32, 0.48, 0.68, 0.84], white, base, 3)
        rect([0.25, 0.68, 0.75, 0.80], base)
        ellipse([0.42, 0.62, 0.46, 0.66], dark)
        ellipse([0.54, 0.62, 0.58, 0.66], dark)
        round_rect([0.30, 0.18, 0.70, 0.50], 0.05, base, dark, 2)
        rect([0.22, 0.30, 0.31, 0.42], base)
        rect([0.69, 0.30, 0.78, 0.42], base)
    elif kind.startswith("tower"):
        color = red if enemy else navy
        _path_polygon(c, [(0.22*s, 0.16*s), (0.78*s, 0.16*s), (0.68*s, 0.76*s), (0.32*s, 0.76*s)], grey, color, 2)
        for x in (0.28, 0.44, 0.60):
            rect([x, 0.71, x + 0.12, 0.84], grey, color, 2)
        _path_polygon(c, [(0.50*s, 0.98*s), (0.62*s, 0.83*s), (0.50*s, 0.69*s), (0.38*s, 0.83*s)], blue if not enemy else red, color, 2)
        round_rect([0.40, 0.16, 0.60, 0.40], 0.03, color)
    elif kind == "bullet":
        _path_polygon(c, [(0.82*s, 0.50*s), (0.36*s, 0.75*s), (0.18*s, 0.50*s), (0.36*s, 0.25*s)], gold, dark, 2)
        c.setStrokeColor(gold); c.setLineWidth(5); c.line(0.18*s, 0.50*s, 0.04*s, 0.50*s)
    elif kind == "cake":
        ellipse([0.18, 0.16, 0.82, 0.46], _pdf_color((229, 196, 122)), gold, 2)
        round_rect([0.20, 0.32, 0.80, 0.64], 0.06, _pdf_color((250, 235, 198)), gold, 2)
        ellipse([0.36, 0.60, 0.64, 0.88], _pdf_color((122, 211, 102)), green, 2)
        rect([0.47, 0.67, 0.53, 0.82], white)
        rect([0.42, 0.705, 0.58, 0.765], white)
    elif kind == "monster":
        round_rect([0.18, 0.24, 0.82, 0.72], 0.10, _pdf_color((118, 139, 156)), navy, 2)
        ellipse([0.35, 0.52, 0.65, 0.82], blue, navy, 2)
        ellipse([0.39, 0.57, 0.61, 0.77], _pdf_color((85, 205, 255)))
        ellipse([0.30, 0.44, 0.36, 0.50], white)
        ellipse([0.64, 0.44, 0.70, 0.50], white)
        round_rect([0.10, 0.22, 0.26, 0.50], 0.04, _pdf_color((90, 110, 126)), navy, 2)
        round_rect([0.80, 0.22, 0.96, 0.50], 0.04, _pdf_color((90, 110, 126)), navy, 2)
    elif kind == "target_pointer":
        ellipse([0.20, 0.20, 0.80, 0.80], colors.Color(1, 1, 1, alpha=0), gold, 5)
        ellipse([0.43, 0.43, 0.57, 0.57], gold)
        c.setStrokeColor(gold); c.setLineWidth(5)
        c.line(0.50*s, 0.05*s, 0.50*s, 0.28*s)
        c.line(0.50*s, 0.72*s, 0.50*s, 0.95*s)
        c.line(0.05*s, 0.50*s, 0.28*s, 0.50*s)
        c.line(0.72*s, 0.50*s, 0.95*s, 0.50*s)
    elif kind == "token_card":
        round_rect([0.12, 0.18, 0.88, 0.82], 0.07, _pdf_color((245, 251, 252)), teal, 3)
        for i, color in enumerate([teal, blue, red, gold]):
            x = 0.24 + i * 0.13
            round_rect([x, 0.43, x + 0.08, 0.55], 0.015, color)


def export_icon_pdfs(dst: Path) -> None:
    names = [
        "hero_friendly", "hero_enemy", "minion_friendly", "minion_enemy",
        "tower_friendly", "tower_enemy", "bullet", "cake", "monster",
        "target_pointer", "token_card",
    ]
    page = 144.0
    for name in names:
        out = dst / f"{name}.pdf"
        c = canvas.Canvas(str(out), pagesize=(page, page))
        draw_vector_icon(c, name, page)
        c.save()

    sheet = dst / "icon_sheet_vector.pdf"
    c = canvas.Canvas(str(sheet), pagesize=(page * 4, page * 3))
    for idx, name in enumerate(names):
        x = (idx % 4) * page
        y = (2 - idx // 4) * page
        c.saveState()
        c.translate(x, y)
        draw_vector_icon(c, name, page)
        c.restoreState()
    c.save()


def write_manifest(dirs: dict[str, Path]) -> None:
    lines = [
        "# Uzi Poster Asset Manifest",
        "",
        "This folder contains reusable high-resolution assets for manually composing the poster.",
        "",
        "## AI Components",
        "",
        "- `ai_components_png/*_original.png`: original image-generation components.",
        "- `ai_components_png/*_highres.png`: upscaled high-resolution versions for PPT placement.",
        "",
        "## Labeled Figures",
        "",
        "- `labeled_figures_png/*_600dpi.png`: high-resolution PNG exports of the paper figures.",
        "- `vector_pdf/*.pdf`: PDF figure sources. Data-only figures are vector; mixed figures include embedded AI raster components plus vector labels.",
        "- `vector_svg/*.svg`: SVG exports where conversion succeeded.",
        "",
        "## Icons",
        "",
        "- `icons_png_transparent/*.png`: 1024px transparent standalone icons.",
        "- `icons_png_transparent/icon_sheet_4096_transparent.png`: transparent icon sheet.",
        "- `icons_pdf_vector/*.pdf`: true vector PDF standalone icons and a vector icon sheet.",
        "",
        "## PDF-only Workflow",
        "",
        "- For exact academic diagrams, use `vector_pdf/*.pdf`.",
        "- For icons, use `icons_pdf_vector/*.pdf`; these are true vector primitives.",
        "- For AI-generated illustrations, use `ai_components_pdf/*_raster_embedded.pdf`. These are PDFs for convenience, but the source art is still raster embedded inside the PDF.",
        "",
        "Recommended poster workflow: use the AI components as visual backgrounds, add the labeled PNG/PDF figures when you want the exact academic diagram, and use the transparent icons for extra callouts.",
    ]
    (ASSET / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    dirs = ensure_dirs()
    copy_ai_components(dirs["ai"])
    export_ai_component_pdfs(dirs["ai_pdf"])
    export_pdf_assets(dirs["pdf"], dirs["svg"], dirs["fig_png"])
    export_icons(dirs["icons"])
    export_icon_pdfs(dirs["icons_pdf"])
    write_manifest(dirs)
    print(ASSET)


if __name__ == "__main__":
    main()
