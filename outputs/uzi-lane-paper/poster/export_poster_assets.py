#!/usr/bin/env python3
"""Export reusable poster assets as high-resolution PNG/PDF/SVG files."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


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
        "",
        "Recommended poster workflow: use the AI components as visual backgrounds, add the labeled PNG/PDF figures when you want the exact academic diagram, and use the transparent icons for extra callouts.",
    ]
    (ASSET / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    dirs = ensure_dirs()
    copy_ai_components(dirs["ai"])
    export_pdf_assets(dirs["pdf"], dirs["svg"], dirs["fig_png"])
    export_icons(dirs["icons"])
    write_manifest(dirs)
    print(ASSET)


if __name__ == "__main__":
    main()
