#!/usr/bin/env python3
"""Build a one-page PDF poster for the Uzi project."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
POSTER_DIR = ROOT / "poster"
IMG_DIR = POSTER_DIR / "images"
OUT = POSTER_DIR / "uzi_poster.pdf"

W, H = 16 * 72, 9 * 72

COLORS = {
    "bg": colors.HexColor("#F6F7F2"),
    "ink": colors.HexColor("#1E2329"),
    "muted": colors.HexColor("#56616D"),
    "navy": colors.HexColor("#173B5F"),
    "teal": colors.HexColor("#1B8A7A"),
    "gold": colors.HexColor("#D8A31A"),
    "red": colors.HexColor("#C44E52"),
    "line": colors.HexColor("#D6DDE2"),
    "white": colors.white,
}


def draw_round_rect(c, x, y, w, h, stroke=COLORS["line"], fill=COLORS["white"], radius=8):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(1)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)


def draw_label(c, x, y, text, color=COLORS["teal"]):
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y, text.upper())


def draw_wrapped(c, text, x, y, w, font="Helvetica", size=12, leading=15, color=COLORS["ink"]):
    c.setFont(font, size)
    c.setFillColor(color)
    words = text.split()
    line = ""
    yy = y
    max_chars = max(18, int(w / (size * 0.52)))
    for word in words:
        test = word if not line else line + " " + word
        if len(test) > max_chars:
            c.drawString(x, yy, line)
            yy -= leading
            line = word
        else:
            line = test
    if line:
        c.drawString(x, yy, line)
        yy -= leading
    return yy


def draw_image_fit(c, path, x, y, w, h):
    img = Image.open(path)
    iw, ih = img.size
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    c.drawImage(ImageReader(img), x + (w - dw) / 2, y + (h - dh) / 2, dw, dh, mask="auto")


def bullet(c, x, y, text, color=COLORS["teal"]):
    c.setFillColor(color)
    c.circle(x, y + 4, 3, fill=1, stroke=0)
    return draw_wrapped(c, text, x + 12, y, 240, size=10.5, leading=13, color=COLORS["ink"])


def main() -> None:
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    c.setTitle("Training Uzi Poster")

    c.setFillColor(COLORS["bg"])
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Header
    c.setFillColor(COLORS["navy"])
    c.rect(0, H - 92, W, 92, fill=1, stroke=0)
    c.setFillColor(COLORS["white"])
    c.setFont("Helvetica-Bold", 31)
    c.drawString(34, H - 48, "Training Uzi")
    c.setFont("Helvetica", 15)
    c.drawString(36, H - 73, "PPO for lane-dominant 1v1 Honor of Kings agents")
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(COLORS["gold"])
    c.drawRightString(W - 36, H - 46, "Rank #3  |  131 / 67  |  66.2%")
    c.setFillColor(COLORS["white"])
    c.setFont("Helvetica", 10.5)
    c.drawRightString(W - 36, H - 68, "Uzi-style policy: pressure, spacing, target selection, tower conversion")

    # Column geometry
    margin = 34
    gap = 18
    col_w = (W - 2 * margin - 2 * gap) / 3
    top = H - 112
    bottom = 34
    panel_h = top - bottom
    xs = [margin, margin + col_w + gap, margin + 2 * (col_w + gap)]

    for x in xs:
        draw_round_rect(c, x, bottom, col_w, panel_h)

    # Left: method
    x = xs[0] + 18
    y = top - 30
    draw_label(c, x, y, "Method")
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(COLORS["ink"])
    c.drawString(x, y - 24, "Make semantics agree")
    y -= 52
    y = bullet(c, x, y, "Structured tokens for heroes, towers, minions, bullets, cakes, and global state.")
    y = bullet(c, x, y - 6, "Button-conditioned target pointer: target logits depend on the selected action button.")
    y = bullet(c, x, y - 6, "Action-mask repair removes normal attacks with only None/Self targets.")
    y = bullet(c, x, y - 6, "Rewards emphasize safe trades, last hits, tower pressure, danger avoidance, and small action-quality shaping.")
    draw_image_fit(c, IMG_DIR / "architecture.png", xs[0] + 14, bottom + 40, col_w - 28, 170)

    # Middle: results
    x = xs[1] + 18
    y = top - 30
    draw_label(c, x, y, "Results")
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(COLORS["ink"])
    c.drawString(x, y - 24, "Strong external finish")
    draw_image_fit(c, IMG_DIR / "leaderboard_card.png", xs[1] + 16, top - 184, col_w - 32, 120)
    draw_image_fit(c, IMG_DIR / "training_progress.png", xs[1] + 16, top - 342, col_w - 32, 135)
    draw_image_fit(c, IMG_DIR / "lane_pressure.png", xs[1] + 16, bottom + 42, col_w - 32, 150)

    # Right: case studies
    x = xs[2] + 18
    y = top - 30
    draw_label(c, x, y, "Case studies")
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(COLORS["ink"])
    c.drawString(x, y - 24, "Honest failures explain design")
    y -= 52
    y = bullet(c, x, y, "Target semantics: real probes showed Soldier slots align by runtime id after nearest-four selection.")
    y = bullet(c, x, y - 6, "Recall: reward shaping and exploration could start Button9, but channel completion stayed too sparse; final run disables recall.")
    y = bullet(c, x, y - 6, "Gongsun Li: Arli mark features are observable, but shared lane rewards underfit hero-specific timing.")
    draw_image_fit(c, IMG_DIR / "recall_diagnostics.png", xs[2] + 16, top - 390, col_w - 32, 150)
    draw_image_fit(c, IMG_DIR / "historical_probe_matrix.png", xs[2] + 16, bottom + 36, col_w - 32, 160)

    # Footer
    c.setStrokeColor(COLORS["line"])
    c.line(margin, 22, W - margin, 22)
    c.setFont("Helvetica", 8.5)
    c.setFillColor(COLORS["muted"])
    c.drawString(margin, 10, "Evidence: course leaderboard, logs/ppo-5-1 metrics, git-history synthetic probes, and diag_feature_probes real frames.")
    c.drawRightString(W - margin, 10, "Team Uzi")

    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
