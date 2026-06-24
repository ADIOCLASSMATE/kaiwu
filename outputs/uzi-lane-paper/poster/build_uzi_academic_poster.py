#!/usr/bin/env python3
"""Build the final academic-style Uzi poster."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
POSTER_DIR = ROOT / "poster"
IMG_DIR = POSTER_DIR / "images"
SELECTED = IMG_DIR / "selected"
BACKGROUND = POSTER_DIR / "poster背景高清.png"
OUT = POSTER_DIR / "uzi_academic_poster.pdf"

W, H = 1701, 2551

INK = colors.HexColor("#1f1f1f")
MUTED = colors.HexColor("#4c4c4c")
TEAL = colors.HexColor("#1B8A7A")
NAVY = colors.HexColor("#1F4E79")
GOLD = colors.HexColor("#D8A31A")
LINE = colors.HexColor("#d6d6d6")
WHITE = colors.white

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def draw_wrapped(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    w: float,
    font: str = "Times-Roman",
    size: float = 24,
    leading: float = 33,
    color=INK,
) -> float:
    c.setFont(font, size)
    c.setFillColor(color)
    yy = y
    for paragraph in text.split("\n"):
        words = paragraph.split()
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if stringWidth(candidate, font, size) <= w:
                line = candidate
                continue
            if line:
                c.drawString(x, yy, line)
                yy -= leading
            line = word
        if line:
            c.drawString(x, yy, line)
            yy -= leading
        yy -= leading * 0.30
    return yy


def heading(c: canvas.Canvas, x: float, y: float, label: str) -> float:
    c.setFont("Times-Bold", 35)
    c.setFillColor(INK)
    c.drawString(x, y, label)
    return y - 44


def subheading(c: canvas.Canvas, x: float, y: float, label: str) -> float:
    c.setFont("Times-Bold", 25)
    c.setFillColor(INK)
    c.drawString(x, y, label)
    return y - 33


def bullet(c: canvas.Canvas, x: float, y: float, text: str, w: float, size: float = 24) -> float:
    c.setFillColor(INK)
    c.setFont("Times-Roman", size)
    c.drawString(x, y, "\u2022")
    return draw_wrapped(c, text, x + 26, y, w - 26, size=size, leading=size * 1.35)


def draw_image_fit(
    c: canvas.Canvas,
    path: Path,
    x: float,
    y: float,
    w: float,
    h: float,
    border: bool = False,
) -> tuple[float, float]:
    img = Image.open(path)
    iw, ih = img.size
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    xx = x + (w - dw) / 2
    yy = y + (h - dh) / 2
    c.drawImage(ImageReader(img), xx, yy, width=dw, height=dh, mask="auto")
    if border:
        c.setStrokeColor(LINE)
        c.setLineWidth(0.8)
        c.rect(xx, yy, dw, dh, stroke=1, fill=0)
    return dw, dh


def caption(c: canvas.Canvas, x: float, y: float, text: str, w: float) -> float:
    return draw_wrapped(c, text, x, y, w, font="Times-Roman", size=18, leading=23, color=MUTED)


def draw_title(c: canvas.Canvas) -> None:
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 57)
    c.drawString(108, H - 300, "Training Uzi: PPO for Lane-Dominant")
    c.drawString(108, H - 375, "1v1 Honor of Kings Agents")
    c.setFont("Times-Bold", 31)
    c.drawString(108, H - 435, "Team Uzi")
    c.setFont("STSong-Light", 25)
    c.setFillColor(colors.HexColor("#EAF7F5"))
    c.drawString(108, H - 485, "万佳欣 253108030048    刘悦 253108030049")


def draw_footer(c: canvas.Canvas) -> None:
    c.setFillColor(WHITE)
    c.setFont("Times-Bold", 27)
    c.drawString(106, 58, "Shanghai Innovation Institute")
    c.drawCentredString(W / 2, 58, "Team Uzi")
    c.setFont("Helvetica-Bold", 24)
    c.drawRightString(W - 106, 72, "Rank #3  |  131 / 67  |  66.2%")
    c.setFont("Helvetica", 18)
    c.drawRightString(W - 106, 42, "course leaderboard aggregate win rate")


def main() -> None:
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    c.setTitle("Training Uzi Academic Poster")
    c.drawImage(ImageReader(str(BACKGROUND)), 0, 0, width=W, height=H)
    draw_title(c)

    left_x, left_w = 106, 725
    right_x, right_w = 875, 715
    top_y = 1930

    # Left column.
    y = heading(c, left_x, top_y, "Abstract")
    abstract = (
        "We train Uzi, a PPO-based 1v1 Honor of Kings agent, under a lane-dominant strategy. "
        "The system converts raw frame state into canonical entity tokens, repairs target/action "
        "semantics, and uses a pointer-style policy to choose buttons and entity targets. "
        "The final agent emphasizes reliable lane pressure: last hits, safe trades, valid "
        "targets, tower conversion, and external match win rate."
    )
    y = draw_wrapped(c, abstract, left_x, y, left_w, size=25, leading=34)

    y -= 10
    y = heading(c, left_x, y, "Motivation")
    for item in [
        "Raw game state mixes heroes, towers, minions, bullets, cakes, and transient visibility.",
        "Action buttons and entity targets must agree; otherwise the policy learns invalid choices.",
        "Sparse win/loss feedback is delayed, so reward shaping must expose lane pressure early.",
        "Negative findings, including recall and hero-specific timing, are kept as design evidence.",
    ]:
        y = bullet(c, left_x, y, item, left_w, size=23)
        y -= 4

    y -= 8
    y = heading(c, left_x, y, "Method")
    draw_image_fit(c, SELECTED / "feature_pipeline.png", left_x - 6, y - 405, left_w + 12, 390, border=True)
    y -= 424
    y = caption(
        c,
        left_x + 26,
        y,
        "Figure 1: raw frame state is canonicalized into stable tokens and target slots, producing a 561-dim policy input.",
        left_w - 52,
    )
    y -= 4
    method_tail = (
        "The policy then encodes tokens, carries temporal state through an LSTM, and predicts button, "
        "target, and value heads. Target logits are conditioned on the selected button, matching the "
        "game API rather than treating actions as independent labels."
    )
    y = draw_wrapped(c, method_tail, left_x, y, left_w, size=24, leading=32)
    draw_image_fit(c, SELECTED / "policy_architecture.png", left_x, 232, left_w, 315, border=True)
    caption(
        c,
        left_x + 26,
        203,
        "Policy architecture: token encoder and LSTM memory feed pointer heads for buttons, targets, and value.",
        left_w - 52,
    )

    # Right column.
    y = heading(c, right_x, top_y, "Experiment")
    y = subheading(c, right_x, y, "Setup:")
    for item in [
        "Environment: 1v1 Honor of Kings lane task.",
        "Policy: token encoder, LSTM memory, pointer button and target heads.",
        "Training: PPO with curriculum stages, self-play/common-AI mix, and reward shaping.",
        "Evidence: course leaderboard, local logs, real frame probes, and synthetic history probes.",
    ]:
        y = bullet(c, right_x, y, item, right_w, size=22)
        y -= 2

    y -= 6
    y = subheading(c, right_x, y, "Main Result:")
    draw_image_fit(c, SELECTED / "leaderboard_card.png", right_x + 10, y - 205, 305, 188, border=False)
    result_text = (
        "Uzi finished Rank #3 with 131 wins and 67 losses over 198 games, for a 66.2% aggregate win rate."
    )
    draw_wrapped(c, result_text, right_x + 345, y - 25, right_w - 345, size=22, leading=30)
    y -= 230

    y = subheading(c, right_x, y, "Training Curriculum:")
    draw_image_fit(c, SELECTED / "training_curriculum.png", right_x - 2, y - 330, right_w + 4, 305, border=True)
    y -= 352
    y = caption(
        c,
        right_x + 24,
        y,
        "Figure 2: staged training moves from basic control to lane aggression and final win conversion.",
        right_w - 48,
    )

    y -= 2
    y = subheading(c, right_x, y, "Diagnostics:")
    draw_image_fit(c, SELECTED / "lane_pressure.png", right_x, y - 215, right_w, 205, border=True)
    y -= 237
    y = caption(
        c,
        right_x + 24,
        y,
        "Figure 3: lower enemy tower HP and positive kill-death gap indicate pressure conversion.",
        right_w - 48,
    )
    draw_image_fit(c, SELECTED / "historical_probe_matrix.png", right_x, 257, right_w, 250, border=True)
    caption(
        c,
        right_x + 24,
        230,
        "Figure 4: probe matrix tracks semantic fixes and remaining limitations across code history.",
        right_w - 48,
    )

    draw_footer(c)
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
