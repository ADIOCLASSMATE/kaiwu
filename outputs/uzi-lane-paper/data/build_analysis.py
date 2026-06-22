#!/usr/bin/env python3
"""Build analysis tables and vector figures for the Uzi mini paper."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "outputs" / "uzi-lane-paper"
DATA_DIR = OUT_ROOT / "data"
FIG_DIR = OUT_ROOT / "figures"
AI_DIR = OUT_ROOT / "generated-components"
PROBE_RESULTS = OUT_ROOT / "probes" / "results" / "historical_probe_results.json"

PALETTE = {
    "navy": colors.HexColor("#1F4E79"),
    "teal": colors.HexColor("#1B8A7A"),
    "red": colors.HexColor("#C44E52"),
    "gold": colors.HexColor("#D8A31A"),
    "gray": colors.HexColor("#666666"),
    "light_gray": colors.HexColor("#E8ECEF"),
    "ink": colors.HexColor("#20242A"),
    "green": colors.HexColor("#4C9A2A"),
    "purple": colors.HexColor("#6B5CA5"),
}

LEADERBOARD = {
    "team": "Uzi",
    "rank": 3,
    "a_wins": 131,
    "b_wins": 67,
    "games": 198,
}
LEADERBOARD["win_rate"] = LEADERBOARD["a_wins"] / LEADERBOARD["games"]


def get_nested(data: dict, path: tuple[str, str]) -> float | None:
    value = data.get("latest", {}).get(path[0], {}).get(path[1])
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_run_summary() -> pd.DataFrame:
    rows = []
    for path in sorted((REPO_ROOT / "logs").glob("ppo-*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        latest = data.get("latest", {})
        row = {
            "run": data.get("task_name", path.parent.name),
            "task_id": data.get("task_id"),
            "status": data.get("task_status"),
            "start": data.get("time_range", {}).get("start"),
            "end": data.get("time_range", {}).get("end"),
            "metrics_available": data.get("total_metrics_available"),
            "train_global_step": get_nested(data, ("training_basic", "train_global_step")),
            "episode_cnt": get_nested(data, ("training_basic", "episode_cnt")),
            "reward": get_nested(data, ("algorithm", "reward")),
            "total_loss": get_nested(data, ("algorithm", "total_loss")),
            "value_loss": get_nested(data, ("algorithm", "value_loss")),
            "policy_loss": get_nested(data, ("algorithm", "policy_loss")),
            "entropy_loss": get_nested(data, ("algorithm", "entropy_loss")),
            "grad_norm": get_nested(data, ("algorithm", "grad_norm")),
            "win_rate": get_nested(data, ("env", "win_rate")),
            "win": get_nested(data, ("env", "win")),
            "kill": get_nested(data, ("env", "kill")),
            "death": get_nested(data, ("env", "death")),
            "money_per_frame": get_nested(data, ("env", "money_per_frame")),
            "hurt_to_hero": get_nested(data, ("env", "hurt_to_hero")),
            "hurt_by_hero": get_nested(data, ("env", "hurt_by_hero")),
            "self_tower_hp": get_nested(data, ("env", "self_tower_hp")),
            "enemy_tower_hp": get_nested(data, ("env", "enemy_tower_hp")),
            "final_hp_ratio": get_nested(data, ("env", "final_hp_ratio")),
            "episode_len": get_nested(data, ("env", "episode_len")),
            "rwd_tower_hp_point": get_nested(data, ("reward_items", "rwd_tower_hp_point")),
            "rwd_hp_point": get_nested(data, ("reward_items", "rwd_hp_point")),
            "rwd_last_hit": get_nested(data, ("reward_items", "rwd_last_hit")),
            "rwd_tower_attack": get_nested(data, ("reward_items", "rwd_tower_attack")),
            "rwd_recall_recover": get_nested(data, ("reward_items", "rwd_recall_recover")),
            "action_button_9": get_nested(data, ("action", "action_button_9")),
            "action_button_9_rate": get_nested(data, ("action", "action_button_9_rate")),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("start").reset_index(drop=True)
        df.insert(0, "stage_index", range(1, len(df) + 1))
    return df


def collect_final_run_note() -> dict:
    summary_path = REPO_ROOT / "logs" / "ppo-5-1" / "summary.json"
    error_path = REPO_ROOT / "logs" / "ppo-5-1" / "errors.jsonl.summary.json"
    legacy_error_path = REPO_ROOT / "logs" / "ppo-5-1-errors.jsonl.summary.json"
    if not summary_path.exists() and not error_path.exists() and not legacy_error_path.exists():
        return {"available": False}
    data = json.loads(error_path.read_text(encoding="utf-8")) if error_path.exists() else (
        json.loads(legacy_error_path.read_text(encoding="utf-8")) if legacy_error_path.exists() else {}
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {
        "available": True,
        "task_id": summary.get("task_id", data.get("task_id")),
        "task_name": summary.get("task_name", data.get("task_name")),
        "status": summary.get("task_status"),
        "start": summary.get("time_range", {}).get("start", data.get("start")),
        "end": summary.get("time_range", {}).get("end", data.get("end")),
        "summary_path": str(summary_path.relative_to(REPO_ROOT)) if summary_path.exists() else None,
        "error_summary_path": str((error_path if error_path.exists() else legacy_error_path).relative_to(REPO_ROOT)),
        "error_warning_entries": data.get("entries"),
        "levels": data.get("levels"),
    }


def probe_table() -> pd.DataFrame:
    if not PROBE_RESULTS.exists():
        return pd.DataFrame()
    data = json.loads(PROBE_RESULTS.read_text(encoding="utf-8"))
    rows = []
    for item in data.get("results", []):
        probes = item.get("probes", {})
        mask = probes.get("normal_attack_mask", {})
        order = probes.get("soldier_target_order", {})
        recall = probes.get("recall_low_hp_safe", {})
        arli = probes.get("arli_mark_feature", {})
        rows.append(
            {
                "label": item.get("label"),
                "commit": (item.get("commit") or item.get("short_commit") or "")[:7],
                "kind": item.get("kind"),
                "dirty": item.get("dirty", False),
                "mask_available": mask.get("available", False),
                "button3_after": mask.get("button3_after"),
                "soldier_order_available": order.get("available", False),
                "soldier_slots": ",".join(map(str, order.get("observed_slot_runtime_ids", []))),
                "recall_available": recall.get("available", False),
                "recall_recover": recall.get("recall_recover"),
                "recall_need_cnt": (recall.get("stats") or {}).get("recall_need_cnt"),
                "arli_available": arli.get("available", False),
                "arli_has_mark_field": arli.get("has_arli_mark_field", False),
                "arli_nonzero_marks": sum(1 for v in arli.get("mark_values", []) if v.get("nonzero")),
                "error_count": len(item.get("errors", [])),
            }
        )
    return pd.DataFrame(rows)


def write_csvs(run_df: pd.DataFrame, probe_df: pd.DataFrame, final_note: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    run_df.to_csv(DATA_DIR / "ppo_run_summary.csv", index=False)
    if not probe_df.empty:
        probe_df.to_csv(DATA_DIR / "historical_probe_summary.csv", index=False)
    (DATA_DIR / "leaderboard.json").write_text(
        json.dumps(LEADERBOARD, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (DATA_DIR / "final_run_note.json").write_text(
        json.dumps(final_note, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "leaderboard": LEADERBOARD,
        "final_run_note": final_note,
        "best_logged_win_rate": None if run_df.empty else float(run_df["win_rate"].max()),
        "best_logged_reward": None if run_df.empty else float(run_df["reward"].max()),
        "latest_logged_run": None if run_df.empty else run_df.iloc[-1].to_dict(),
    }
    (DATA_DIR / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def draw_axes(c: canvas.Canvas, x: float, y: float, w: float, h: float, title: str, y_label: str) -> None:
    c.setStrokeColor(PALETTE["light_gray"])
    for i in range(5):
        yy = y + h * i / 4
        c.line(x, yy, x + w, yy)
    c.setStrokeColor(PALETTE["ink"])
    c.setLineWidth(1)
    c.line(x, y, x, y + h)
    c.line(x, y, x + w, y)
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(PALETTE["ink"])
    c.drawString(x, y + h + 13, title)
    c.setFont("Helvetica", 7)
    c.setFillColor(PALETTE["gray"])
    c.drawString(x, y + h + 2, y_label)


def draw_line(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    values: list[float | None],
    color,
    y_min: float | None = None,
    y_max: float | None = None,
    marker: bool = True,
) -> None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return
    y_min = min(clean) if y_min is None else y_min
    y_max = max(clean) if y_max is None else y_max
    if abs(y_max - y_min) < 1e-9:
        y_min -= 1
        y_max += 1
    points = []
    n = len(values)
    for i, v in enumerate(values):
        if v is None or not math.isfinite(v):
            points.append(None)
            continue
        px = x + (w * i / max(1, n - 1))
        py = y + (h * (v - y_min) / (y_max - y_min))
        points.append((px, py))
    c.setStrokeColor(color)
    c.setLineWidth(2)
    last = None
    for pt in points:
        if pt is not None and last is not None:
            c.line(last[0], last[1], pt[0], pt[1])
        last = pt
    if marker:
        c.setFillColor(color)
        for pt in points:
            if pt is not None:
                c.circle(pt[0], pt[1], 2.1, fill=1, stroke=0)


def draw_legend(c: canvas.Canvas, x: float, y: float, items: list[tuple[str, object]]) -> None:
    c.setFont("Helvetica", 7.5)
    for label, color in items:
        c.setStrokeColor(color)
        c.setLineWidth(2)
        c.line(x, y, x + 12, y)
        c.setFillColor(PALETTE["ink"])
        c.drawString(x + 16, y - 2.5, label)
        x += 70


def safe_series(df: pd.DataFrame, col: str) -> list[float | None]:
    if col not in df:
        return []
    out = []
    for value in df[col].tolist():
        if pd.isna(value):
            out.append(None)
        else:
            out.append(float(value))
    return out


def create_training_progress(run_df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "training_progress.pdf"
    c = canvas.Canvas(str(path), pagesize=(7.0 * inch, 3.0 * inch))
    c.setTitle("Training progress")
    margin = 0.45 * inch
    panel_w = 2.9 * inch
    panel_h = 1.65 * inch
    y = 0.75 * inch

    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(PALETTE["ink"])
    c.drawString(margin, 2.65 * inch, "PPO training progression from local logs")
    c.setFont("Helvetica", 7)
    c.setFillColor(PALETTE["gray"])
    c.drawString(margin, 2.48 * inch, "Each point is the latest pulled monitor value for one training task.")

    draw_axes(c, margin, y, panel_w, panel_h, "Evaluation win rate", "fraction")
    draw_line(c, margin, y, panel_w, panel_h, safe_series(run_df, "win_rate"), PALETTE["teal"], 0, 1)
    draw_line(c, margin, y, panel_w, panel_h, [LEADERBOARD["win_rate"]] * len(run_df), PALETTE["gold"], 0, 1, marker=False)
    draw_legend(c, margin, 0.48 * inch, [("logged eval", PALETTE["teal"]), ("leaderboard 66.2%", PALETTE["gold"])])

    x2 = margin + panel_w + 0.65 * inch
    rewards = safe_series(run_df, "reward")
    clean_rewards = [v for v in rewards if v is not None]
    rmin = min(clean_rewards + [0])
    rmax = max(clean_rewards + [1])
    pad = max(5.0, (rmax - rmin) * 0.12)
    draw_axes(c, x2, y, panel_w, panel_h, "Training reward", "monitor scalar")
    draw_line(c, x2, y, panel_w, panel_h, rewards, PALETTE["navy"], rmin - pad, rmax + pad)
    draw_legend(c, x2, 0.48 * inch, [("reward", PALETTE["navy"])])
    c.save()


def create_lane_pressure(run_df: pd.DataFrame) -> None:
    path = FIG_DIR / "lane_pressure.pdf"
    c = canvas.Canvas(str(path), pagesize=(7.0 * inch, 3.0 * inch))
    margin = 0.45 * inch
    panel_w = 2.9 * inch
    panel_h = 1.65 * inch
    y = 0.75 * inch

    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(PALETTE["ink"])
    c.drawString(margin, 2.65 * inch, "Lane pressure diagnostics")
    c.setFont("Helvetica", 7)
    c.setFillColor(PALETTE["gray"])
    c.drawString(margin, 2.48 * inch, "Lower enemy tower HP and positive kill/death gap indicate pressure conversion.")

    draw_axes(c, margin, y, panel_w, panel_h, "Tower HP at episode end", "HP")
    tower_vals = safe_series(run_df, "self_tower_hp") + safe_series(run_df, "enemy_tower_hp")
    clean = [v for v in tower_vals if v is not None]
    ymax = max(clean + [10000])
    draw_line(c, margin, y, panel_w, panel_h, safe_series(run_df, "self_tower_hp"), PALETTE["green"], 0, ymax)
    draw_line(c, margin, y, panel_w, panel_h, safe_series(run_df, "enemy_tower_hp"), PALETTE["red"], 0, ymax)
    draw_legend(c, margin, 0.48 * inch, [("self tower", PALETTE["green"]), ("enemy tower", PALETTE["red"])])

    x2 = margin + panel_w + 0.65 * inch
    gap = []
    for k, d in zip(safe_series(run_df, "kill"), safe_series(run_df, "death")):
        gap.append(None if k is None or d is None else k - d)
    draw_axes(c, x2, y, panel_w, panel_h, "Kill-death gap", "kills minus deaths")
    clean_gap = [v for v in gap if v is not None]
    gmax = max(abs(min(clean_gap + [0])), abs(max(clean_gap + [1])))
    draw_line(c, x2, y, panel_w, panel_h, gap, PALETTE["purple"], -gmax - 0.2, gmax + 0.2)
    draw_legend(c, x2, 0.48 * inch, [("K-D", PALETTE["purple"])])
    c.save()


def create_recall_diagnostics(run_df: pd.DataFrame, probe_df: pd.DataFrame) -> None:
    path = FIG_DIR / "recall_diagnostics.pdf"
    c = canvas.Canvas(str(path), pagesize=(7.0 * inch, 3.0 * inch))
    margin = 0.45 * inch
    panel_w = 2.9 * inch
    panel_h = 1.65 * inch
    y = 0.75 * inch

    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(PALETTE["ink"])
    c.drawString(margin, 2.65 * inch, "Recall was a real negative result")
    c.setFont("Helvetica", 7)
    c.setFillColor(PALETTE["gray"])
    c.drawString(margin, 2.48 * inch, "Button9 was exposed in several experiments, but final config disables recall.")

    draw_axes(c, margin, y, panel_w, panel_h, "Button9 action count", "latest monitor count")
    values = safe_series(run_df, "action_button_9")
    clean = [v for v in values if v is not None]
    draw_line(c, margin, y, panel_w, panel_h, values, PALETTE["gold"], 0, max(clean + [1]))
    draw_legend(c, margin, 0.48 * inch, [("Button9", PALETTE["gold"])])

    x2 = margin + panel_w + 0.65 * inch
    draw_axes(c, x2, y, panel_w, panel_h, "Synthetic recall reward", "recall_recover")
    if not probe_df.empty:
        recall_values = []
        for _, row in probe_df.iterrows():
            val = row.get("recall_recover")
            recall_values.append(None if pd.isna(val) else float(val))
    else:
        recall_values = []
    clean_r = [v for v in recall_values if v is not None]
    rmin = min(clean_r + [0])
    rmax = max(clean_r + [1])
    pad = max(0.05, (rmax - rmin) * 0.12)
    draw_line(c, x2, y, panel_w, panel_h, recall_values, PALETTE["red"], rmin - pad, rmax + pad)
    draw_legend(c, x2, 0.48 * inch, [("probe reward", PALETTE["red"])])
    c.save()


def _draw_academic_header(c: canvas.Canvas, title: str, subtitle: str, w: float, h: float) -> None:
    c.setFillColor(colors.white)
    c.rect(0, 0, w, h, fill=1, stroke=0)
    c.setFillColor(PALETTE["red"])
    c.setFont("Helvetica-Bold", 15)
    c.drawString(0.28 * inch, h - 0.35 * inch, title)
    c.setFillColor(PALETTE["gray"])
    c.setFont("Helvetica", 7.6)
    c.drawString(0.29 * inch, h - 0.53 * inch, subtitle)


def _draw_image_cover(c: canvas.Canvas, path: Path, x: float, y: float, w: float, h: float) -> bool:
    if not path.exists():
        return False
    c.drawImage(
        ImageReader(str(path)),
        x,
        y,
        width=w,
        height=h,
        preserveAspectRatio=True,
        anchor="c",
        mask="auto",
    )
    return True


def _soft_white(c: canvas.Canvas, x: float, y: float, w: float, h: float, alpha: float = 0.9, radius: float = 7) -> None:
    c.saveState()
    c.setFillAlpha(alpha)
    c.setStrokeAlpha(0.9)
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.HexColor("#D7DEE4"))
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)
    c.restoreState()


def _round_box(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    fill=colors.white,
    stroke=PALETTE["ink"],
    font_size: float = 7.6,
    bold: bool = True,
    radius: float = 5,
    dash: tuple[int, int] | None = None,
) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(1.05)
    if dash:
        c.setDash(*dash)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)
    if dash:
        c.setDash()
    c.setFillColor(PALETTE["ink"])
    c.setFont("Helvetica-Bold" if bold else "Helvetica", font_size)
    lines = label.split("\n")
    line_h = font_size + 1.5
    start = y + h / 2 + line_h * (len(lines) - 1) / 2 - font_size / 2
    for i, line in enumerate(lines):
        c.drawCentredString(x + w / 2, start - i * line_h, line)


def _arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, color=PALETTE["ink"], width=1.15) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 6.5
    for delta in (0.48, -0.48):
        c.line(x2, y2, x2 - head * math.cos(angle + delta), y2 - head * math.sin(angle + delta))


def _callout(c: canvas.Canvas, text: str, x: float, y: float, w: float, color=PALETTE["ink"]) -> None:
    c.setFillColor(color)
    c.setFont("Helvetica", 6.8)
    words = text.split()
    line = ""
    yy = y
    max_chars = max(12, int(w / 3.2))
    for word in words:
        probe = word if not line else line + " " + word
        if len(probe) > max_chars:
            c.drawString(x, yy, line)
            yy -= 8
            line = word
        else:
            line = probe
    if line:
        c.drawString(x, yy, line)


def _tag(c: canvas.Canvas, text: str, x: float, y: float, fill, stroke=None) -> None:
    c.setFont("Helvetica-Bold", 6.5)
    width = c.stringWidth(text, "Helvetica-Bold", 6.5) + 10
    c.setFillColor(fill)
    c.setStrokeColor(stroke or fill)
    c.roundRect(x, y, width, 13, 5, fill=1, stroke=1)
    c.setFillColor(colors.white)
    c.drawCentredString(x + width / 2, y + 4, text)


def _draw_cute_icon(c: canvas.Canvas, kind: str, x: float, y: float, scale: float = 1.0, enemy: bool = False) -> None:
    """Draw small 2D icons for game entities. (x, y) is icon center."""
    if kind == "hero":
        main = PALETTE["red"] if enemy else PALETTE["teal"]
        c.setFillColor(colors.whitesmoke)
        c.setStrokeColor(main)
        c.setLineWidth(1.0)
        c.circle(x, y + 7 * scale, 9 * scale, fill=1, stroke=1)
        c.setFillColor(main)
        c.circle(x - 3 * scale, y + 9 * scale, 1.2 * scale, fill=1, stroke=0)
        c.circle(x + 3 * scale, y + 9 * scale, 1.2 * scale, fill=1, stroke=0)
        c.setStrokeColor(main)
        c.line(x - 3 * scale, y + 4 * scale, x + 3 * scale, y + 4 * scale)
        c.setFillColor(main)
        c.roundRect(x - 8 * scale, y - 12 * scale, 16 * scale, 12 * scale, 4 * scale, fill=1, stroke=0)
    elif kind == "minion":
        main = PALETTE["red"] if enemy else PALETTE["green"]
        c.setFillColor(colors.whitesmoke)
        c.setStrokeColor(main)
        c.setLineWidth(0.9)
        c.circle(x, y + 4 * scale, 6 * scale, fill=1, stroke=1)
        c.setFillColor(main)
        c.rect(x - 7 * scale, y + 6 * scale, 14 * scale, 4 * scale, fill=1, stroke=0)
        c.roundRect(x - 6 * scale, y - 8 * scale, 12 * scale, 9 * scale, 3 * scale, fill=1, stroke=0)
    elif kind == "tower":
        main = PALETTE["red"] if enemy else PALETTE["navy"]
        c.setFillColor(colors.HexColor("#EEF2F4"))
        c.setStrokeColor(main)
        c.setLineWidth(1.0)
        c.rect(x - 9 * scale, y - 12 * scale, 18 * scale, 22 * scale, fill=1, stroke=1)
        for dx in (-8, -2, 4):
            c.rect(x + dx * scale, y + 10 * scale, 5 * scale, 5 * scale, fill=1, stroke=1)
        c.setFillColor(main)
        c.roundRect(x - 4 * scale, y - 12 * scale, 8 * scale, 10 * scale, 3 * scale, fill=1, stroke=0)
    elif kind == "bullet":
        c.setFillColor(PALETTE["gold"])
        c.setStrokeColor(PALETTE["gold"])
        p = c.beginPath()
        p.moveTo(x + 10 * scale, y)
        p.lineTo(x - 3 * scale, y + 6 * scale)
        p.lineTo(x - 8 * scale, y)
        p.lineTo(x - 3 * scale, y - 6 * scale)
        p.close()
        c.drawPath(p, fill=1, stroke=0)
        c.setStrokeColor(PALETTE["gold"])
        c.line(x - 11 * scale, y, x - 18 * scale, y)
    elif kind == "cake":
        c.setFillColor(colors.HexColor("#57B7FF"))
        c.setStrokeColor(PALETTE["navy"])
        p = c.beginPath()
        p.moveTo(x, y + 10 * scale)
        p.lineTo(x + 10 * scale, y)
        p.lineTo(x, y - 10 * scale)
        p.lineTo(x - 10 * scale, y)
        p.close()
        c.drawPath(p, fill=1, stroke=1)
        c.setStrokeColor(colors.white)
        c.line(x - 4 * scale, y, x + 4 * scale, y)
        c.line(x, y - 4 * scale, x, y + 4 * scale)
    elif kind == "monster":
        c.setFillColor(colors.HexColor("#8E74B8"))
        c.setStrokeColor(PALETTE["purple"])
        c.roundRect(x - 10 * scale, y - 8 * scale, 20 * scale, 16 * scale, 7 * scale, fill=1, stroke=1)
        c.setFillColor(colors.white)
        c.circle(x - 4 * scale, y + 2 * scale, 1.4 * scale, fill=1, stroke=0)
        c.circle(x + 4 * scale, y + 2 * scale, 1.4 * scale, fill=1, stroke=0)


def create_feature_pipeline() -> None:
    path = FIG_DIR / "feature_pipeline.pdf"
    w, h = 7.2 * inch, 3.35 * inch
    c = canvas.Canvas(str(path), pagesize=(w, h))
    c.setTitle("Feature pipeline")
    if not _draw_image_cover(c, AI_DIR / "feature_processing_ai.png", 0, 0, w, h):
        _draw_academic_header(
            c,
            "Input feature processing",
            "Raw frame state is canonicalized into typed entity tokens, global context, and button-aware legal targets.",
            w,
            h,
        )

    _soft_white(c, 0.18 * inch, 2.77 * inch, 3.9 * inch, 0.38 * inch, 0.93, 9)
    c.setFillColor(PALETTE["red"])
    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.3 * inch, 3.0 * inch, "Input feature processing")
    c.setFillColor(PALETTE["gray"])
    c.setFont("Helvetica", 6.9)
    c.drawString(0.31 * inch, 2.86 * inch, "Frame state -> canonical tokens -> target slots -> 561-dim policy input")

    _soft_white(c, 4.92 * inch, 0.55 * inch, 1.95 * inch, 1.96 * inch, 0.9, 10)
    c.setFillColor(PALETTE["ink"])
    c.setFont("Helvetica-Bold", 8.2)
    c.drawString(5.08 * inch, 2.32 * inch, "Token schema")
    rows = [
        ("hero", "2 x 124"),
        ("tower", "2 x 17"),
        ("minion", "8 x 22"),
        ("monster", "1 x 8"),
        ("bullet", "4 x 16"),
        ("cake", "2 x 6"),
        ("global", "19"),
    ]
    c.setFont("Helvetica", 6.7)
    for i, (name, dim) in enumerate(rows):
        yy = 2.14 * inch - i * 0.19 * inch
        c.setFillColor(colors.HexColor("#EEF7F6") if i % 2 == 0 else colors.white)
        c.rect(5.07 * inch, yy - 0.04 * inch, 1.48 * inch, 0.13 * inch, fill=1, stroke=0)
        c.setFillColor(PALETTE["ink"])
        c.drawString(5.12 * inch, yy, name)
        c.drawRightString(6.48 * inch, yy, dim)

    _round_box(c, 5.08 * inch, 0.70 * inch, 1.50 * inch, 0.25 * inch, "542 token + 19 global = 561", fill=colors.HexColor("#FFF7DF"), stroke=PALETTE["gold"], font_size=6.4)
    _round_box(c, 4.95 * inch, 2.60 * inch, 1.92 * inch, 0.32 * inch, "Soldier1-4: nearest visible, then runtime-id order", fill=colors.white, stroke=PALETTE["red"], font_size=6.1)
    _round_box(c, 2.98 * inch, 0.26 * inch, 2.02 * inch, 0.33 * inch, "exists is padding; visible/alive/time-since-seen are features", fill=colors.white, stroke=PALETTE["teal"], font_size=5.9)
    c.save()


def create_architecture() -> None:
    path = FIG_DIR / "architecture.pdf"
    w, h = 7.2 * inch, 3.45 * inch
    c = canvas.Canvas(str(path), pagesize=(w, h))
    c.setTitle("Uzi PPO architecture")
    if not _draw_image_cover(c, AI_DIR / "policy_architecture_ai.png", 0, 0, w, h):
        _draw_academic_header(
            c,
            "Uzi policy architecture",
            "A raw MLP preserves dense fields while a small token encoder repairs entity and target semantics.",
            w,
            h,
        )
    else:
        _soft_white(c, 0.18 * inch, 2.91 * inch, 4.25 * inch, 0.37 * inch, 0.93, 9)
        c.setFillColor(PALETTE["red"])
        c.setFont("Helvetica-Bold", 14)
        c.drawString(0.3 * inch, 3.14 * inch, "Uzi policy architecture")
        c.setFillColor(PALETTE["gray"])
        c.setFont("Helvetica", 6.9)
        c.drawString(0.31 * inch, 3.00 * inch, "Exact flow from agent_ppo/model/model.py, with generated icons as visual components")

    _soft_white(c, 0.24 * inch, 0.38 * inch, 6.82 * inch, 2.48 * inch, 0.84, 10)

    input_x, input_y, input_w, input_h = 0.28 * inch, 0.58 * inch, 1.08 * inch, 2.28 * inch
    _round_box(c, input_x, input_y, input_w, input_h, "", fill=colors.HexColor("#EDEDED"), stroke=PALETTE["ink"])
    c.setFont("Helvetica-Bold", 8.3)
    c.setFillColor(PALETTE["ink"])
    c.drawCentredString(input_x + input_w / 2, input_y + input_h - 16, "Feature vector")
    _tag(c, "561 dims", input_x + 13, input_y + input_h - 36, PALETTE["navy"])
    for i, (kind, enemy) in enumerate([("hero", False), ("hero", True), ("tower", False), ("minion", True), ("bullet", False), ("cake", False)]):
        _draw_cute_icon(c, kind, input_x + 23 + (i % 2) * 30, input_y + 124 - (i // 2) * 42, 0.55, enemy=enemy)
    c.setFont("Helvetica", 6.5)
    c.setFillColor(PALETTE["gray"])
    c.drawCentredString(input_x + input_w / 2, input_y + 16, "19 tokens + global")

    raw_x = 1.72 * inch
    _round_box(c, raw_x, 2.18 * inch, 1.08 * inch, 0.44 * inch, "Raw MLP\n561 -> 256", fill=colors.HexColor("#F7F7F7"), stroke=PALETTE["gray"])
    _round_box(c, raw_x, 0.81 * inch, 1.08 * inch, 0.44 * inch, "Type-shared\nprojection", fill=colors.HexColor("#E9F5F2"), stroke=PALETTE["teal"])
    _round_box(c, raw_x, 1.48 * inch, 1.08 * inch, 0.36 * inch, "Global MLP\n19 -> 64", fill=colors.HexColor("#EEF2FA"), stroke=PALETTE["navy"])
    _arrow(c, input_x + input_w + 10, 2.4 * inch, raw_x - 10, 2.4 * inch, PALETTE["gray"])
    _arrow(c, input_x + input_w + 10, 1.03 * inch, raw_x - 10, 1.03 * inch, PALETTE["gray"])
    _arrow(c, input_x + input_w + 10, 1.66 * inch, raw_x - 10, 1.66 * inch, PALETTE["gray"])

    attn_x, attn_y, attn_w, attn_h = 3.08 * inch, 0.58 * inch, 1.48 * inch, 2.28 * inch
    _round_box(c, attn_x, attn_y, attn_w, attn_h, "", fill=colors.HexColor("#F7D7D6"), stroke=PALETTE["red"])
    c.setFont("Helvetica-Bold", 8.2)
    c.setFillColor(PALETTE["ink"])
    c.drawCentredString(attn_x + attn_w / 2, attn_y + attn_h - 16, "Token encoder")
    _round_box(c, attn_x + 18, attn_y + attn_h - 44, attn_w - 36, 17, "2 register tokens", fill=colors.white, stroke=PALETTE["gray"], font_size=6.3)
    _round_box(c, attn_x + 18, attn_y + 100, attn_w - 36, 22, "AdaLN attention", fill=colors.HexColor("#DDEFF7"), stroke=PALETTE["navy"], font_size=6.5)
    _round_box(c, attn_x + 18, attn_y + 70, attn_w - 36, 22, "FFN + residual", fill=colors.HexColor("#DDEFF7"), stroke=PALETTE["navy"], font_size=6.5)
    _round_box(c, attn_x + 18, attn_y + 38, attn_w - 36, 22, "repeat x2", fill=colors.white, stroke=PALETTE["gray"], font_size=6.5)
    _round_box(c, attn_x + 18, attn_y + 11, attn_w - 36, 18, "exists -> key mask", fill=colors.HexColor("#FFF7DF"), stroke=PALETTE["gold"], font_size=6.1)
    _arrow(c, raw_x + 1.08 * inch + 10, 1.03 * inch, attn_x - 12, 1.43 * inch, PALETTE["gray"])
    _arrow(c, raw_x + 1.08 * inch + 10, 1.66 * inch, attn_x - 12, 1.43 * inch, PALETTE["gray"])

    fuse_x = 4.88 * inch
    _round_box(c, fuse_x, 1.92 * inch, 1.05 * inch, 0.43 * inch, "Residual fuse\nraw + 0.05 token", fill=colors.HexColor("#F7F7F7"), stroke=PALETTE["ink"])
    _round_box(c, fuse_x, 1.18 * inch, 1.05 * inch, 0.43 * inch, "LSTM residual\n512, T=16", fill=colors.HexColor("#EFE8F7"), stroke=PALETTE["purple"])
    _arrow(c, raw_x + 1.08 * inch + 10, 2.4 * inch, fuse_x - 12, 2.13 * inch, PALETTE["gray"])
    _arrow(c, attn_x + attn_w + 10, 1.62 * inch, fuse_x - 12, 2.06 * inch, PALETTE["gray"])
    _arrow(c, fuse_x + 0.52 * inch, 1.92 * inch, fuse_x + 0.52 * inch, 1.61 * inch, PALETTE["gray"])

    out_x = 6.22 * inch
    out_specs = [
        (2.40, "button\n12", PALETTE["gold"]),
        (1.96, "move / skill\n16 x 4", PALETTE["gold"]),
        (1.35, "target pointer\nbutton -> 9", PALETTE["red"]),
        (0.80, "value\ncritic", PALETTE["teal"]),
    ]
    trunk_x = out_x - 0.18 * inch
    _arrow(c, fuse_x + 1.05 * inch + 8, 1.40 * inch, trunk_x, 1.40 * inch, PALETTE["gray"], width=1.0)
    c.setStrokeColor(PALETTE["gray"])
    c.setLineWidth(1.0)
    c.line(trunk_x, 0.95 * inch, trunk_x, 2.55 * inch)
    for yy, label, col in out_specs:
        _round_box(c, out_x, yy * inch, 0.72 * inch, 0.31 * inch, label, fill=colors.white, stroke=col, font_size=6.1)
        _arrow(c, trunk_x, (yy + 0.15) * inch, out_x - 9, (yy + 0.15) * inch, PALETTE["gray"], width=0.9)

    c.setDash(3, 3)
    _arrow(c, attn_x + attn_w, 1.12 * inch, out_x - 10, 1.50 * inch, PALETTE["red"], width=0.8)
    c.setDash()
    c.setDash(3, 3)
    c.setStrokeColor(PALETTE["red"])
    c.roundRect(5.97 * inch, 1.14 * inch, 1.13 * inch, 0.67 * inch, 7, fill=0, stroke=1)
    c.setDash()
    _callout(c, "Target logits are gathered for the selected button during PPO and during sampling.", 5.94 * inch, 0.47 * inch, 1.1 * inch, PALETTE["red"])
    _callout(c, "Action mask: Button3 cannot choose None/Self; Button9 recall is disabled in the final model.", 3.02 * inch, 0.26 * inch, 2.2 * inch, PALETTE["gray"])
    c.save()


def create_training_curriculum() -> None:
    path = FIG_DIR / "training_curriculum.pdf"
    w, h = 7.2 * inch, 3.15 * inch
    c = canvas.Canvas(str(path), pagesize=(w, h))
    c.setTitle("Training curriculum")
    if not _draw_image_cover(c, AI_DIR / "training_curriculum_ai.png", 0, 0, w, h):
        _draw_academic_header(
            c,
            "Three-stage training strategy",
            "The submission policy was shaped from basic movement to aggressive laning and finally efficient win conversion.",
            w,
            h,
        )
    else:
        _soft_white(c, 0.18 * inch, 2.60 * inch, 4.1 * inch, 0.37 * inch, 0.93, 9)
        c.setFillColor(PALETTE["red"])
        c.setFont("Helvetica-Bold", 14)
        c.drawString(0.3 * inch, 2.83 * inch, "Three-stage training strategy")
        c.setFillColor(PALETTE["gray"])
        c.setFont("Helvetica", 6.9)
        c.drawString(0.31 * inch, 2.69 * inch, "Basic movement -> lane aggression -> efficient win conversion")

    base_y = 0.62 * inch
    col_w = 1.75 * inch
    xs = [0.38 * inch, 2.75 * inch, 5.08 * inch]
    stage = [
        (
            "Stage I",
            "Basic control",
            ["walk to lane", "avoid idle", "stay observable", "safe retreat"],
            PALETTE["teal"],
            ["lane_progress", "lane_presence", "idle penalty"],
        ),
        (
            "Stage II",
            "Lane aggression",
            ["last-hit focus", "safe trades", "valid targets", "tower windows"],
            PALETTE["red"],
            ["target pointer", "Button3 mask", "tower_attack"],
        ),
        (
            "Stage III",
            "Win conversion",
            ["multi-hero mix", "self-play/common-AI", "disable recall", "submit ppo-5-1"],
            PALETTE["gold"],
            ["96k learner steps", "20.5k episodes", "Rank #3"],
        ),
    ]
    for i, (tag, title, bullets, color, notes) in enumerate(stage):
        x = xs[i]
        _round_box(c, x, base_y, col_w, 1.78 * inch, "", fill=colors.HexColor("#F8F8F8"), stroke=colors.HexColor("#C7CCD1"))
        _tag(c, tag, x + 12, base_y + 1.55 * inch, color)
        c.setFillColor(PALETTE["ink"])
        c.setFont("Helvetica-Bold", 9.4)
        c.drawString(x + 12, base_y + 1.36 * inch, title)
        for j, bullet_text in enumerate(bullets):
            yy = base_y + 1.12 * inch - j * 0.22 * inch
            c.setFillColor(color)
            c.circle(x + 16, yy + 2, 2.4, fill=1, stroke=0)
            c.setFillColor(PALETTE["ink"])
            c.setFont("Helvetica", 6.9)
            c.drawString(x + 24, yy, bullet_text)
        for j, note in enumerate(notes):
            _round_box(
                c,
                x + 12 + (j % 2) * 0.75 * inch,
                base_y + 0.17 * inch - (j // 2) * 0.21 * inch,
                0.66 * inch,
                0.16 * inch,
                note,
                fill=colors.white,
                stroke=color,
                font_size=4.9,
                bold=False,
                radius=3,
            )
        if i < 2:
            _arrow(c, x + col_w + 14, base_y + 0.96 * inch, xs[i + 1] - 15, base_y + 0.96 * inch, PALETTE["gray"], width=1.2)

    c.setStrokeColor(PALETTE["navy"])
    c.setLineWidth(1.1)
    c.line(0.52 * inch, 0.42 * inch, 6.75 * inch, 0.42 * inch)
    for x, label in [
        (0.62 * inch, "feature migration"),
        (2.86 * inch, "target semantics fixed"),
        (5.05 * inch, "final monitor: 68.8% WR"),
        (6.25 * inch, "leaderboard: 131/67"),
    ]:
        c.setFillColor(PALETTE["navy"])
        c.circle(x, 0.42 * inch, 3, fill=1, stroke=0)
        c.setFont("Helvetica", 6.2)
        c.setFillColor(PALETTE["gray"])
        c.drawCentredString(x, 0.24 * inch, label)
    c.save()


def create_leaderboard_card() -> None:
    path = FIG_DIR / "leaderboard_card.pdf"
    c = canvas.Canvas(str(path), pagesize=(3.5 * inch, 2.1 * inch))
    c.setFillColor(colors.white)
    c.rect(0, 0, 3.5 * inch, 2.1 * inch, fill=1, stroke=0)
    c.setStrokeColor(PALETTE["navy"])
    c.setLineWidth(1.5)
    c.roundRect(0.16 * inch, 0.16 * inch, 3.18 * inch, 1.78 * inch, 6, fill=0, stroke=1)
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(PALETTE["navy"])
    c.drawString(0.35 * inch, 1.48 * inch, "Uzi")
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(PALETTE["gold"])
    c.drawString(2.25 * inch, 1.5 * inch, "Rank #3")
    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(PALETTE["ink"])
    c.drawString(0.35 * inch, 0.88 * inch, "131 / 67")
    c.setFont("Helvetica", 8)
    c.setFillColor(PALETTE["gray"])
    c.drawString(0.36 * inch, 0.68 * inch, "A wins / B wins over 198 games")
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(PALETTE["teal"])
    c.drawString(2.22 * inch, 0.88 * inch, "66.2%")
    c.setFont("Helvetica", 8)
    c.setFillColor(PALETTE["gray"])
    c.drawString(2.23 * inch, 0.68 * inch, "aggregate win rate")
    c.save()


def create_probe_matrix(probe_df: pd.DataFrame) -> None:
    path = FIG_DIR / "historical_probe_matrix.pdf"
    c = canvas.Canvas(str(path), pagesize=(7.0 * inch, 3.3 * inch))
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(PALETTE["ink"])
    c.drawString(0.35 * inch, 3.0 * inch, "Synthetic frame probes across code history")
    c.setFont("Helvetica", 7)
    c.setFillColor(PALETTE["gray"])
    c.drawString(0.35 * inch, 2.82 * inch, "Cells summarize deterministic code behavior, not full match evaluation.")

    headers = ["version", "mask", "soldier", "recall", "Arli mark"]
    x = [0.35, 1.72, 2.62, 3.72, 4.82]
    y = 2.52
    c.setFont("Helvetica-Bold", 7.2)
    c.setFillColor(PALETTE["ink"])
    for xpos, header in zip(x, headers):
        c.drawString(xpos * inch, y * inch, header)
    c.setStrokeColor(PALETTE["light_gray"])
    c.line(0.35 * inch, 2.44 * inch, 6.65 * inch, 2.44 * inch)

    def flag(value, good=True):
        return PALETTE["green"] if value and good else PALETTE["red"] if value is False and good else PALETTE["gray"]

    rows = probe_df if not probe_df.empty else pd.DataFrame()
    max_rows = min(len(rows), 11)
    c.setFont("Helvetica", 6.6)
    for i in range(max_rows):
        row = rows.iloc[i]
        yy = 2.25 - i * 0.18
        label = str(row["label"]).replace("_", " ")
        if len(label) > 21:
            label = label[:20] + "."
        c.setFillColor(PALETTE["ink"])
        c.drawString(x[0] * inch, yy * inch, label)

        mask_ok = bool(row.get("mask_available")) and float(row.get("button3_after") or 0) == 0.0
        soldier_ok = row.get("soldier_slots") == "10,20,30"
        recall_ok = bool(row.get("recall_available"))
        arli_ok = bool(row.get("arli_has_mark_field")) and int(row.get("arli_nonzero_marks") or 0) > 0
        vals = [
            ("blocks", mask_ok),
            (str(row.get("soldier_slots") or "-"), soldier_ok),
            ("avail" if recall_ok else "-", recall_ok),
            ("nonzero" if arli_ok else "-", arli_ok),
        ]
        for j, (text, ok) in enumerate(vals, start=1):
            c.setFillColor(flag(ok))
            c.circle((x[j] - 0.05) * inch, (yy + 0.02) * inch, 2.1, fill=1, stroke=0)
            c.setFillColor(PALETTE["ink"])
            c.drawString(x[j] * inch, yy * inch, text[:12])
    c.save()


def write_latex_tables(run_df: pd.DataFrame, probe_df: pd.DataFrame) -> None:
    rows = []
    selected = run_df[
        [
            "run",
            "reward",
            "win_rate",
            "kill",
            "death",
            "self_tower_hp",
            "enemy_tower_hp",
            "action_button_9",
        ]
    ].tail(8)
    for _, row in selected.iterrows():
        def fmt(value, digits=2):
            return "--" if pd.isna(value) else f"{float(value):.{digits}f}"
        rows.append(
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                row["run"],
                fmt(row["reward"], 1),
                fmt(100 * row["win_rate"], 1),
                fmt(row["kill"], 2),
                fmt(row["death"], 2),
                fmt(row["self_tower_hp"], 0),
                fmt(row["enemy_tower_hp"], 0),
                fmt(row["action_button_9"], 0),
            )
        )
    (DATA_DIR / "ppo_table_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    if not probe_df.empty:
        probe_rows = []
        for _, row in probe_df.tail(7).iterrows():
            mask = "yes" if bool(row.get("mask_available")) and float(row.get("button3_after") or 0) == 0.0 else "no"
            soldier = row.get("soldier_slots") or "--"
            recall = "--" if pd.isna(row.get("recall_recover")) else f"{float(row.get('recall_recover')):.3f}"
            arli = "yes" if bool(row.get("arli_has_mark_field")) else "no"
            probe_rows.append(
                "{} & {} & {} & {} & {} \\\\".format(
                    str(row["label"]).replace("_", "\\_"),
                    mask,
                    soldier,
                    recall,
                    arli,
                )
            )
        (DATA_DIR / "probe_table_rows.tex").write_text("\n".join(probe_rows) + "\n", encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    run_df = collect_run_summary()
    final_note = collect_final_run_note()
    probes = probe_table()
    write_csvs(run_df, probes, final_note)
    write_latex_tables(run_df, probes)
    create_training_progress(run_df)
    create_lane_pressure(run_df)
    create_feature_pipeline()
    create_architecture()
    create_training_curriculum()
    create_leaderboard_card()
    create_recall_diagnostics(run_df, probes)
    create_probe_matrix(probes)
    print(json.dumps({
        "runs": len(run_df),
        "probe_rows": len(probes),
        "figures": sorted(p.name for p in FIG_DIR.glob("*.pdf")),
        "leaderboard_win_rate": LEADERBOARD["win_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
