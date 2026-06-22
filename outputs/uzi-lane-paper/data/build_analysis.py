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
from reportlab.pdfgen import canvas


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "outputs" / "uzi-lane-paper"
DATA_DIR = OUT_ROOT / "data"
FIG_DIR = OUT_ROOT / "figures"
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


def create_architecture() -> None:
    path = FIG_DIR / "architecture.pdf"
    c = canvas.Canvas(str(path), pagesize=(7.0 * inch, 2.8 * inch))
    c.setTitle("Architecture")
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(PALETTE["ink"])
    c.drawString(0.35 * inch, 2.45 * inch, "Uzi PPO architecture")

    boxes = [
        (0.35, 1.55, 1.05, 0.45, "Entity tokens", PALETTE["teal"]),
        (0.35, 0.78, 1.05, 0.45, "Global state", PALETTE["green"]),
        (1.75, 1.15, 1.2, 0.55, "Hybrid encoder\\nMLP + tokens", PALETTE["navy"]),
        (3.25, 1.15, 0.95, 0.55, "LSTM\\nresidual", PALETTE["purple"]),
        (4.55, 1.45, 1.05, 0.45, "Button head", PALETTE["gold"]),
        (4.55, 0.75, 1.05, 0.45, "Move/skill\\nheads", PALETTE["gold"]),
        (5.9, 1.1, 0.8, 0.65, "Target\\npointer", PALETTE["red"]),
    ]

    for bx, by, bw, bh, label, color in boxes:
        x, y, w, h = bx * inch, by * inch, bw * inch, bh * inch
        c.setFillColor(colors.white)
        c.setStrokeColor(color)
        c.setLineWidth(1.4)
        c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
        c.setFillColor(PALETTE["ink"])
        c.setFont("Helvetica-Bold", 7.5)
        lines = label.split("\\n")
        for i, line in enumerate(lines):
            c.drawCentredString(x + w / 2, y + h / 2 + (len(lines) - 1 - 2 * i) * 4, line)

    def arrow(x1, y1, x2, y2, color=PALETTE["gray"]):
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.1)
        c.line(x1 * inch, y1 * inch, x2 * inch, y2 * inch)
        ang = math.atan2(y2 - y1, x2 - x1)
        ah = 0.06
        for delta in (2.6, -2.6):
            c.line(
                x2 * inch,
                y2 * inch,
                (x2 - ah * math.cos(ang + delta)) * inch,
                (y2 - ah * math.sin(ang + delta)) * inch,
            )

    arrow(1.4, 1.78, 1.75, 1.45)
    arrow(1.4, 1.0, 1.75, 1.35)
    arrow(2.95, 1.42, 3.25, 1.42)
    arrow(4.2, 1.42, 4.55, 1.67)
    arrow(4.2, 1.35, 4.55, 0.98)
    arrow(5.6, 1.67, 5.9, 1.43)
    arrow(1.0, 1.55, 5.9, 1.2, PALETTE["light_gray"])

    c.setFont("Helvetica", 7.4)
    c.setFillColor(PALETTE["gray"])
    c.drawString(0.35 * inch, 0.35 * inch, "Target logits are conditioned on the selected button, aligning actions with entity slots.")
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
    create_architecture()
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
