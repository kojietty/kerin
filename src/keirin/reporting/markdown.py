"""Render the daily Markdown report (Git-friendly mirror of the HTML dashboard)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# 決戦速報 — {payload['date_label']}")
    lines.append("")
    lines.append(f"_generated: {payload['generated_at']}_")
    lines.append("")
    if payload.get("emergency_stop"):
        lines.append(f"> **緊急停止中** — 月初累積 ¥{payload['mtd_pnl']:+,d} が下限を割り込みました。本番購入停止。")
        lines.append("")

    lines.append("## 収支")
    lines.append("")
    lines.append("| 月初比 P&L | 月次 ROI | 月次 的中率 | 本日 投資 | 残り予算 |")
    lines.append("|---|---|---|---|---|")
    roi = f"{payload['mtd_roi']:.0%}" if payload.get("mtd_roi") is not None else "—"
    hr = f"{payload['mtd_hit_rate']:.0%}" if payload.get("mtd_hit_rate") is not None else "—"
    lines.append(
        f"| ¥{payload['mtd_pnl']:+,d} | {roi} | {hr} | ¥{payload['daily_invest']:,d} | ¥{payload['daily_remaining']:,d} |"
    )
    lines.append("")

    rec = payload.get("recommended", [])
    skip = payload.get("skipped", [])
    total_picks = sum(len(r["picks"]) for r in rec)
    lines.append(
        f"本日開催 **{payload['total_races']}**R / 推奨 **{len(rec)}**R / 総点数 **{total_picks}** / 投資 **¥{payload['daily_invest']:,d}**"
    )
    lines.append("")

    for r in rec:
        lines.append(f"## {r['venue_name']} {r['race_no']}R")
        meta = " / ".join(
            x for x in (r.get("grade"), r.get("race_class"), r.get("post_time"),
                        f"{r.get('distance_m')}m" if r.get("distance_m") else None,
                        r.get("weather")) if x
        )
        if meta:
            lines.append(f"_{meta}_")
        if r.get("lines"):
            lines.append("**並び**: " + " / ".join("-".join(str(c) for c in line) for line in r["lines"]))
        if r.get("axis"):
            ax = r["axis"]
            note = f" ({ax['note']})" if ax.get("note") else ""
            lines.append(f"**軸**: {ax['car_no']}番 {ax['name']}{note}")
        lines.append("")
        lines.append("| 買い目 | オッズ | 推定確率 | EV | 金額 |")
        lines.append("|---|---:|---:|---:|---:|")
        for p in r["picks"]:
            combo = "-".join(str(c) for c in p["cars"])
            stake = f"¥{p['stake_yen']:,d}" if p["stake_yen"] > 0 else "paper"
            lines.append(
                f"| {combo} | {p['odds']:.1f} | {p['prob']:.1%} | {p['ev']:.2f} | {stake} |"
            )
        lines.append("")
        lines.append(f"**計**: {len(r['picks'])}点 / ¥{r['total_stake']:,d}")
        lines.append("")

    if not rec:
        lines.append("> 本日は推奨レースなし（EV閾値 {0:.2f} を超える買い目なし）".format(payload.get("ev_threshold", 1.10)))
        lines.append("")

    if skip:
        lines.append(f"## 見送りレース ({len(skip)}本)")
        lines.append("")
        for s in skip:
            mx = f"max EV {s['max_ev']:.2f}" if s.get("max_ev") is not None else "—"
            lines.append(f"- ~~{s['venue_name']} {s['race_no']}R~~ — {mx}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_markdown(out_path: Path, payload: dict[str, Any]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(payload), encoding="utf-8")
    return out_path
