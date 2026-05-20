"""Terminal and HTML rendering for single-race deep analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import click


def render_analysis_terminal(analysis: dict[str, Any]) -> str:
    """Render the analysis dict as a colorized terminal string using click.style()."""
    lines: list[str] = []
    W = 62

    venue = analysis.get("venue_name", "?")
    race_no = analysis.get("race_no", "?")
    date = analysis.get("date", "?")
    grade = analysis.get("grade") or ""
    bank = analysis.get("bank_length")
    bank_str = f"  バンク{bank}m" if bank else ""
    header = f"  {date}  {venue} {race_no}R  {grade}{bank_str}"

    lines.append(click.style("=" * W, fg="cyan"))
    lines.append(click.style(header.center(W), fg="cyan", bold=True))
    lines.append(click.style("=" * W, fg="cyan"))

    # ── Participants ──────────────────────────────────────────────────────
    lines.append(click.style("\n【出走選手データ】", fg="yellow", bold=True))
    lines.append(f"  {'車':>2}  {'選手名':<8}  {'階級':>4}  {'得点':>5}  {'休養':>4}  {'3着内(5走)':>9}  {'スタイル':>4}  {'ライン役割':>6}")
    lines.append("  " + "─" * 56)

    for p in analysis.get("participants", []):
        car = p.get("car_no", "?")
        name = str(p.get("name") or "?")[:8]
        cls = str(p.get("rank_class") or "?")
        rating = p.get("rating")
        rating_s = f"{rating:5.1f}" if rating is not None else "  ?  "
        rest = p.get("rest_days")
        rest_s = f"{int(rest):>2}日" if rest is not None else "  ? "
        t3 = p.get("top3_rate_5")
        t3_s = f"{t3:.0%}" if t3 is not None else "   ?"
        style = str(p.get("style") or "?")[:4]
        role = "先行" if p.get("is_leader") else ("番手" if p.get("is_follower") else "独走")
        div = p.get("divergence_score")
        div_s = f"({div:+.1f})" if div is not None else ""
        row_str = f"  {car:>2}  {name:<8}  {cls:>4}  {rating_s}  {rest_s}  {t3_s:>9}  {style:>4}  {role:>4}{div_s}"
        lines.append(row_str)

    # ── Line formation ────────────────────────────────────────────────────
    lines.append(click.style("\n【ライン並び】", fg="yellow", bold=True))
    if analysis.get("lines"):
        for ln in analysis["lines"]:
            cars_str = "─".join(str(c) for c in ln["cars"])
            mm = ln.get("mismatch_score", 0.0) or 0.0
            note = ln.get("interpretation", "")
            mm_str = click.style(f"  ミスマッチ:{mm:.1f}", fg="red") if mm > 3.0 else (
                f"  ミスマッチ:{mm:.1f}" if mm > 1.0 else ""
            )
            note_str = click.style(f"  → {note}", fg="magenta") if note else ""
            lines.append(f"  {cars_str}{mm_str}{note_str}")
    else:
        lines.append("  (ライン情報なし)")

    # ── Similar past races ────────────────────────────────────────────────
    similar = analysis.get("similar_races", [])
    if similar:
        lines.append(click.style("\n【過去の似たようなレース】", fg="yellow", bold=True))
        for i, sr in enumerate(similar, 1):
            result_str = "→".join(str(c) for c in sr.get("result", [])) or "?"
            kimarite = sr.get("winner_style") or ""
            sim_s = click.style(f"{sr['similarity_score']:.2f}", fg="green" if sr["similarity_score"] > 0.8 else "white")
            lines.append(
                f"  {i}. {sr['date']}  {sr['venue_name']}{sr['race_no']}R"
                f"  ({sr.get('grade') or '?'})  類似度:{sim_s}  結果:{result_str}  {kimarite}"
            )
    else:
        lines.append(click.style("\n【過去の似たようなレース】", fg="yellow", bold=True))
        lines.append("  (類似レースが見つかりませんでした — 学習データ不足の可能性)")

    # ── Ranking prediction ────────────────────────────────────────────────
    lines.append(click.style("\n【着順予想】", fg="yellow", bold=True))
    ranking = analysis.get("ranking", [])
    for rank, r in enumerate(ranking, 1):
        p1 = r.get("pred_prob_1st", 0.0)
        p2 = r.get("pred_prob_2nd", 0.0)
        p3 = r.get("pred_prob_3rd", 0.0)
        bar_len = max(1, int(p1 * 30))
        bar = "█" * bar_len + "░" * (30 - bar_len)
        name = str(r.get("name") or "?")[:8]
        row_str = (
            f"  予測{rank:>2}位: {r['car_no']:>2}番 {name:<8} "
            f"  1着:{p1:5.1%}  2着:{p2:5.1%}  3着:{p3:5.1%}  [{bar}]"
        )
        if rank <= 3:
            lines.append(click.style(row_str, fg="green" if rank == 1 else "bright_green"))
        else:
            lines.append(click.style(row_str, fg="bright_black"))

    # ── Bet picks ─────────────────────────────────────────────────────────
    if analysis.get("has_odds"):
        if analysis.get("picks"):
            lines.append(click.style("\n【三連単推奨】", fg="yellow", bold=True))
            for p in analysis["picks"]:
                combo = "-".join(str(c) for c in p["cars"])
                ev = p.get("ev", 0.0)
                stake = p.get("stake_yen", 0)
                ev_color = "green" if ev >= 1.50 else ("bright_green" if ev >= 1.30 else "white")
                row_str = (
                    f"  {combo}  オッズ:{p['odds']:5.1f}  確率:{p['prob']:5.1%}"
                    f"  EV:{ev:.2f}  賭け金:¥{stake}"
                )
                lines.append(click.style(row_str, fg=ev_color))
        else:
            lines.append(click.style("\n【三連単推奨】", fg="yellow", bold=True))
            lines.append("  EV基準を満たすbet候補なし (見送り推奨)")
    else:
        lines.append(click.style("\n  ※ オッズなし → 賭け推奨なし", fg="bright_black"))
        lines.append(click.style("    --with-odds を使うか fetch --kind odds を先に実行", fg="bright_black"))

    lines.append(click.style("\n" + "─" * W, fg="cyan"))
    return "\n".join(lines)
