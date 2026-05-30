"""Terminal and HTML rendering for single-race deep analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import click

# ---------------------------------------------------------------------------
# Daily report HTML (full-day prediction page for GitHub Pages)
# ---------------------------------------------------------------------------

_DAILY_CSS = """
body{font-family:'Helvetica Neue',sans-serif;background:#0d0d0d;color:#e0e0e0;margin:0;padding:1rem 1.5rem}
h1{color:#ffd700;border-bottom:2px solid #333;padding-bottom:.5rem;margin-bottom:1.5rem}
.meta{color:#888;font-size:.85em;margin-top:-.5rem;margin-bottom:1.5rem}
.race-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:1.2rem}
.race-card{background:#141414;border:1px solid #2a2a2a;border-radius:8px;padding:1rem 1.2rem}
.race-header{display:flex;align-items:baseline;gap:.6rem;margin-bottom:.8rem}
.race-title{font-size:1.1em;font-weight:700;color:#ffd700}
.race-grade{font-size:.8em;color:#aaa;background:#222;padding:.15rem .4rem;border-radius:3px}
.race-model{font-size:.75em;color:#4caf50;margin-left:auto}
.rank-table{width:100%;border-collapse:collapse;font-size:.88em}
.rank-table th{color:#888;font-weight:400;text-align:left;padding:.25rem .4rem;border-bottom:1px solid #2a2a2a}
.rank-table td{padding:.3rem .4rem;border-bottom:1px solid #1e1e1e}
.rank-1{color:#ffd700;font-weight:700}
.rank-2{color:#c0c0c0}
.rank-3{color:#cd7f32}
.rank-rest{color:#555}
.prob-bar{height:8px;background:#222;border-radius:4px;overflow:hidden;margin-top:2px;min-width:60px}
.prob-fill{height:100%;background:#4caf50;border-radius:4px}
.lines-section{margin-top:.7rem;font-size:.82em;color:#aaa}
.lines-label{color:#666;margin-right:.4rem}
.line-group{display:inline-block;margin-right.8rem;background:#1e1e1e;padding:.15rem .4rem;border-radius:3px}
.mismatch-warn{color:#ff7043}
footer{margin-top:2rem;color:#444;font-size:.8em;text-align:center}
"""


def write_daily_report_html(out_path: Path, analyses: list[dict[str, Any]]) -> None:
    """Write a full-day ranking prediction report HTML from analyze_race() results.

    Produces a responsive card layout — one card per race, each showing the
    predicted finish order with Plackett-Luce probabilities. No betting content.
    """
    if not analyses:
        date_s = "—"
    else:
        date_s = analyses[0].get("date", "—")

    race_cards = "\n".join(_race_card_html(a) for a in analyses)
    generated = Path(__file__).parent  # just for import; use datetime at call site
    from datetime import datetime
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M JST")

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>競輪着順予想 {date_s}</title>"
        f"<style>{_DAILY_CSS}</style></head><body>\n"
        f"<h1>競輪着順予想 {date_s}</h1>\n"
        f'<p class="meta">生成: {generated_at} &nbsp;|&nbsp; {len(analyses)} レース</p>\n'
        f'<div class="race-grid">\n{race_cards}\n</div>\n'
        "<footer>賭けるかどうかはオッズとご自身の判断で</footer>\n"
        "</body></html>"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def _race_card_html(analysis: dict[str, Any]) -> str:
    venue = analysis.get("venue_name", "?")
    race_no = analysis.get("race_no", "?")
    grade = analysis.get("grade") or ""
    model_tag = "LambdaRank" if analysis.get("used_ranker") else "PL近似"
    ranking = analysis.get("ranking", [])
    lines = analysis.get("lines", [])

    # Ranking rows
    rows = ""
    for rank, r in enumerate(ranking, 1):
        p1 = r.get("pred_prob_1st", 0.0)
        p2 = r.get("pred_prob_2nd", 0.0)
        p3 = r.get("pred_prob_3rd", 0.0)
        name = str(r.get("name") or "?")[:8]
        cls = str(r.get("rank_class") or "")
        bar_w = max(2, int(p1 * 100))
        if rank == 1:
            tr_cls = "rank-1"
        elif rank == 2:
            tr_cls = "rank-2"
        elif rank == 3:
            tr_cls = "rank-3"
        else:
            tr_cls = "rank-rest"
        p1_s = f"{p1:.1%}"
        p2_s = f"{p2:.1%}"
        p3_s = f"{p3:.1%}"
        rows += (
            f'<tr class="{tr_cls}">'
            f"<td>{rank}</td><td>{r['car_no']}</td>"
            f"<td>{name}</td><td>{cls}</td>"
            f"<td>{p1_s}"
            f'<div class="prob-bar"><div class="prob-fill" style="width:{bar_w}%"></div></div>'
            f"</td>"
            f"<td>{p2_s}</td><td>{p3_s}</td>"
            f"</tr>\n"
        )

    # Line formation
    lines_html = ""
    if lines:
        parts = []
        for ln in lines:
            cars_s = "─".join(str(c) for c in ln.get("cars", []))
            mm = ln.get("mismatch_score", 0.0) or 0.0
            warn = ' class="mismatch-warn"' if mm > 3.0 else ""
            parts.append(f'<span class="line-group"{warn}>{cars_s}</span>')
        lines_html = (
            f'<div class="lines-section">'
            f'<span class="lines-label">ライン:</span>'
            f'{"&nbsp; ".join(parts)}</div>'
        )

    return (
        f'<div class="race-card">\n'
        f'  <div class="race-header">'
        f'    <span class="race-title">{venue} {race_no}R</span>'
        f'    <span class="race-grade">{grade}</span>'
        f'    <span class="race-model">{model_tag}</span>'
        f"  </div>\n"
        f"  <table class=\"rank-table\">\n"
        f"    <thead><tr><th>#</th><th>車</th><th>選手名</th><th>階級</th>"
        f"    <th>P(1着)</th><th>P(2着)</th><th>P(3着)</th></tr></thead>\n"
        f"    <tbody>\n{rows}    </tbody>\n"
        f"  </table>\n"
        f"  {lines_html}\n"
        f"</div>"
    )


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

    # ── Ranking prediction (the core output) ─────────────────────────────
    model_label = "LambdaRank" if analysis.get("used_ranker") else "PL-from-Top3"
    lines.append(click.style(f"\n【着順予想】 ({model_label})", fg="yellow", bold=True))
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
        if rank == 1:
            lines.append(click.style(row_str, fg="green", bold=True))
        elif rank <= 3:
            lines.append(click.style(row_str, fg="bright_green"))
        else:
            lines.append(click.style(row_str, fg="bright_black"))

    # Top 3 confidence summary
    if len(ranking) >= 3:
        top3_combined = sum(r.get("pred_prob_1st", 0) for r in ranking[:3])
        lines.append(
            click.style(
                f"\n  上位3車の合計1着確率: {top3_combined:.1%}",
                fg="cyan",
            )
        )

    lines.append(click.style("\n" + "─" * W, fg="cyan"))
    lines.append(click.style("  賭けるかどうかはオッズとあなたの判断で決めてください。", fg="bright_black"))
    return "\n".join(lines)
