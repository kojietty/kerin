"""CLI entry point: `python -m keirin <command>`."""
from __future__ import annotations

import logging
from datetime import date as _date
from datetime import datetime, timedelta
from pathlib import Path

import click

from keirin.config import HARD_DAILY_BUDGET_YEN, load_app_config, load_betting_config
from keirin.db.engine import get_engine, init_db
from keirin.db.repository import accuracy_metrics, log_prediction, month_pnl, settle_bets_for_race, settle_prediction_log
from keirin.logging_setup import setup_logging
from keirin.reporting.html_renderer import example_payload, write_dashboard
from keirin.reporting.markdown import write_markdown

log = logging.getLogger(__name__)


def _parse_day(s: str) -> _date:
    s = s.strip().lower()
    if s == "today":
        return _date.today()
    if s == "yesterday":
        return _date.today() - timedelta(days=1)
    if s == "tomorrow":
        return _date.today() + timedelta(days=1)
    return _date.fromisoformat(s)


@click.group()
@click.option("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
def main(log_level: str | None) -> None:
    cfg = load_app_config()
    setup_logging(cfg.paths.logs, level=log_level or cfg.logging.level)


@main.command("init-db")
def init_db_cmd() -> None:
    """Create the SQLite schema (idempotent)."""
    cfg = load_app_config()
    engine = get_engine(cfg.paths.db)
    init_db(engine)
    click.echo(f"DB initialized at {cfg.paths.db}")


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

@main.command("fetch")
@click.option("--date", "date_s", default="today", help="today/yesterday/YYYY-MM-DD")
@click.option(
    "--kind",
    type=click.Choice(["schedule", "cards", "odds", "results", "all"]),
    default="all",
)
@click.option("--race-id", default=None, help="Restrict to a single race (cards/odds/results)")
def fetch_cmd(date_s: str, kind: str, race_id: str | None) -> None:
    """Fetch data from KEIRIN.JP into local SQLite + raw HTML cache."""
    from keirin.scraper.keirin_jp import KeirinJp

    day = _parse_day(date_s)
    kj = KeirinJp()
    if kind in ("schedule", "all"):
        races = kj.fetch_schedule(day)
        click.echo(f"[schedule] {len(races)} races on {day}")

    if kind in ("cards", "all"):
        race_ids = [race_id] if race_id else _race_ids_for(kj, day)
        for rid in race_ids:
            kj.fetch_card(rid)
        click.echo(f"[cards] fetched {len(race_ids)} cards")

    if kind in ("odds", "all"):
        race_ids = [race_id] if race_id else _race_ids_for(kj, day)
        n = 0
        for rid in race_ids:
            n += kj.fetch_odds(rid)
        click.echo(f"[odds] inserted {n} combo rows across {len(race_ids)} races")

    if kind == "results":
        race_ids = [race_id] if race_id else _race_ids_for(kj, day)
        for rid in race_ids:
            kj.fetch_result(rid)
        click.echo(f"[results] fetched {len(race_ids)} races")


def _race_ids_for(kj, day: _date) -> list[str]:
    from sqlalchemy import text
    with kj.engine.begin() as conn:
        rows = conn.execute(
            text("SELECT race_id FROM races WHERE date = :d ORDER BY race_id"),
            {"d": day.isoformat()},
        ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------

@main.command("backfill")
@click.option("--from", "from_s", required=True, help="YYYY-MM-DD")
@click.option("--to", "to_s", required=True, help="YYYY-MM-DD inclusive")
def backfill_cmd(from_s: str, to_s: str) -> None:
    """Sweep historical schedule/cards/odds/results across a date range."""
    from keirin.scraper.keirin_jp import KeirinJp

    start = _date.fromisoformat(from_s)
    end = _date.fromisoformat(to_s)
    kj = KeirinJp()
    cur = start
    total = 0
    while cur <= end:
        races = kj.fetch_schedule(cur)
        for r in races:
            kj.fetch_card(r["race_id"])
            kj.fetch_result(r["race_id"])
            total += 1
        click.echo(f"{cur}: +{len(races)} races (running total {total})")
        cur = cur + timedelta(days=1)


# ---------------------------------------------------------------------------
# predict (skeleton)
# ---------------------------------------------------------------------------

@main.command("predict")
@click.option("--date", "date_s", default="today")
@click.option("--with-odds/--no-odds", default=True, help="Fetch live odds before predicting")
def predict_cmd(date_s: str, with_odds: bool) -> None:
    """Run full prediction pipeline → write HTML + Markdown dashboard."""
    from keirin.models.predict import predict_today
    from keirin.reporting.pnl import build_pnl_context

    cfg = load_app_config()
    bcfg = load_betting_config()
    day = _parse_day(date_s)
    engine = get_engine(cfg.paths.db)
    init_db(engine)

    # Always show MTD PnL first (dependency-prevention)
    ym = day.strftime("%Y-%m")
    pnl_s = month_pnl(engine, ym)
    _print_pnl(pnl_s, ym, bcfg)

    # Check emergency stop
    from keirin.config import HARD_EMERGENCY_STOP_YEN
    if pnl_s["pnl"] <= HARD_EMERGENCY_STOP_YEN:
        click.secho(
            f"\n緊急停止中: 月初累積 ¥{pnl_s['pnl']:+,d} ≤ ¥{HARD_EMERGENCY_STOP_YEN:,d}。"
            " 予想は生成しますがペーパー記録のみ推奨。",
            fg="red",
        )

    # Get today's race_ids
    race_ids = _race_ids_for_day(engine, day)
    if not race_ids:
        click.echo(f"No races found for {day}. Run: python -m keirin fetch --date {day_s} --kind schedule")
        return

    click.echo(f"Predicting {len(race_ids)} races on {day} ...")

    # Optionally fetch live odds
    latest_odds: dict[str, dict] = {}
    if with_odds:
        from keirin.scraper.keirin_jp import KeirinJp
        kj = KeirinJp()
        for rid in race_ids:
            kj.fetch_odds(rid)
        # Load from DB
        from sqlalchemy import text
        with engine.begin() as conn:
            rows = conn.execute(text(
                """
                SELECT o.race_id, o.combo, o.odds
                FROM odds_trifecta o
                WHERE o.race_id IN ({})
                  AND o.snapshot_at = (
                    SELECT MAX(snapshot_at) FROM odds_trifecta o2
                    WHERE o2.race_id = o.race_id
                  )
                """.format(",".join(f"'{r}'" for r in race_ids))
            )).fetchall()
        for rid, combo, odds in rows:
            latest_odds.setdefault(rid, {})[combo] = odds

    # Run prediction
    has_model = any(cfg.paths.models.glob("lgbm_top3_*.pkl"))
    if not has_model:
        click.echo("No trained model. Generating sample dashboard. Run `python -m keirin retrain` first.")
        payload = example_payload(date_label=day.isoformat())
    else:
        result = predict_today(
            engine, race_ids, cfg.paths.models,
            latest_odds_by_race=latest_odds or None,
        )
        payload = _build_dashboard_payload(result, day, pnl_s, bcfg)

    out_html = cfg.paths.reports / "predictions" / f"{day.isoformat()}.html"
    out_md   = cfg.paths.reports / "predictions" / f"{day.isoformat()}.md"
    write_dashboard(out_html, payload)
    if cfg.reporting.output_markdown:
        write_markdown(out_md, payload)
    click.echo(f"→ {out_html}")
    click.echo(f"→ {out_md}")
    click.echo(f"推奨レース: {len(payload.get('recommended', []))}  見送り: {len(payload.get('skipped', []))}")


def _print_pnl(s: dict, ym: str, bcfg) -> None:
    pnl = s["pnl"]
    roi = f"{s['roi']:.0%}" if s["roi"] else "—"
    click.echo(f"\n{ym} 収支: ¥{pnl:+,d}  ROI={roi}  的中={s['hits']}/{s['bet_count']}")
    from keirin.config import HARD_EMERGENCY_STOP_YEN, HARD_DAILY_BUDGET_YEN
    bar = int(max(0, (pnl - HARD_EMERGENCY_STOP_YEN) / (HARD_DAILY_BUDGET_YEN * 30) * 20))
    click.echo("  [" + "█" * bar + "░" * (20 - bar) + f"]  残り上限: ¥{HARD_EMERGENCY_STOP_YEN - pnl:+,d}")


def _race_ids_for_day(engine, day) -> list[str]:
    from sqlalchemy import text
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT race_id FROM races WHERE date=:d ORDER BY race_id"),
            {"d": day.isoformat()},
        ).fetchall()
    return [r[0] for r in rows]


def _build_dashboard_payload(result: dict, day, pnl_s: dict, bcfg) -> dict:
    from keirin.config import HARD_DAILY_BUDGET_YEN
    from datetime import datetime
    daily_invest = sum(
        sum(p["stake_yen"] for p in r["picks"])
        for r in result.get("recommended", [])
    )
    return {
        "date_label": day.isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M JST"),
        "mtd_pnl": pnl_s["pnl"],
        "mtd_roi": pnl_s.get("roi"),
        "mtd_hit_rate": pnl_s.get("hit_rate"),
        "daily_invest": daily_invest,
        "daily_remaining": HARD_DAILY_BUDGET_YEN - daily_invest,
        "daily_cap": HARD_DAILY_BUDGET_YEN,
        "total_races": len(result.get("recommended", [])) + len(result.get("skipped", [])),
        "ev_threshold": bcfg.ev_threshold,
        "emergency_stop": pnl_s["pnl"] <= -3000,
        "recommended": result.get("recommended", []),
        "skipped": result.get("skipped", []),
    }


# ---------------------------------------------------------------------------
# record-result
# ---------------------------------------------------------------------------

@main.command("import-cache")
@click.argument("file", type=click.Path(exists=True))
@click.option("--skip-existing/--no-skip-existing", default=True, help="Skip races already in DB")
def import_cache_cmd(file: str, skip_existing: bool) -> None:
    """Import races from D:\\keirin\\backtest\\real_race_cache.json into SQLite."""
    from keirin.data.import_cache import import_cache
    from pathlib import Path

    cfg = load_app_config()
    engine = get_engine(cfg.paths.db)
    init_db(engine)

    click.echo(f"Importing from {file} ...")
    stats = import_cache(engine, Path(file), skip_existing=skip_existing, verbose=True)
    click.echo(
        f"Done: races={stats['races']}, entries={stats['entries']}, "
        f"results={stats['results']}, payouts={stats['payouts']}, "
        f"skipped={stats['skipped']}"
    )


@main.command("sync-log")
def sync_log_cmd() -> None:
    """Derive player_race_log from entries+results already in DB. Run after backfill."""
    from keirin.db.repository import sync_player_race_log

    cfg = load_app_config()
    engine = get_engine(cfg.paths.db)
    init_db(engine)
    n = sync_player_race_log(engine)
    click.echo(f"sync_player_race_log: inserted {n} rows")


@main.command("record-result")
@click.option("--date", "date_s", default="yesterday")
def record_result_cmd(date_s: str) -> None:
    """Fetch results for `date` and settle bets (compute payouts/hit)."""
    from keirin.scraper.keirin_jp import KeirinJp

    day = _parse_day(date_s)
    kj = KeirinJp()
    race_ids = _race_ids_for(kj, day)
    settled = 0
    settled_preds = 0
    for rid in race_ids:
        kj.fetch_result(rid)
        settled += settle_bets_for_race(kj.engine, rid)
        settled_preds += settle_prediction_log(kj.engine, rid)
    click.echo(f"settled {settled} bets, {settled_preds} predictions across {len(race_ids)} races")


# ---------------------------------------------------------------------------
# pnl
# ---------------------------------------------------------------------------

@main.command("pnl")
@click.option("--month", "month_s", default="current", help="current/YYYY-MM")
def pnl_cmd(month_s: str) -> None:
    """Print month-to-date P&L summary."""
    cfg = load_app_config()
    engine = get_engine(cfg.paths.db)
    init_db(engine)
    ym = datetime.now().strftime("%Y-%m") if month_s == "current" else month_s
    s = month_pnl(engine, ym)
    click.echo(f"month: {ym}")
    click.echo(f"  bets   : {s['bet_count']}")
    click.echo(f"  hits   : {s['hits']}")
    click.echo(f"  stake  : ¥{s['stake']:,}")
    click.echo(f"  payout : ¥{s['payout']:,}")
    click.echo(f"  P&L    : ¥{s['pnl']:+,d}")
    roi = f"{s['roi']:.0%}" if s["roi"] is not None else "—"
    hr = f"{s['hit_rate']:.0%}" if s["hit_rate"] is not None else "—"
    click.echo(f"  ROI    : {roi}")
    click.echo(f"  HitRate: {hr}")


# ---------------------------------------------------------------------------
# sample-dashboard
# ---------------------------------------------------------------------------

@main.command("sample-dashboard")
@click.option("--out", "out_path", default=None, help="Output HTML path")
def sample_dashboard_cmd(out_path: str | None) -> None:
    """Render an example dashboard to verify the renderer + design system."""
    cfg = load_app_config()
    payload = example_payload()
    out = Path(out_path) if out_path else cfg.paths.reports / "predictions" / "sample.html"
    md_out = out.with_suffix(".md")
    write_dashboard(out, payload)
    write_markdown(md_out, payload)
    click.echo(f"wrote {out}")
    click.echo(f"wrote {md_out}")
    click.echo(f"Open in browser: {out.absolute().as_uri()}")


# ---------------------------------------------------------------------------
# analyze — interactive single-race deep analysis
# ---------------------------------------------------------------------------

@main.command("analyze")
@click.argument("race_id", required=False)
@click.option("--date", "date_s", default="today", help="today/yesterday/YYYY-MM-DD")
@click.option("--venue", "venue_s", default=None, help="会場コード 2桁 e.g. 11")
@click.option("--race-no", "race_no", default=None, type=int, help="レース番号 e.g. 11")
@click.option("--with-odds/--no-odds", default=True,
              help="オッズを取得して乖離特徴量に利用 (予測精度向上のため・賭け推奨はしない)")
@click.option("--similar-n", default=5, type=int, help="類似レース表示件数")
@click.option("--out", "out_path", default=None, help="HTML出力パス")
def analyze_cmd(
    race_id: str | None,
    date_s: str,
    venue_s: str | None,
    race_no: int | None,
    with_odds: bool,
    similar_n: int,
    out_path: str | None,
) -> None:
    """単レース着順予想: 出走データ・ライン並び・類似レース・予測順位。

    予測精度に特化したコマンド。賭けるかどうかはユーザーが判断。
    """
    from keirin.models.predict import analyze_race
    from keirin.models.train import load_latest_model, load_latest_ranker
    from keirin.features.builder import build_race_features
    from keirin.reporting.analyze import render_analysis_terminal
    from sqlalchemy import text

    cfg = load_app_config()
    engine = get_engine(cfg.paths.db)
    init_db(engine)
    day = _parse_day(date_s)

    # --- Resolve race_id ---
    if race_id is None:
        if venue_s is not None and race_no is not None:
            race_id = f"{day.strftime('%Y%m%d')}{venue_s.zfill(2)}{str(race_no).zfill(2)}"
        else:
            race_ids = _race_ids_for_day(engine, day)
            if not race_ids:
                click.echo(f"レースが見つかりません: {day}  →  fetch コマンドを先に実行してください")
                return
            click.echo(f"\n{day} のレース一覧:")
            for i, rid in enumerate(race_ids, 1):
                click.echo(f"  {i:>2}. {rid}")
            idx = click.prompt("番号を選択", type=int, default=1) - 1
            if idx < 0 or idx >= len(race_ids):
                click.echo("無効な番号です")
                return
            race_id = race_ids[idx]

    date_iso = f"{race_id[0:4]}-{race_id[4:6]}-{race_id[6:8]}"
    click.echo(f"\n対象レース: {race_id}  ({date_iso})")

    # --- Optionally fetch odds ---
    latest_odds: dict[str, float] | None = None
    if with_odds:
        try:
            from keirin.scraper.keirin_jp import KeirinJp
            kj = KeirinJp()
            n_odds = kj.fetch_odds(race_id)
            if n_odds > 0:
                with engine.begin() as conn:
                    rows = conn.execute(text(
                        """
                        SELECT combo, odds FROM odds_trifecta
                        WHERE race_id = :rid
                          AND snapshot_at = (
                            SELECT MAX(snapshot_at) FROM odds_trifecta
                            WHERE race_id = :rid2
                          )
                        """
                    ), {"rid": race_id, "rid2": race_id}).fetchall()
                latest_odds = {r[0]: r[1] for r in rows}
                click.echo(f"オッズ取得: {len(latest_odds)} コンボ")
            else:
                click.echo("オッズ取得できませんでした (--no-odds モードで継続)")
        except Exception as exc:
            click.echo(f"オッズ取得エラー: {exc} — 予想のみ実行")

    # --- Load model ---
    has_model = any(cfg.paths.models.glob("lgbm_top3_*.pkl"))
    if not has_model:
        click.echo("モデルが見つかりません。先に: python -m keirin retrain")
        return

    bundle = load_latest_model(cfg.paths.models)
    if bundle is None:
        click.echo("モデル読み込み失敗。retrain を実行してください。")
        return
    model, feature_cols, calibrator = bundle

    ranker_bundle = load_latest_ranker(cfg.paths.models)
    ranker, ranker_features = ranker_bundle if ranker_bundle else (None, None)
    if ranker is None:
        click.echo("注: LambdaRank モデルなし → PL近似で着順予想 (retrain で精度向上)")

    # --- Build features ---
    click.echo("特徴量を構築中...")
    df = build_race_features(engine, race_id, ref_date=date_iso, latest_odds=latest_odds)
    if df.empty:
        click.echo(f"出走データが見つかりません: {race_id}  → fetch --kind cards を実行してください")
        return

    # --- Model version ---
    model_files = sorted(cfg.paths.models.glob("lgbm_top3_*.pkl"))
    model_version = model_files[-1].stem if model_files else "unknown"

    # --- Analyze (prediction-only, no betting) ---
    click.echo("予測中...")
    result = analyze_race(
        engine, race_id, df, model, feature_cols, calibrator,
        ranker=ranker, ranker_features=ranker_features,
        similar_n=similar_n,
        model_version=model_version,
    )

    # --- Print terminal report ---
    click.echo(render_analysis_terminal(result))

    # --- Log prediction for feedback loop ---
    ranking = result.get("ranking", [])
    if len(ranking) >= 3:
        try:
            log_prediction(
                engine,
                race_id=race_id,
                pred_rank1_car=ranking[0]["car_no"],
                pred_rank2_car=ranking[1]["car_no"],
                pred_rank3_car=ranking[2]["car_no"],
                pred_rank1_prob=ranking[0]["pred_prob_1st"],
                pred_rank2_prob=ranking[1]["pred_prob_2nd"],
                pred_rank3_prob=ranking[2]["pred_prob_3rd"],
                model_version=model_version,
            )
            click.echo("予測をログに記録しました (record-result で精度追跡)")
        except Exception as exc:
            log.debug("prediction log failed: %s", exc)

    # --- HTML output ---
    if out_path:
        _write_analyze_html(result, Path(out_path))
        click.echo(f"→ {out_path}")


def _write_analyze_html(analysis: dict, out_path: Path) -> None:
    """Write a minimal standalone HTML report for the analysis."""
    from keirin.reporting.html_renderer import _ASSETS_DIR

    ranking_rows = ""
    for rank, r in enumerate(analysis.get("ranking", []), 1):
        p1 = r.get("pred_prob_1st", 0)
        color = "#4caf50" if rank <= 3 else "#888"
        ranking_rows += (
            f"<tr style='color:{color}'>"
            f"<td>{rank}</td><td>{r['car_no']}</td><td>{r.get('name','?')}</td>"
            f"<td>{p1:.1%}</td><td>{r.get('pred_prob_2nd',0):.1%}</td>"
            f"<td>{r.get('pred_prob_3rd',0):.1%}</td></tr>\n"
        )

    similar_rows = ""
    for sr in analysis.get("similar_races", []):
        result_str = "→".join(str(c) for c in sr.get("result", []))
        similar_rows += (
            f"<tr><td>{sr['date']}</td><td>{sr['venue_name']}{sr['race_no']}R</td>"
            f"<td>{sr.get('grade','')}</td><td>{sr['similarity_score']:.2f}</td>"
            f"<td>{result_str}</td><td>{sr.get('winner_style','') or ''}</td></tr>\n"
        )

    css = ""
    try:
        css = _ASSETS_DIR.joinpath("style.css").read_text(encoding="utf-8")
    except Exception:
        pass

    # Build participant rows outside the main f-string to avoid nested f-string issues
    # (Python 3.11 does not support same-delimiter nested f-strings).
    participant_rows = ""
    for p in analysis.get("participants", []):
        rating_s = f"{(p.get('rating') or 0):.1f}"
        rest_s = f"{int(p['rest_days'])}日" if p.get("rest_days") is not None else "?"
        t3_s = f"{p['top3_rate_5']:.0%}" if p.get("top3_rate_5") is not None else "?"
        participant_rows += (
            f"<tr><td>{p['car_no']}</td><td>{p.get('name', '?')}</td>"
            f"<td>{p.get('rank_class', '')}</td><td>{rating_s}</td>"
            f"<td>{rest_s}</td><td>{t3_s}</td>"
            f"<td>{p.get('style', '')}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<title>{analysis.get('date','')} {analysis.get('venue_name','')} {analysis.get('race_no','')}R 分析</title>
<style>{css}
body{{font-family:monospace;background:#0a0a0a;color:#e0e0e0;padding:1rem}}
table{{border-collapse:collapse;width:100%;margin-bottom:1.5rem}}
th,td{{border:1px solid #333;padding:.4rem .7rem;text-align:left}}
th{{background:#1a1a1a;color:#ffd700}}
h2{{color:#ffd700;margin-top:2rem}}
</style></head><body>
<h1>{analysis.get('date','')} {analysis.get('venue_name','')} {analysis.get('race_no','')}R
  <span style="color:#888;font-size:.8em">{analysis.get('grade','')}</span></h1>

<h2>出走選手</h2>
<table><tr><th>車</th><th>選手名</th><th>階級</th><th>得点</th><th>休養</th>
<th>3着内率(5走)</th><th>スタイル</th></tr>
{participant_rows}</table>

<h2>着順予想</h2>
<table><tr><th>予測順位</th><th>車</th><th>選手名</th>
<th>P(1着)</th><th>P(2着)</th><th>P(3着)</th></tr>
{ranking_rows}</table>

<h2>過去の似たようなレース</h2>
<table><tr><th>日付</th><th>会場</th><th>グレード</th><th>類似度</th><th>結果</th><th>決まり手</th></tr>
{similar_rows if similar_rows else '<tr><td colspan="6">類似レースなし</td></tr>'}
</table>
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# metrics — prediction accuracy tracking
# ---------------------------------------------------------------------------

@main.command("metrics")
@click.option("--from", "from_s", default=None, help="YYYY-MM-DD (省略時: 30日前)")
@click.option("--to", "to_s", default=None, help="YYYY-MM-DD (省略時: 今日)")
@click.option("--model", "model_v", default=None, help="モデルバージョンでフィルタ")
def metrics_cmd(from_s: str | None, to_s: str | None, model_v: str | None) -> None:
    """予測精度レポート: 1位的中率・3着内的中率・三連単完全一致率の推移。"""
    from datetime import date as _dt, timedelta

    cfg = load_app_config()
    engine = get_engine(cfg.paths.db)
    init_db(engine)

    from_date = from_s or (_dt.today() - timedelta(days=30)).isoformat()
    to_date = to_s or _dt.today().isoformat()

    m = accuracy_metrics(engine, from_date=from_date, to_date=to_date, model_version=model_v)

    click.echo(f"\n== 予測精度レポート ==")
    click.echo(f"期間: {from_date} 〜 {to_date}" + (f"  モデル: {model_v}" if model_v else ""))
    click.echo(f"─" * 50)
    click.echo(f"総予測数   : {m['total_predictions']}")
    click.echo(f"決着済み   : {m['settled']}")

    def _fmt(v: float | None) -> str:
        return f"{v:.1%}" if v is not None else "—"

    r1 = m["rank1_hit_rate"]
    t3 = m["top3_hit_rate"]
    ex = m["trifecta_exact_hit_rate"]

    r1_color = "green" if (r1 or 0) >= 0.40 else ("yellow" if (r1 or 0) >= 0.30 else "red")
    t3_color = "green" if (t3 or 0) >= 0.60 else ("yellow" if (t3 or 0) >= 0.45 else "red")

    click.echo(f"1位的中率  : {click.style(_fmt(r1), fg=r1_color)}  (目標: 35-45%)")
    click.echo(f"3着内的中率: {click.style(_fmt(t3), fg=t3_color)}  (目標: 55-65%)")
    click.echo(f"三連単一致 : {_fmt(ex)}  (目標: 10-20%)")

    if m["monthly"]:
        click.echo(f"\n月別推移:")
        for row in m["monthly"]:
            r1r = row.get("rank1_rate")
            t3r = row.get("top3_rate")
            bar_len = max(0, int((r1r or 0) * 20))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            click.echo(
                f"  {row['month']}  n={row['count']:>3}"
                f"  1位:{_fmt(r1r):>6}  3着内:{_fmt(t3r):>6}  [{bar}]"
            )

    click.echo(
        click.style(
            "\n注: 三連単80-90%的中は控除率上不可能。"
            " 3着内65%・1位40%超が現実的な高精度目標です。",
            fg="bright_black",
        )
    )


# ---------------------------------------------------------------------------
# retrain / backtest (skeletons, Phase 2-3)
# ---------------------------------------------------------------------------

@main.command("retrain")
@click.option("--from", "from_s", default=None, help="YYYY-MM-DD (default: 90 days ago)")
@click.option("--to", "to_s", default=None, help="YYYY-MM-DD (default: today)")
@click.option("--eval/--no-eval", "do_eval", default=True, help="Run walk-forward eval")
def retrain_cmd(from_s: str | None, to_s: str | None, do_eval: bool) -> None:
    """Re-train LightGBM top-3 model on local SQLite data."""
    from keirin.models.train import retrain_and_save
    from keirin.config import load_app_config

    cfg = load_app_config()
    engine = get_engine(cfg.paths.db)
    init_db(engine)

    click.echo("Building training frame...")
    model_path = retrain_and_save(
        engine, cfg.paths.models,
        from_date=from_s, to_date=to_s,
    )
    click.echo(f"Model saved: {model_path}")


@main.command("backtest")
@click.option("--from", "from_s", required=True)
@click.option("--to", "to_s", required=True)
def backtest_cmd(from_s: str, to_s: str) -> None:
    """Walk-forward backtest (Phase 3 task)."""
    _ = (from_s, to_s)
    click.echo("backtest: not implemented yet (Phase 3)")


# ---------------------------------------------------------------------------
# safety: doctor
# ---------------------------------------------------------------------------

@main.command("doctor")
def doctor_cmd() -> None:
    """Verify environment and safety invariants."""
    from keirin.config import HARD_EMERGENCY_STOP_YEN

    cfg = load_app_config()
    bcfg = load_betting_config()
    click.echo("== Configuration ==")
    click.echo(f"  db                  : {cfg.paths.db}")
    click.echo(f"  reports             : {cfg.paths.reports}")
    click.echo(f"  daily_budget_yen    : {bcfg.daily_budget_yen}  (hard cap {HARD_DAILY_BUDGET_YEN})")
    click.echo(f"  emergency_stop_yen  : {bcfg.emergency_stop_drawdown_yen}  (hard floor {HARD_EMERGENCY_STOP_YEN})")
    click.echo(f"  ev_threshold        : {bcfg.ev_threshold}")
    click.echo(f"  odds_band           : [{bcfg.odds_min}, {bcfg.odds_max}]")
    click.echo(f"  combo_method        : {bcfg.combo_method}")
    assert bcfg.daily_budget_yen <= HARD_DAILY_BUDGET_YEN, "daily budget exceeds hard cap"
    assert bcfg.emergency_stop_drawdown_yen >= HARD_EMERGENCY_STOP_YEN, "emergency stop below hard floor"
    click.echo("safety invariants OK")
