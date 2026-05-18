"""CLI entry point: `python -m keirin <command>`."""
from __future__ import annotations

import logging
from datetime import date as _date
from datetime import datetime, timedelta
from pathlib import Path

import click

from keirin.config import HARD_DAILY_BUDGET_YEN, load_app_config, load_betting_config
from keirin.db.engine import get_engine, init_db
from keirin.db.repository import month_pnl, settle_bets_for_race
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
    for rid in race_ids:
        kj.fetch_result(rid)
        settled += settle_bets_for_race(kj.engine, rid)
    click.echo(f"settled {settled} bets across {len(race_ids)} races")


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
