"""
GitHub Actions 朝ジョブ (9:00 JST):
  1. 会場スキャン + 出走表取得 (login不要)
  2. オッズ取得 (NETKEIRIN_COOKIE が設定されていれば・予測精度向上のみ)
  3. 全レース着順予想 (analyze_race)
  4. HTML を docs/ に出力 → GitHub Pages で公開
"""
from __future__ import annotations
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from datetime import date, datetime
from pathlib import Path
from sqlalchemy import text

from keirin.config import load_app_config
from keirin.db.engine import get_engine, init_db
from keirin.db.repository import upsert_race, upsert_entries, insert_odds_snapshot, log_prediction
from keirin.features.style_norm import normalize_style
from keirin.scraper.session import fetch_text
from keirin.scraper.netkeirin import (
    entry_url, odds_url, parse_entries, parse_line_assignments, parse_trifecta_odds
)
from keirin.data.import_cache import VENUE_TABLE
from keirin.models.train import load_latest_model, load_latest_ranker
from keirin.models.predict import analyze_race
from keirin.features.builder import build_race_features
from keirin.reporting.analyze import write_daily_report_html

cfg = load_app_config()
eng = get_engine(cfg.paths.db)
init_db(eng)
today = date.today()
TODAY_STR = today.strftime("%Y%m%d")

# Cookie 認証 (GitHub Secret: NETKEIRIN_COOKIE)
# オッズは賭け用ではなく、乖離特徴量による予測精度向上のために使用
COOKIE = os.environ.get("NETKEIRIN_COOKIE", "")
HEADERS = {
    "User-Agent": "KeirinResearchBot/0.1 (personal; pagudaruma@gmail.com)",
    "Accept-Language": "ja-JP,ja;q=0.9",
}
if COOKIE:
    HEADERS["Cookie"] = COOKIE
    print("[auth] Cookie設定済み (オッズ取得→乖離特徴量で精度向上)")
else:
    print("[auth] Cookie未設定 (出走表+予想のみ)")

SLEEP = float(os.environ.get("REQUEST_SLEEP", "1.2"))


def _get(url: str) -> str:
    return fetch_text(url, headers=HEADERS, timeout=20)


def _to_int(v):
    try:
        return int(str(v).split(".")[0])
    except Exception:
        return 0


def _int_or_none(v):
    try:
        return int(str(v).split(".")[0])
    except Exception:
        return None


def _flt(v):
    try:
        return float(v)
    except Exception:
        return None


def _style(code):
    return normalize_style(code)


# ── 1. 会場スキャン ──────────────────────────────────────────────────────────
print(f"\n{today} 開催会場スキャン中...")
active_venues: list[int] = []
for vid_str in VENUE_TABLE:
    vc = int(vid_str)
    html = _get(entry_url(f"{TODAY_STR}{vc:02d}01"))
    time.sleep(SLEEP)
    if html and len(parse_entries(html)) >= 5:
        print(f"  ✅ {VENUE_TABLE[vid_str][0]}")
        active_venues.append(vc)

if not active_venues:
    print("本日開催なし")
    sys.exit(0)

# ── 2. 出走表取得 ─────────────────────────────────────────────────────────────
all_race_ids: list[str] = []
for vc in active_venues:
    vid = f"{vc:02d}"
    for rno in range(1, 13):
        rid = f"{TODAY_STR}{vc:02d}{rno:02d}"
        html = _get(entry_url(rid))
        time.sleep(SLEEP)
        if not html:
            continue
        raw = parse_entries(html)
        if len(raw) < 5:
            break
        enriched = parse_line_assignments(raw)
        grades = [e.get("grade") for e in enriched if e.get("grade")]
        rank_map = {"SS": 0, "S1": 1, "S2": 2, "A1": 3, "A2": 4, "A3": 5}
        grade = min(set(grades), key=lambda g: rank_map.get(g, 9)) if grades else None

        upsert_race(eng, {
            "race_id": rid, "date": today.isoformat(), "venue_id": vid,
            "race_no": rno, "grade": grade, "race_class": None,
            "distance_m": None, "weather": None, "track_cond": None, "post_time": None,
        })
        with eng.begin() as conn:
            for e in enriched:
                if not e.get("player_id"):
                    continue
                conn.execute(text(
                    "INSERT INTO players (player_id,name,style,rank_class,rating,updated_at)"
                    " VALUES (:pid,:name,:style,:rc,:rating,:upd)"
                    " ON CONFLICT(player_id) DO UPDATE SET name=excluded.name,"
                    "style=excluded.style,rank_class=excluded.rank_class,"
                    "rating=excluded.rating,updated_at=excluded.updated_at"
                ), {"pid": e["player_id"], "name": e.get("name", "?"),
                    "style": _style(e.get("style_code")), "rc": e.get("grade"),
                    "rating": _flt(e.get("race_score")), "upd": today.isoformat()})
        upsert_entries(eng, rid, [
            {"race_id": rid, "car_no": _to_int(e.get("car_no")),
             "player_id": e.get("player_id"), "rank_class": e.get("grade"),
             "rating": _flt(e.get("race_score")), "gear_ratio": _flt(e.get("gear")),
             "line_id": e.get("line_id"), "line_pos": e.get("line_pos"),
             "style": _style(e.get("style_code")), "scratched": 0,
             "nige_cnt": _int_or_none(e.get("nige")),
             "makuri_cnt": _int_or_none(e.get("makuri")),
             "sashi_cnt": _int_or_none(e.get("sashi")),
             "mark_cnt": _int_or_none(e.get("mark")),
             "s_count": _int_or_none(e.get("sprints")),
             "b_count": _int_or_none(e.get("backs")),
             "line_raw": e.get("line_raw") or None}
            for e in enriched
        ])
        all_race_ids.append(rid)

        # ライン割当の診断ログ: 1レースで割当できた車数 (0=並びなし/パーサ不調)
        n_lined = sum(1 for e in enriched if e.get("line_id") is not None)
        if n_lined == 0:
            sample_raw = next((e.get("line_raw") for e in enriched if e.get("line_raw")), "")
            print(f"    [line] {rid}: 割当0台 line_raw={sample_raw[:40]!r}")

print(f"出走表: {len(all_race_ids)} レース")

# ── 3. オッズ取得 (乖離特徴量用・賭け推奨には使わない) ─────────────────────
live_odds: dict[str, dict] = {}
if COOKIE:
    print("オッズ取得中 (予測精度向上のため)...")
    snap = datetime.now().isoformat(timespec="seconds")
    for rid in all_race_ids:
        html = _get(odds_url(rid))
        time.sleep(SLEEP)
        if not html:
            continue
        combos = parse_trifecta_odds(html)
        if combos:
            insert_odds_snapshot(eng, rid, snap, combos)
            live_odds[rid] = dict(combos)
    print(f"オッズ: {len(live_odds)} レース分取得")

# ── 4. 全レース着順予想 ───────────────────────────────────────────────────────
bundle = load_latest_model(cfg.paths.models)
if not bundle:
    print("モデルなし → python -m keirin retrain を先に実行してください")
    sys.exit(1)

model, feature_cols, calibrator = bundle
ranker_bundle = load_latest_ranker(cfg.paths.models)
ranker, ranker_features = ranker_bundle if ranker_bundle else (None, None)
model_files = sorted(cfg.paths.models.glob("lgbm_top3_*.pkl"))
model_version = model_files[-1].stem if model_files else "unknown"

if ranker:
    print(f"モデル: {model_version} + LambdaRank")
else:
    print(f"モデル: {model_version} (LambdaRankなし → PL近似)")

analyses: list[dict] = []
for rid in all_race_ids:
    vc = int(rid[8:10])
    rno = int(rid[10:12])
    venue_name = VENUE_TABLE.get(f"{vc:02d}", ("?", 400))[0]
    df = build_race_features(eng, rid, ref_date=today.isoformat(),
                             latest_odds=live_odds.get(rid))
    if df.empty:
        print(f"  {venue_name} {rno}R: 特徴量なし (スキップ)")
        continue

    analysis = analyze_race(
        eng, rid, df, model, feature_cols, calibrator,
        ranker=ranker, ranker_features=ranker_features,
        model_version=model_version,
    )
    analyses.append(analysis)

    ranking = analysis.get("ranking", [])
    top3_str = "→".join(str(r["car_no"]) for r in ranking[:3])
    model_tag = "LR" if analysis.get("used_ranker") else "PL"
    print(f"  {venue_name} {rno}R [{model_tag}]: 予測 {top3_str}")

    # 予測ログ記録 (record-result 後に精度追跡)
    if len(ranking) >= 3:
        try:
            log_prediction(
                eng, race_id=rid,
                pred_rank1_car=ranking[0]["car_no"],
                pred_rank2_car=ranking[1]["car_no"],
                pred_rank3_car=ranking[2]["car_no"],
                pred_rank1_prob=ranking[0]["pred_prob_1st"],
                pred_rank2_prob=ranking[1]["pred_prob_2nd"],
                pred_rank3_prob=ranking[2]["pred_prob_3rd"],
                model_version=model_version,
            )
        except Exception:
            pass

# ── 5. HTML 出力 → docs/ ─────────────────────────────────────────────────────
docs = Path("docs")
docs.mkdir(exist_ok=True)
out_html = docs / f"{today.isoformat()}.html"
write_daily_report_html(out_html, analyses)

(docs / "index.html").write_text(
    f'<!DOCTYPE html><meta charset="utf-8">'
    f'<meta http-equiv="refresh" content="0;url={today.isoformat()}.html">'
    f'<title>競輪着順予想 {today}</title>',
    encoding="utf-8",
)

print(f"\n完了: {len(analyses)} レース予想生成")
print(f"HTML: {out_html}")

