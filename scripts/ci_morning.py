"""
GitHub Actions 朝ジョブ (9:00 JST):
  1. 会場スキャン + 出走表取得 (login不要)
  2. オッズ取得 (NETKEIRIN_COOKIE が設定されていれば)
  3. LightGBM 予想生成
  4. HTML/MD を docs/ に出力 → GitHub Pages で公開
"""
from __future__ import annotations
import sys, io, os, re, time, json, requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from datetime import date, datetime
from pathlib import Path
from sqlalchemy import text

from keirin.config import load_app_config, load_betting_config
from keirin.config import HARD_DAILY_BUDGET_YEN, HARD_EMERGENCY_STOP_YEN
from keirin.db.engine import get_engine, init_db
from keirin.db.repository import upsert_race, upsert_entries, insert_odds_snapshot, month_pnl
from keirin.scraper.netkeirin import (
    entry_url, odds_url, parse_entries, parse_line_assignments, parse_trifecta_odds
)
from keirin.data.import_cache import VENUE_TABLE
from keirin.models.train import load_latest_model
from keirin.models.combo import expand_trifecta
from keirin.models.predict import predict_race
from keirin.features.builder import build_race_features
from keirin.betting.ev import build_candidates
from keirin.betting.selector import select_picks
from keirin.betting.stake import plan_stakes
from keirin.reporting.html_renderer import write_dashboard
from keirin.reporting.markdown import write_markdown

cfg = load_app_config()
bcfg = load_betting_config()
eng = get_engine(cfg.paths.db)
init_db(eng)
today = date.today()
TODAY_STR = today.strftime("%Y%m%d")

# Cookie 認証 (GitHub Secret: NETKEIRIN_COOKIE)
COOKIE = os.environ.get("NETKEIRIN_COOKIE", "")
HEADERS = {
    "User-Agent": "KeirinResearchBot/0.1 (personal; pagudaruma@gmail.com)",
    "Accept-Language": "ja-JP,ja;q=0.9",
}
if COOKIE:
    HEADERS["Cookie"] = COOKIE
    print("[auth] Cookie設定済み (odds取得可)")
else:
    print("[auth] Cookie未設定 (出走表+予想のみ)")

SLEEP = float(os.environ.get("REQUEST_SLEEP", "1.2"))


def _get(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        enc = r.apparent_encoding or "utf-8"
        if enc.lower() in ("windows-1252", "iso-8859-1", "latin-1"):
            enc = "cp932"
        r.encoding = enc
        return r.text
    except Exception as e:
        print(f"  GET failed {url[:50]}: {e}")
        return ""


def _to_int(v):
    try:
        return int(str(v).split(".")[0])
    except Exception:
        return 0


def _flt(v):
    try:
        return float(v)
    except Exception:
        return None


def _style(code):
    if not code:
        return None
    s = str(code).strip()
    for k, v in {"逃": "逃", "捲": "捲", "差": "差", "マ": "マーク", "追": "マーク"}.items():
        if k in s:
            return v
    return s[:4]


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
        # Players first (FK)
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
             "style": _style(e.get("style_code")), "scratched": 0}
            for e in enriched
        ])
        all_race_ids.append(rid)

print(f"出走表: {len(all_race_ids)} レース")

# ── 3. オッズ取得 ─────────────────────────────────────────────────────────────
live_odds: dict[str, dict] = {}
if COOKIE:
    print("オッズ取得中...")
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

# ── 4. 予想生成 ───────────────────────────────────────────────────────────────
bundle = load_latest_model(cfg.paths.models)
if not bundle:
    print("モデルなし → python -m keirin retrain を先に実行してください")
    sys.exit(1)

model, feature_cols, calibrator = bundle
ym = today.strftime("%Y-%m")
pnl_data = month_pnl(eng, ym)
mtd_pnl = pnl_data["pnl"]

recommended: list[dict] = []
skipped: list[dict] = []

for rid in all_race_ids:
    vc = int(rid[8:10])
    rno = int(rid[10:12])
    venue_name = VENUE_TABLE.get(f"{vc:02d}", ("?", 400))[0]
    odds = live_odds.get(rid)
    df = build_race_features(eng, rid, ref_date=today.isoformat(), latest_odds=odds)
    if df.empty:
        continue
    p_top3 = predict_race(df, model, feature_cols, calibrator)
    if not p_top3:
        continue
    combo_probs = expand_trifecta(p_top3, method=bcfg.combo_method)

    with eng.begin() as conn:
        row = conn.execute(text("SELECT grade FROM races WHERE race_id=:r"), {"r": rid}).fetchone()
    grade = row[0] if row else None

    if odds:
        candidates = build_candidates(combo_probs, odds)
        result = select_picks(candidates, bcfg)
        plan = plan_stakes(result.picks, bcfg, month_to_date_pnl_yen=mtd_pnl)
        if result.skipped:
            skipped.append({"venue_name": venue_name, "race_no": rno, "max_ev": result.max_ev})
        else:
            picks_out = [{
                "cars": [int(x) for x in p.pick.combo.split("-")],
                "odds": p.pick.odds, "prob": p.pick.prob, "ev": p.pick.ev,
                "stake_yen": p.stake_yen if not plan.emergency_stop_triggered else 0,
                "hit": None,
            } for p in plan.assignments]
            best = max(p_top3, key=p_top3.get)
            recommended.append({
                "venue_name": venue_name, "race_no": rno, "grade": grade,
                "race_class": None, "post_time": None,
                "distance_m": VENUE_TABLE.get(f"{vc:02d}", ("?", 400))[1],
                "weather": None, "lines": [],
                "axis": {"car_no": best, "name": "?", "note": f"p={p_top3[best]:.1%}"},
                "picks": picks_out, "max_ev": result.max_ev, "total_stake": plan.total_yen,
            })
    else:
        top3 = sorted(combo_probs.items(), key=lambda x: -x[1])[:3]
        skipped.append({"venue_name": venue_name, "race_no": rno, "max_ev": None, "top_combos": top3})

# ── 5. ダッシュボード出力 → docs/ ─────────────────────────────────────────────
daily_invest = sum(sum(p["stake_yen"] for p in r["picks"]) for r in recommended)
payload = {
    "date_label": today.isoformat(),
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M JST"),
    "mtd_pnl": mtd_pnl, "mtd_roi": pnl_data.get("roi"),
    "mtd_hit_rate": pnl_data.get("hit_rate"),
    "daily_invest": daily_invest,
    "daily_remaining": HARD_DAILY_BUDGET_YEN - daily_invest,
    "daily_cap": HARD_DAILY_BUDGET_YEN,
    "total_races": len(all_race_ids),
    "ev_threshold": bcfg.ev_threshold,
    "emergency_stop": mtd_pnl <= HARD_EMERGENCY_STOP_YEN,
    "recommended": recommended,
    "skipped": skipped,
}

docs = Path("docs")
docs.mkdir(exist_ok=True)
out_html = docs / f"{today.isoformat()}.html"
write_dashboard(out_html, payload)
write_markdown(
    cfg.paths.reports / "predictions" / f"{today.isoformat()}.md",
    payload
)

# index.html = 最新予想へリダイレクト
(docs / "index.html").write_text(
    f'<!DOCTYPE html><meta charset="utf-8">'
    f'<meta http-equiv="refresh" content="0;url={today.isoformat()}.html">'
    f'<title>競輪予想 {today}</title>',
    encoding="utf-8",
)

print(f"\n完了: 推奨={len(recommended)}R  見送り={len(skipped)}R  投資=¥{daily_invest:,}")
print(f"HTML: {out_html}")
