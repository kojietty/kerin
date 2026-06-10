"""
GitHub Actions 夜ジョブ (21:00 JST):
  1. 今日のレース結果取得 + DB保存
  2. player_race_log 更新
  3. 月次 ROI サマリーをコンソール出力
"""
from __future__ import annotations
import sys, io, os, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from datetime import date
from sqlalchemy import text

from keirin.config import load_app_config
from keirin.db.engine import get_engine, init_db
from keirin.db.repository import (
    sync_player_race_log, month_pnl, settle_prediction_log,
    insert_results, insert_payouts,
)
from keirin.data.import_cache import VENUE_TABLE
from keirin.scraper.session import fetch_text
from keirin.scraper.netkeirin import parse_full_result
from keirin.reporting.analyze import append_results_to_daily_html

cfg = load_app_config()
eng = get_engine(cfg.paths.db)
init_db(eng)
today = date.today()
TODAY_STR = today.strftime("%Y%m%d")

HEADERS = {
    "User-Agent": "KeirinResearchBot/0.1 (personal; pagudaruma@gmail.com)",
    "Accept-Language": "ja-JP,ja;q=0.9",
}
SLEEP = float(os.environ.get("REQUEST_SLEEP", "1.2"))


def _get(url: str) -> str:
    return fetch_text(url, headers=HEADERS, timeout=15)


def save_result(rid: str, data: dict) -> None:
    insert_results(eng, rid, data.get("results", []))
    if data.get("trifecta") and data.get("payout"):
        insert_payouts(eng, rid, [{
            "bet_type": "trifecta", "combo": data["trifecta"],
            "payout_yen": data["payout"], "popularity": None,
        }])


# 今日 DB に入っている race_ids を取得
with eng.begin() as conn:
    rows = conn.execute(text(
        "SELECT race_id FROM races WHERE date=:d AND race_id NOT IN"
        " (SELECT DISTINCT race_id FROM results) ORDER BY race_id"
    ), {"d": today.isoformat()}).fetchall()
race_ids = [r[0] for r in rows]

print(f"結果取得対象: {len(race_ids)} レース")
saved = 0
n_full = 0
for rid in race_ids:
    url = f"https://keirin.netkeiba.com/race/result/?race_id={rid}"
    html = _get(url)
    time.sleep(SLEEP)
    data = parse_full_result(html) if html else None
    if data:
        save_result(rid, data)
        saved += 1
        if len(data.get("results", [])) >= 5:
            n_full += 1
        vc = int(rid[8:10])
        rno = int(rid[10:12])
        name = VENUE_TABLE.get(f"{vc:02d}", ("?", 400))[0]
        payout_str = f"¥{data['payout']:,}" if data.get("payout") else "?"
        print(f"  {name} {rno}R → {data.get('trifecta','?')} {payout_str}"
              f" ({len(data.get('results', []))}着分)")

print(f"全着順取得: {n_full}/{saved} レース (5着以上の行が取れた数)")

print(f"\n保存: {saved}/{len(race_ids)} レース")

# prediction_log 決着処理 + HTML 更新
settled_preds = 0
for rid in race_ids:
    settled_preds += settle_prediction_log(eng, rid)
print(f"prediction_log 決着: {settled_preds} 件")

html_path = Path("docs") / f"{today.isoformat()}.html"
if append_results_to_daily_html(html_path, eng, today.isoformat()):
    print(f"HTML更新: {html_path} (的中結果追加)")
else:
    print(f"HTML更新スキップ (決着済み予測なし、またはファイル未存在)")

# player_race_log 更新
n = sync_player_race_log(eng)
print(f"sync_player_race_log: +{n} 行")

# 月次サマリー
ym = today.strftime("%Y-%m")
s = month_pnl(eng, ym)
roi_s = f"{s['roi']:.0%}" if s.get("roi") else "—"
print(f"\n{ym} 月次: ¥{s['pnl']:+,d}  ROI={roi_s}"
      f"  的中={s['hits']}/{s['bet_count']}")
