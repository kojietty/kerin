"""
GitHub Actions 夜ジョブ (21:00 JST):
  1. 今日のレース結果取得 + DB保存
  2. player_race_log 更新
  3. 月次 ROI サマリーをコンソール出力
"""
from __future__ import annotations
import sys, io, os, re, time, requests
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from datetime import date
from bs4 import BeautifulSoup
from sqlalchemy import text

from keirin.config import load_app_config
from keirin.db.engine import get_engine, init_db
from keirin.db.repository import sync_player_race_log, month_pnl, settle_prediction_log
from keirin.data.import_cache import VENUE_TABLE
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
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        enc = r.apparent_encoding or "utf-8"
        if enc.lower() in ("windows-1252", "iso-8859-1", "latin-1"):
            enc = "cp932"
        r.encoding = enc
        return r.text
    except Exception as e:
        print(f"  GET failed: {e}")
        return ""


def parse_result(html: str, race_id: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")
    body = soup.get_text("\n")
    lines = [l.strip() for l in body.split("\n") if l.strip()]
    trifecta = payout = None
    for i, line in enumerate(lines):
        if "３連単" in line or "3連単" in line:
            chunk = " ".join(lines[i:i+8])
            m_combo = re.search(r"([1-9])[>＞－\-]([1-9])[>＞－\-]([1-9])", chunk)
            m_pay = re.search(r"([\d,]+)円", chunk)
            if m_combo:
                trifecta = f"{m_combo.group(1)}-{m_combo.group(2)}-{m_combo.group(3)}"
            if m_pay:
                payout = int(m_pay.group(1).replace(",", ""))
            if trifecta:
                break
    results = []
    for i, line in enumerate(lines):
        m = re.match(r"^([123])着$", line)
        if m:
            for j in range(i+1, min(i+6, len(lines))):
                if re.match(r"^[1-9]$", lines[j]):
                    results.append({"finish": int(m.group(1)), "car_no": int(lines[j])})
                    break
    if not trifecta and not results:
        return None
    if trifecta and len(results) < 3:
        cars = [int(c) for c in trifecta.split("-")]
        results = [{"finish": i+1, "car_no": c} for i, c in enumerate(cars)]
    return {"race_id": race_id, "results": results[:3], "trifecta": trifecta, "payout": payout}


def save_result(data: dict) -> None:
    rid = data["race_id"]
    with eng.begin() as conn:
        for r in data.get("results", []):
            conn.execute(text(
                "INSERT OR IGNORE INTO results (race_id,car_no,finish,kimarite,time_sec)"
                " VALUES (:rid,:cn,:finish,NULL,NULL)"
            ), {"rid": rid, "cn": r["car_no"], "finish": r["finish"]})
        if data.get("trifecta") and data.get("payout"):
            conn.execute(text(
                "INSERT OR IGNORE INTO payouts (race_id,bet_type,combo,payout_yen,popularity)"
                " VALUES (:rid,'trifecta',:combo,:payout,NULL)"
            ), {"rid": rid, "combo": data["trifecta"], "payout": data["payout"]})


# 今日 DB に入っている race_ids を取得
with eng.begin() as conn:
    rows = conn.execute(text(
        "SELECT race_id FROM races WHERE date=:d AND race_id NOT IN"
        " (SELECT DISTINCT race_id FROM results) ORDER BY race_id"
    ), {"d": today.isoformat()}).fetchall()
race_ids = [r[0] for r in rows]

print(f"結果取得対象: {len(race_ids)} レース")
saved = 0
for rid in race_ids:
    url = f"https://keirin.netkeiba.com/race/result/?race_id={rid}"
    html = _get(url)
    time.sleep(SLEEP)
    data = parse_result(html, rid) if html else None
    if data:
        save_result(data)
        saved += 1
        vc = int(rid[8:10])
        rno = int(rid[10:12])
        name = VENUE_TABLE.get(f"{vc:02d}", ("?", 400))[0]
        payout_str = f"¥{data['payout']:,}" if data.get("payout") else "?"
        print(f"  {name} {rno}R → {data.get('trifecta','?')} {payout_str}")

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
