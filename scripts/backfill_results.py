"""
過去レースの全着順+決まり手バックフィル

既存の results は各レース上位3着のみ・kimarite 全NULL。本スクリプトは
結果ページを再取得し、全着順 (1-9着)・決まり手・上がりタイムで UPSERT する。
完了後 sync_player_race_log を再実行して選手履歴に反映する
(着外センチネル finish=9 は実際の着順で上書きされる)。

対象: 結果が3行以下 or 決まり手未取得の過去レース
      (結果ゼロの古いレースも含む)

使い方:
  python scripts/backfill_results.py --from 2026-02-01 --to 2026-06-08
  python scripts/backfill_results.py --limit 1000

クラウド環境は 403 でスクレイプ不可 → GitHub Actions か手元で実行すること。
"""
from __future__ import annotations
import sys, io, os, time, argparse
from datetime import date
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from sqlalchemy import text

from keirin.config import load_app_config
from keirin.db.engine import get_engine, init_db
from keirin.db.repository import insert_results, insert_payouts, sync_player_race_log
from keirin.scraper.session import fetch_text
from keirin.scraper.netkeirin import result_url, parse_full_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_d", default=None, help="YYYY-MM-DD")
    ap.add_argument("--to", dest="to_d", default=None, help="YYYY-MM-DD")
    ap.add_argument("--limit", type=int, default=2000, help="最大レース数")
    args = ap.parse_args()

    cfg = load_app_config()
    eng = get_engine(cfg.paths.db)
    init_db(eng)

    cookie = os.environ.get("NETKEIRIN_COOKIE", "")
    headers = {
        "User-Agent": "KeirinResearchBot/0.1 (personal; pagudaruma@gmail.com)",
        "Accept-Language": "ja-JP,ja;q=0.9",
    }
    if cookie:
        headers["Cookie"] = cookie
    sleep = float(os.environ.get("REQUEST_SLEEP", "1.5"))

    # 取得対象: 全着順が無い (結果行 < 5) か kimarite 未取得の過去レース
    where = [
        "r.date < :today",
        "((SELECT COUNT(*) FROM results res WHERE res.race_id = r.race_id) < 5"
        " OR NOT EXISTS (SELECT 1 FROM results res WHERE res.race_id = r.race_id"
        "                AND res.kimarite IS NOT NULL))",
    ]
    params: dict = {"today": date.today().isoformat()}
    if args.from_d:
        where.append("r.date >= :fd"); params["fd"] = args.from_d
    if args.to_d:
        where.append("r.date <= :td"); params["td"] = args.to_d
    sql = ("SELECT r.race_id FROM races r WHERE " + " AND ".join(where)
           + " ORDER BY r.date DESC LIMIT :lim")
    params["lim"] = args.limit
    with eng.begin() as conn:
        race_ids = [row[0] for row in conn.execute(text(sql), params).fetchall()]

    print(f"対象レース: {len(race_ids)} 件")
    updated = 0
    n_full = 0
    n_kim = 0

    for rid in race_ids:
        html = fetch_text(result_url(rid), headers=headers)
        time.sleep(sleep)
        if not html:
            continue
        data = parse_full_result(html)
        if not data or not data.get("results"):
            continue
        insert_results(eng, rid, data["results"])
        if data.get("trifecta") and data.get("payout"):
            insert_payouts(eng, rid, [{
                "bet_type": "trifecta", "combo": data["trifecta"],
                "payout_yen": data["payout"], "popularity": None,
            }])
        updated += 1
        if len(data["results"]) >= 5:
            n_full += 1
        if any(r.get("kimarite") for r in data["results"]):
            n_kim += 1
        if updated % 50 == 0:
            print(f"  ...{updated} レース更新済み (全着順 {n_full} / 決まり手 {n_kim})")

    print(f"\n更新: {updated} レース (全着順 {n_full} / 決まり手あり {n_kim})")

    n = sync_player_race_log(eng)
    print(f"sync_player_race_log: {n} 行 upsert")

    with eng.begin() as conn:
        kim_total = conn.execute(text(
            "SELECT COUNT(*) FROM results WHERE kimarite IS NOT NULL"
        )).scalar()
        full_races = conn.execute(text(
            "SELECT COUNT(*) FROM (SELECT race_id FROM results"
            " GROUP BY race_id HAVING COUNT(*) >= 5)"
        )).scalar()
    print(f"results: 決まり手あり {kim_total} 行 / 全着順レース {full_races}")
    print("→ 完了。決まり手×ライン位置の統計が使えるようになります。")


if __name__ == "__main__":
    main()
