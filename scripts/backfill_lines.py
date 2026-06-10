"""
過去レースのライン(並び)データ・バックフィル

旧パーサ破損により、ライン構成は1レースも正しく取れていなかった
(全レースで car_no=1 のみ line_id=1/line_pos=1 の偽データ)。
fix_db_quality.py で偽データをクリアした後、本スクリプトで出走表ページを
再取得し、修復済みパーサで line_id/line_pos を埋め直す。

同じページから取れる 逃/捲/差/マ回数・S/B回数・line_raw も同時に保存する
(そのレース時点の出走表の値なのでリークなし)。

re-scrape が部分的でも他の列(rating, player_id 等)を壊さないよう、
対象列のみを UPDATE する(upsert_entries は使わない)。

使い方 (要 NETKEIRIN_COOKIE / ネットワーク到達):
  python scripts/backfill_lines.py --from 2026-02-01 --to 2026-04-30
  python scripts/backfill_lines.py --limit 500

クラウド環境は 403 でスクレイプ不可 → GitHub Actions か手元で実行すること。
"""
from __future__ import annotations
import sys, io, os, time, argparse
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from sqlalchemy import text

from keirin.config import load_app_config
from keirin.db.engine import get_engine, init_db
from keirin.scraper.session import fetch_text
from keirin.scraper.netkeirin import entry_url, parse_entries, parse_line_assignments


def _to_int(v):
    try:
        return int(str(v).split(".")[0])
    except Exception:
        return None


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

    # カバー率: ライン付きレース = 2台以上に line_id があるレース
    def _coverage(conn):
        tot = conn.execute(text("SELECT COUNT(DISTINCT race_id) FROM entries")).scalar()
        lined = conn.execute(text(
            "SELECT COUNT(*) FROM (SELECT race_id FROM entries WHERE line_id IS NOT NULL"
            " GROUP BY race_id HAVING COUNT(*) >= 2)"
        )).scalar()
        return lined, tot

    with eng.begin() as conn:
        l0, tot = _coverage(conn)
    print(f"ライン付きレース (前): {l0}/{tot} ({100*l0/max(tot,1):.1f}%)")

    # ライン未取得 (割当2台未満) の過去レースを抽出
    where = ["(SELECT COUNT(*) FROM entries e WHERE e.race_id=r.race_id"
             " AND e.line_id IS NOT NULL) < 2"]
    params: dict = {}
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
    updated_races = 0
    updated_rows = 0
    lined_dist: Counter = Counter()   # レースあたり割当車数の分布 (診断用)

    for rid in race_ids:
        html = fetch_text(entry_url(rid), headers=headers)
        time.sleep(sleep)
        if not html:
            continue
        raw = parse_entries(html)
        if len(raw) < 5:
            continue
        enriched = parse_line_assignments(raw)

        n_lined = sum(1 for e in enriched if e.get("line_id") is not None)
        lined_dist[n_lined] += 1

        rows = []
        for e in enriched:
            cn = _to_int(e.get("car_no"))
            if cn is None:
                continue
            rows.append({
                "rid": rid, "cn": cn,
                "lid": e.get("line_id"), "lp": e.get("line_pos"),
                "lraw": e.get("line_raw") or None,
                "nige": _to_int(e.get("nige")), "maku": _to_int(e.get("makuri")),
                "sashi": _to_int(e.get("sashi")), "mark": _to_int(e.get("mark")),
                "s_cnt": _to_int(e.get("sprints")), "b_cnt": _to_int(e.get("backs")),
            })
        if not rows:
            continue
        with eng.begin() as conn:
            for lr in rows:
                conn.execute(text(
                    "UPDATE entries SET"
                    "  line_id=:lid, line_pos=:lp, line_raw=:lraw,"
                    "  nige_cnt=COALESCE(:nige, nige_cnt),"
                    "  makuri_cnt=COALESCE(:maku, makuri_cnt),"
                    "  sashi_cnt=COALESCE(:sashi, sashi_cnt),"
                    "  mark_cnt=COALESCE(:mark, mark_cnt),"
                    "  s_count=COALESCE(:s_cnt, s_count),"
                    "  b_count=COALESCE(:b_cnt, b_count)"
                    " WHERE race_id=:rid AND car_no=:cn"
                ), lr)
        updated_races += 1
        updated_rows += len(rows)
        if updated_races % 25 == 0:
            print(f"  ...{updated_races} レース更新済み")

    # 診断: レースあたり割当車数の分布 (1台/レースなら旧バグの再発)
    print("\nレースあたりライン割当車数の分布:")
    for k in sorted(lined_dist):
        print(f"  {k}台: {lined_dist[k]} レース")
    if lined_dist and max(lined_dist) == 0:
        sample = None
        with eng.begin() as conn:
            sample = conn.execute(text(
                "SELECT line_raw FROM entries WHERE line_raw IS NOT NULL LIMIT 5"
            )).fetchall()
        print(f"⚠ 全レースで割当0台。line_raw サンプル: {sample}")

    with eng.begin() as conn:
        l1, tot = _coverage(conn)
    print(f"\n更新: {updated_races} レース / {updated_rows} 行")
    print(f"ライン付きレース (後): {l1}/{tot} ({100*l1/max(tot,1):.1f}%)")
    print("→ 完了。次に precompute_sim_features.py → 再学習でライン特徴量が有効化されます。")


if __name__ == "__main__":
    main()
