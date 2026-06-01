"""
過去レースのライン(並び)データ・バックフィル

学習データのライン情報カバー率が低い(過去の一括インポートに並びが含まれていない)ため、
ライン特徴量がモデルでほぼ使われていない。本スクリプトは line_id/line_pos が未設定の
過去レースの出走表ページを再取得し、並びだけを安全に in-place 更新する。

re-scrape が部分的でも他の列(rating, player_id 等)を壊さないよう、line_id/line_pos のみを
UPDATE する(upsert_entries は全列上書きのため使わない)。

使い方 (要 NETKEIRIN_COOKIE / ネットワーク到達):
  python scripts/backfill_lines.py --from 2026-02-01 --to 2026-04-30
  python scripts/backfill_lines.py --limit 500

クラウド環境は 403 でスクレイプ不可 → GitHub Actions か手元で実行すること。
"""
from __future__ import annotations
import sys, io, os, time, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

import requests
from sqlalchemy import text

from keirin.config import load_app_config
from keirin.db.engine import get_engine, init_db
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

    def _get(url: str) -> str:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            enc = r.apparent_encoding or "utf-8"
            if enc.lower() in ("windows-1252", "iso-8859-1", "latin-1"):
                enc = "cp932"
            r.encoding = enc
            return r.text
        except Exception as e:
            print(f"  GET失敗 {url[:50]}: {e}")
            return ""

    # カバー率 (前)
    with eng.begin() as conn:
        tot = conn.execute(text("SELECT COUNT(*) FROM entries")).scalar()
        wl0 = conn.execute(text("SELECT COUNT(*) FROM entries WHERE line_id IS NOT NULL")).scalar()
    print(f"ライン カバー率 (前): {wl0}/{tot} ({100*wl0/max(tot,1):.1f}%)")

    # line_id 未設定の出走を含む過去レースを抽出
    where = ["EXISTS (SELECT 1 FROM entries e WHERE e.race_id=r.race_id AND e.line_id IS NULL)"]
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

    for rid in race_ids:
        html = _get(entry_url(rid))
        time.sleep(sleep)
        if not html:
            continue
        raw = parse_entries(html)
        if len(raw) < 5:
            continue
        enriched = parse_line_assignments(raw)
        # line_id を取得できた車だけを対象に line_id/line_pos のみ更新
        line_rows = [
            {"rid": rid, "cn": _to_int(e.get("car_no")),
             "lid": e.get("line_id"), "lp": e.get("line_pos")}
            for e in enriched
            if e.get("line_id") is not None and _to_int(e.get("car_no")) is not None
        ]
        if not line_rows:
            continue
        with eng.begin() as conn:
            for lr in line_rows:
                conn.execute(text(
                    "UPDATE entries SET line_id=:lid, line_pos=:lp "
                    "WHERE race_id=:rid AND car_no=:cn"
                ), lr)
        updated_races += 1
        updated_rows += len(line_rows)
        if updated_races % 25 == 0:
            print(f"  ...{updated_races} レース更新済み")

    # カバー率 (後)
    with eng.begin() as conn:
        wl1 = conn.execute(text("SELECT COUNT(*) FROM entries WHERE line_id IS NOT NULL")).scalar()
    print(f"\n更新: {updated_races} レース / {updated_rows} 行")
    print(f"ライン カバー率 (後): {wl1}/{tot} ({100*wl1/max(tot,1):.1f}%)")
    print("→ 完了。次に再学習 (python -m keirin retrain) でライン特徴量が有効化されます。")


if __name__ == "__main__":
    main()
