"""
展開シミュレーション特徴量の一括事前計算 → sim_features キャッシュ

ライン付き (line_id 2台以上) の全レースについて simulate_race を実行し、
p_sim_win / p_sim_top3 / p_front を sim_features テーブルに保存する。
build_race_features は学習時このキャッシュを参照する (無ければ NaN)。

ネットワーク不要・ローカル/Actions どちらでも実行可。再学習の前段で実行する。

使い方:
  python scripts/precompute_sim_features.py
  python scripts/precompute_sim_features.py --from 2026-04-01 --force
"""
from __future__ import annotations
import sys, io, argparse, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

import pandas as pd
from sqlalchemy import text

from keirin.config import load_app_config
from keirin.db.engine import get_engine, init_db
from keirin.db.repository import upsert_sim_features
from keirin.simulation import SIM_VERSION, simulate_race


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_d", default=None, help="YYYY-MM-DD")
    ap.add_argument("--to", dest="to_d", default=None, help="YYYY-MM-DD")
    ap.add_argument("--force", action="store_true",
                    help="計算済み (同一 sim_version) のレースも再計算")
    args = ap.parse_args()

    cfg = load_app_config()
    eng = get_engine(cfg.paths.db)
    init_db(eng)

    where = [
        "(SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id"
        " AND e.line_id IS NOT NULL) >= 2"
    ]
    params: dict = {}
    if args.from_d:
        where.append("r.date >= :fd"); params["fd"] = args.from_d
    if args.to_d:
        where.append("r.date <= :td"); params["td"] = args.to_d
    if not args.force:
        where.append(
            "NOT EXISTS (SELECT 1 FROM sim_features sf WHERE sf.race_id = r.race_id"
            " AND sf.sim_version = :ver)"
        )
        params["ver"] = SIM_VERSION

    sql = "SELECT r.race_id FROM races r WHERE " + " AND ".join(where) + " ORDER BY r.date"
    with eng.begin() as conn:
        race_ids = [row[0] for row in conn.execute(text(sql), params).fetchall()]

    print(f"対象レース (ライン付き): {len(race_ids)} 件  sim_version={SIM_VERSION}")
    t0 = time.time()
    done = 0
    skipped = 0

    for rid in race_ids:
        with eng.begin() as conn:
            df = pd.read_sql(text(
                """
                SELECT e.car_no, e.line_id, e.line_pos, e.rating,
                       e.b_count, e.nige_cnt, e.makuri_cnt, e.sashi_cnt, e.mark_cnt,
                       v.bank_length
                FROM entries e
                JOIN races r ON e.race_id = r.race_id
                LEFT JOIN venues v ON r.venue_id = v.venue_id
                WHERE e.race_id = :rid AND COALESCE(e.scratched, 0) = 0
                ORDER BY e.car_no
                """
            ), conn, params={"rid": rid})

        sim = simulate_race(df)
        if sim is None:
            skipped += 1
            continue
        upsert_sim_features(eng, rid, [
            {"car_no": c, "p_sim_win": sim.p_win.get(c),
             "p_sim_top3": sim.p_top3.get(c), "p_front": sim.p_front.get(c),
             "sim_version": SIM_VERSION}
            for c in sim.p_win
        ])
        done += 1
        if done % 200 == 0:
            print(f"  ...{done} レース ({time.time()-t0:.0f}s)")

    print(f"\n完了: {done} レース計算 / {skipped} スキップ ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
