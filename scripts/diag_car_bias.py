"""
診断: 予測1着の車番分布 vs 実際の1着車番分布

学習済みモデルで過去レースの着順予想を生成し、予測1着がどの車番に偏っているかを
集計する。実際の結果（results）があれば、実1着の車番分布も並べて表示する。

使い方:
  python scripts/diag_car_bias.py                  # 直近300レースをサンプリング
  python scripts/diag_car_bias.py --limit 500
  python scripts/diag_car_bias.py --from 2026-04-01 --to 2026-04-30

①同着判定バグ修正の前後、③ライン特徴量を入れた再学習の前後で実行し、
車番1偏重が改善したか客観的に確認するためのツール。
"""
from __future__ import annotations
import sys, io, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from collections import Counter

from sqlalchemy import text

from keirin.config import load_app_config
from keirin.db.engine import get_engine, init_db
from keirin.models.train import load_latest_model, load_latest_ranker
from keirin.models.predict import predict_race, rank_prediction
from keirin.features.builder import build_race_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_d", default=None, help="YYYY-MM-DD")
    ap.add_argument("--to", dest="to_d", default=None, help="YYYY-MM-DD")
    ap.add_argument("--limit", type=int, default=300, help="サンプリングするレース数")
    args = ap.parse_args()

    cfg = load_app_config()
    eng = get_engine(cfg.paths.db)
    init_db(eng)

    bundle = load_latest_model(cfg.paths.models)
    if not bundle:
        print("モデルなし → python -m keirin retrain を先に実行")
        sys.exit(1)
    model, feature_cols, calibrator = bundle
    ranker_bundle = load_latest_ranker(cfg.paths.models)
    ranker, ranker_features = ranker_bundle if ranker_bundle else (None, None)
    print(f"モデル: {'LambdaRank' if ranker else 'PL近似 (rankerなし)'}")

    # レース抽出 (結果がある = 過去レース)
    where = ["r.race_id IN (SELECT DISTINCT race_id FROM results)"]
    params: dict = {}
    if args.from_d:
        where.append("r.date >= :fd"); params["fd"] = args.from_d
    if args.to_d:
        where.append("r.date <= :td"); params["td"] = args.to_d
    sql = (
        "SELECT r.race_id, r.date FROM races r WHERE "
        + " AND ".join(where)
        + " ORDER BY r.date DESC, r.race_id DESC LIMIT :lim"
    )
    params["lim"] = args.limit
    with eng.begin() as conn:
        races = conn.execute(text(sql), params).fetchall()

    print(f"対象レース: {len(races)} 件")

    pred_1st = Counter()
    pred_top3 = Counter()
    actual_1st = Counter()
    n_pred = 0

    for race_id, _date in races:
        df = build_race_features(eng, race_id, ref_date=_date)
        if df.empty:
            continue
        p_top3 = predict_race(df, model, feature_cols, calibrator)
        if not p_top3:
            continue
        rank_scores = None
        if ranker is not None and ranker_features:
            import numpy as np
            X = df.reindex(columns=ranker_features, fill_value=np.nan).values.astype("float32")
            scores = ranker.predict(X)
            rank_scores = dict(zip(df["car_no"].tolist(), scores.tolist()))
        ranking = rank_prediction(p_top3, rank_scores=rank_scores)
        if len(ranking) < 3:
            continue
        n_pred += 1
        pred_1st[ranking[0]["car_no"]] += 1
        for r in ranking[:3]:
            pred_top3[r["car_no"]] += 1

        with eng.begin() as conn:
            row = conn.execute(text(
                "SELECT car_no FROM results WHERE race_id=:r AND finish=1"
            ), {"r": race_id}).fetchone()
        if row:
            actual_1st[row[0]] += 1

    if not n_pred:
        print("予測生成できたレースなし")
        return

    n_act = sum(actual_1st.values()) or 1
    print(f"\n予測生成: {n_pred} レース  /  実結果: {n_act} レース")
    print(f"\n{'車番':>4} | {'予測1着率':>9} | {'実1着率':>8} | {'予測3着内率':>10}")
    print("-" * 46)
    for car in range(1, 10):
        p1 = pred_1st.get(car, 0) / n_pred
        a1 = actual_1st.get(car, 0) / n_act
        t3 = pred_top3.get(car, 0) / n_pred
        flag = "  ←偏り?" if (p1 - a1) > 0.05 else ""
        print(f"{car:>4} | {p1:>8.1%} | {a1:>7.1%} | {t3:>9.1%}{flag}")

    # 偏りの要約指標
    car1_pred = pred_1st.get(1, 0) / n_pred
    car1_act = actual_1st.get(1, 0) / n_act
    print("-" * 46)
    print(f"車番1: 予測 {car1_pred:.1%} vs 実 {car1_act:.1%}  "
          f"(差 {car1_pred - car1_act:+.1%})")
    # 予測と実分布の総絶対誤差 (小さいほど良い)
    tvd = 0.5 * sum(
        abs(pred_1st.get(c, 0) / n_pred - actual_1st.get(c, 0) / n_act)
        for c in range(1, 10)
    )
    print(f"予測1着分布 vs 実分布の総変動距離: {tvd:.3f} (0=完全一致)")


if __name__ == "__main__":
    main()
