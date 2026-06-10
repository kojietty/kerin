"""
アーム比較バックテスト (週次 walk-forward)

5アームを同一フォールドで比較し、変更の効果を客観評価する:
  baseline  : 旧特徴量セットの classifier (脚質構成・sim 列なし)
  new_feats : + 脚質構成/B回数/bank_angle 特徴量
  sim_only  : 展開シミュレーション p_sim_top3 単体で順位付け (MLなし)
  fused     : new_feats + sim 特徴量注入 (本命構成)
  blend     : w·p_sim_top3 + (1−w)·p_ml (対照アーム, w は train 末尾で選択)

フォールド: 週単位 walk-forward (train = フォールド開始までの全期間, val = 次の1週)。
レース単位でグループ化されているためリークなし。

subset:
  all       : 全レース (sim なしレースは NaN フォールバック)
  line_only : ライン付きレースのみ (sim 系アームの主戦場)

指標: rank1的中率 / top3的中率(set) / NDCG@1,3 / top3 logloss / Brier / 車番TVD /
      仮想三連単ROI (予測1-2-3着に100円)

合格条件: fused が all subset で baseline を下回らないこと。

使い方 (重いので GitHub Actions backtest.yml 推奨):
  python scripts/backtest_compare.py --from 2026-03-01 --to 2026-06-08
  python scripts/backtest_compare.py --subset line_only --arms baseline,fused
"""
from __future__ import annotations
import sys, io, os, json, argparse, time
from collections import Counter
from datetime import date
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from keirin.config import load_app_config
from keirin.db.engine import get_engine, init_db
from keirin.backtest.metrics import (
    rank1_hit_rate, top3_set_hit_rate, tvd_car_bias, virtual_trifecta_roi,
)
from keirin.features.builder import build_training_frame
from keirin.models.calibrate import calibrate_isotonic
from keirin.models.train import TRAIN_FEATURES, _ndcg_at_k, _finish_to_relevance, train_model

# アームごとの特徴量セット
_STYLE_COLS = [
    "style_nige", "style_ryo", "style_oi",
    "n_nige_in_race", "nige_ratio", "is_lone_nige",
    "b_count_rank", "b_count_share",
    "nige_makuri_rate", "sashi_mark_rate",
    "style_x_short_bank", "style_x_long_bank",
]
_SIM_COLS = ["p_sim_win", "p_sim_top3", "p_front", "sim_used"]
_NEW_COLS = set(_STYLE_COLS) | set(_SIM_COLS) | {"bank_angle"}

BASELINE_FEATURES = [c for c in TRAIN_FEATURES if c not in _NEW_COLS]
NEW_FEATS_FEATURES = [c for c in TRAIN_FEATURES if c not in set(_SIM_COLS)]
FUSED_FEATURES = list(TRAIN_FEATURES)

ALL_ARMS = ["baseline", "new_feats", "sim_only", "fused", "blend"]


def _predict_order(probs: dict[int, float]) -> list[int]:
    """確率辞書 → 予測着順 (降順)。同値は確率の次に車番で安定化。"""
    return [c for c, _ in sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))]


def _ml_arm_eval(train_df, val_df, features):
    """classifier を学習し val の per-race 確率 dict を返す。"""
    model, feature_cols = train_model(train_df, val_df=val_df, features=features)
    # train 末尾 15% で isotonic 較正 (本番 retrain_and_save と同じ構成)
    split = int(len(train_df) * 0.85)
    cal_part = train_df.iloc[split:]
    X_cal = cal_part.reindex(columns=feature_cols, fill_value=np.nan)[feature_cols].values.astype(float)
    raw_cal = model.predict_proba(X_cal)[:, 1]
    try:
        calib = calibrate_isotonic(raw_cal, cal_part["label_top3"].values.astype(int))
    except Exception:
        calib = None

    X_val = val_df.reindex(columns=feature_cols, fill_value=np.nan)[feature_cols].values.astype(float)
    raw = model.predict_proba(X_val)[:, 1]
    probs = calib.predict(raw) if calib is not None else raw
    return pd.Series(probs, index=val_df.index)


def _eval_fold_arm(arm, train_df, val_df, blend_w_grid):
    """1フォールド×1アーム → race_id ごとの p_top3 Series (val index)。"""
    if arm == "baseline":
        return _ml_arm_eval(train_df, val_df, BASELINE_FEATURES), None
    if arm == "new_feats":
        return _ml_arm_eval(train_df, val_df, NEW_FEATS_FEATURES), None
    if arm == "fused":
        return _ml_arm_eval(train_df, val_df, FUSED_FEATURES), None
    if arm == "sim_only":
        return pd.to_numeric(val_df["p_sim_top3"], errors="coerce"), None
    if arm == "blend":
        p_ml = _ml_arm_eval(train_df, val_df, NEW_FEATS_FEATURES)
        p_sim = pd.to_numeric(val_df["p_sim_top3"], errors="coerce")
        # w 選択: train 末尾1週を擬似 val にしてグリッド
        dates = pd.to_datetime(train_df["_date"])
        cutoff = dates.max() - pd.Timedelta(days=7)
        inner_train = train_df[dates <= cutoff]
        inner_val = train_df[dates > cutoff]
        best_w = 0.0
        if len(inner_train) > 500 and len(inner_val) > 100 and inner_val["p_sim_top3"].notna().any():
            p_ml_i = _ml_arm_eval(inner_train, inner_val, NEW_FEATS_FEATURES)
            p_sim_i = pd.to_numeric(inner_val["p_sim_top3"], errors="coerce")
            y_i = inner_val["label_top3"].values.astype(int)
            best_ll = np.inf
            for w in blend_w_grid:
                p = (w * p_sim_i.fillna(p_ml_i) + (1 - w) * p_ml_i).clip(1e-6, 1 - 1e-6)
                ll = -np.mean(y_i * np.log(p) + (1 - y_i) * np.log(1 - p))
                if ll < best_ll:
                    best_ll, best_w = ll, w
        blended = best_w * p_sim.fillna(p_ml) + (1 - best_w) * p_ml
        return blended, best_w
    raise ValueError(arm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_d", default=None)
    ap.add_argument("--to", dest="to_d", default=None)
    ap.add_argument("--subset", choices=["all", "line_only"], default="all")
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--min-train-races", type=int, default=300)
    ap.add_argument("--out", default=None, help="結果JSONの出力先")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ALL_ARMS]

    cfg = load_app_config()
    eng = get_engine(cfg.paths.db)
    init_db(eng)

    from_d = args.from_d or "2026-02-14"
    to_d = args.to_d or date.today().isoformat()

    print(f"学習フレーム構築中 {from_d} → {to_d} ...")
    t0 = time.time()
    df = build_training_frame(eng, from_date=from_d, to_date=to_d)
    print(f"  {len(df)} 行 / {df['race_id'].nunique()} レース ({time.time()-t0:.0f}s)")
    if df.empty:
        print("データなし"); sys.exit(1)

    df = df.dropna(subset=["label_top3"]).copy()
    df["_date"] = pd.to_datetime(df["race_id"].str[:8], format="%Y%m%d")

    if args.subset == "line_only":
        lined = df.groupby("race_id")["line_id"].transform(lambda s: s.notna().sum() >= 2)
        df = df[lined]
        print(f"line_only subset: {df['race_id'].nunique()} レース")

    if df.empty:
        print("subset にデータなし"); sys.exit(1)

    # 週次フォールド
    df["_week"] = df["_date"].dt.to_period("W")
    weeks = sorted(df["_week"].unique())
    blend_w_grid = [round(w, 1) for w in np.arange(0.0, 1.01, 0.1)]

    fold_results: list[dict] = []
    agg: dict[str, dict] = {a: {
        "preds": [], "actuals": [], "pred_1st": Counter(), "actual_1st": Counter(),
        "ndcg1": [], "ndcg3": [], "logloss": [], "brier": [],
        "picks": {}, "blend_w": [],
    } for a in arms}

    for wi, week in enumerate(weeks):
        train_df = df[df["_week"] < week]
        val_df = df[df["_week"] == week]
        if train_df["race_id"].nunique() < args.min_train_races or val_df.empty:
            continue
        print(f"fold {week}: train {train_df['race_id'].nunique()}R / val {val_df['race_id'].nunique()}R")

        for arm in arms:
            try:
                probs, blend_w = _eval_fold_arm(arm, train_df, val_df, blend_w_grid)
            except Exception as e:
                print(f"  {arm}: 失敗 {e}")
                continue
            if blend_w is not None:
                agg[arm]["blend_w"].append(blend_w)

            vdf = val_df.copy()
            vdf["_p"] = probs
            y = vdf["label_top3"].values.astype(int)
            p = pd.to_numeric(vdf["_p"], errors="coerce").values
            valid = ~np.isnan(p)
            if valid.sum() > 10:
                pc = np.clip(p[valid], 1e-6, 1 - 1e-6)
                agg[arm]["logloss"].append(float(
                    -np.mean(y[valid] * np.log(pc) + (1 - y[valid]) * np.log(1 - pc))))
                agg[arm]["brier"].append(float(np.mean((pc - y[valid]) ** 2)))

            for rid, race in vdf.groupby("race_id"):
                race = race.dropna(subset=["_p"])
                if len(race) < 4:
                    continue
                pr = dict(zip(race["car_no"].astype(int), race["_p"].astype(float)))
                order = _predict_order(pr)
                actual = (race.dropna(subset=["finish"])
                          .sort_values("finish"))
                actual_order = actual["car_no"].astype(int).tolist()
                if len(actual_order) < 3:
                    continue
                agg[arm]["preds"].append(order)
                agg[arm]["actuals"].append(actual_order)
                agg[arm]["pred_1st"][order[0]] += 1
                agg[arm]["actual_1st"][actual_order[0]] += 1
                agg[arm]["picks"][rid] = ["-".join(map(str, order[:3]))]
                y_rel = race["finish"].apply(_finish_to_relevance).values.astype(float)
                scores = race["_p"].values.astype(float)
                agg[arm]["ndcg1"].append(_ndcg_at_k(y_rel, scores, 1))
                agg[arm]["ndcg3"].append(_ndcg_at_k(y_rel, scores, 3))

    # ── 集計 ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"アーム比較 ({args.subset} subset, {from_d} → {to_d})")
    print(f"{'=' * 78}")
    header = (f"{'arm':<10} {'races':>6} {'rank1':>7} {'top3':>7} {'NDCG@1':>7}"
              f" {'NDCG@3':>7} {'logloss':>8} {'Brier':>7} {'TVD':>6} {'ROI':>7}")
    print(header)
    print("-" * len(header))

    summary: dict[str, dict] = {}
    for arm in arms:
        a = agg[arm]
        n = len(a["preds"])
        if not n:
            print(f"{arm:<10} {'—':>6}")
            continue
        roi = virtual_trifecta_roi(eng, a["picks"])
        s = {
            "races": n,
            "rank1_hit": round(rank1_hit_rate(a["preds"], a["actuals"]), 4),
            "top3_set_hit": round(top3_set_hit_rate(a["preds"], a["actuals"]), 4),
            "ndcg1": round(float(np.mean(a["ndcg1"])), 4),
            "ndcg3": round(float(np.mean(a["ndcg3"])), 4),
            "logloss": round(float(np.mean(a["logloss"])), 4) if a["logloss"] else None,
            "brier": round(float(np.mean(a["brier"])), 4) if a["brier"] else None,
            "tvd_car_bias": round(tvd_car_bias(a["pred_1st"], a["actual_1st"]), 4),
            "virtual_roi": round(roi["roi"], 4) if roi["roi"] is not None else None,
            "roi_hits": roi["hits"],
        }
        if a["blend_w"]:
            s["blend_w_mean"] = round(float(np.mean(a["blend_w"])), 2)
        summary[arm] = s
        print(f"{arm:<10} {n:>6} {s['rank1_hit']:>7.1%} {s['top3_set_hit']:>7.1%}"
              f" {s['ndcg1']:>7.4f} {s['ndcg3']:>7.4f}"
              f" {str(s['logloss']):>8} {str(s['brier']):>7}"
              f" {s['tvd_car_bias']:>6.3f}"
              f" {('%.1f%%' % (100 * s['virtual_roi'])) if s['virtual_roi'] is not None else '—':>7}")

    # 合格判定
    if "baseline" in summary and "fused" in summary:
        ok = summary["fused"]["top3_set_hit"] >= summary["baseline"]["top3_set_hit"] - 1e-9
        print(f"\n合格条件 (fused >= baseline, top3): {'✅ PASS' if ok else '❌ FAIL'}")

    out_path = Path(args.out) if args.out else Path("reports") / (
        f"backtest_{args.subset}_{date.today().strftime('%Y%m%d')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "from": from_d, "to": to_d, "subset": args.subset,
        "arms": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"結果保存: {out_path}")


if __name__ == "__main__":
    main()
