"""バックテスト指標.

diag_car_bias.py / repository.accuracy_metrics と同じ定義で、
アーム比較 (backtest_compare.py) から再利用できる関数群。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine


def rank1_hit_rate(preds: list[list[int]], actuals: list[list[int]]) -> float:
    """preds/actuals: レースごとの着順車番リスト (先頭=1着)。1着完全一致率。"""
    pairs = [(p, a) for p, a in zip(preds, actuals) if p and a]
    if not pairs:
        return float("nan")
    return sum(1 for p, a in pairs if p[0] == a[0]) / len(pairs)


def top3_set_hit_rate(preds: list[list[int]], actuals: list[list[int]]) -> float:
    """予測上位3車 = 実際の上位3車 (順不同) の率。"""
    pairs = [(p, a) for p, a in zip(preds, actuals) if len(p) >= 3 and len(a) >= 3]
    if not pairs:
        return float("nan")
    return sum(1 for p, a in pairs if set(p[:3]) == set(a[:3])) / len(pairs)


def tvd_car_bias(pred_1st: Counter, actual_1st: Counter) -> float:
    """予測1着車番分布 vs 実1着車番分布の総変動距離 (0=完全一致)。

    diag_car_bias.py と同定義。車番依存バイアス (思想②) の監視指標。
    """
    n_pred = sum(pred_1st.values()) or 1
    n_act = sum(actual_1st.values()) or 1
    return 0.5 * sum(
        abs(pred_1st.get(c, 0) / n_pred - actual_1st.get(c, 0) / n_act)
        for c in range(1, 10)
    )


def virtual_trifecta_roi(
    engine: Engine,
    picks_by_race: dict[str, list[str]],
    *,
    stake_yen: int = 100,
) -> dict[str, Any]:
    """payouts テーブルの当たり三連単払戻で仮想ROIを計算する。

    picks_by_race: race_id → 賭けるコンボ ("a-b-c") のリスト。
    各コンボに stake_yen ずつ賭けたと仮定。
    """
    race_ids = list(picks_by_race.keys())
    if not race_ids:
        return {"stake": 0, "payout": 0, "roi": None, "hits": 0, "bets": 0}

    winning: dict[str, tuple[str, int]] = {}
    with engine.begin() as conn:
        # SQLite の IN リスト上限を避けるためチャンク
        for i in range(0, len(race_ids), 500):
            chunk = race_ids[i:i + 500]
            ph = ",".join(f":r{j}" for j in range(len(chunk)))
            rows = conn.execute(
                text(
                    f"SELECT race_id, combo, payout_yen FROM payouts"
                    f" WHERE bet_type = 'trifecta' AND race_id IN ({ph})"
                ),
                {f"r{j}": rid for j, rid in enumerate(chunk)},
            ).fetchall()
            for rid, combo, pay in rows:
                winning[rid] = (combo, int(pay or 0))

    stake = payout = hits = bets = 0
    for rid, picks in picks_by_race.items():
        win = winning.get(rid)
        for combo in picks:
            bets += 1
            stake += stake_yen
            if win and combo == win[0]:
                hits += 1
                payout += int(win[1] * stake_yen / 100)

    return {
        "stake": stake,
        "payout": payout,
        "roi": (payout / stake) if stake else None,
        "hits": hits,
        "bets": bets,
    }
