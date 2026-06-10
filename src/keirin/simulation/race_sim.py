"""展開シミュレーション本体 (3フェーズ離散イベントモデル).

Phase A: 主導権 (先行) 争い — どのラインが前を取るか
         P(F=L) = softmax_τ( front_power_L − δ_solo·1[単騎] )
Phase B: 捲り — 各非主導権ラインのリーダーが主導権ラインを越えられるか
         P(捲り成功) = σ( β0 + β1·(front_power_l − front_power_F)
                          + β2·1[bank≤350] + β3·1[bank≥500] )
Phase C: 直線 —
  C-1 ライン内追い抜き: 番手が先行を交わす / 三番手が前を交わす
         P = σ( γ0 + γ1·z(差+マ) + γ2·Δz(rating) (+ γ3 三番手) )
  C-2 隊列順位 → 最終着順の Plackett-Luce:
         strength_i = exp( −λ·暫定順位_i + μ·z(rating)_i )
         展開崩れ・単騎強豪の食い込みを表現し、決定論化を防ぐ。

v0 はシナリオ空間 (主導権の取り手 × 捲り成否 × ライン内追い抜き) を確率重み付きで
**厳密列挙**する (モンテカルロ不要・決定的・ノイズなし)。シナリオ数は最大でも
~2,000 程度で、微小確率 (<1e-7) は枝刈りする。

ライン情報がないレース (複数人ラインが存在しない) は None を返し、
呼び出し側が ML 単体へフォールバックする。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from keirin.simulation.line_strength import LineStrength, compute_line_strengths, _z
from keirin.simulation.transition import DEFAULT_PARAMS, TransitionParams

_PRUNE_EPS = 1e-7


@dataclass
class SimResult:
    p_win: dict[int, float]    # car_no → P(1着)
    p_top3: dict[int, float]   # car_no → P(3着内)
    p_front: dict[int, float]  # car_no → 自ラインが主導権を取る確率


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _pl_win_top3(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plackett-Luce: P(1着) と P(3着内) を閉形式で計算 (O(n²))."""
    n = len(weights)
    W = weights.sum()
    p1 = weights / W
    if n <= 3:
        return p1, np.ones(n)

    # P(i 2nd) = w_i · Σ_{a≠i} (w_a/W) / (W−w_a)
    denom_a = W - weights                      # (n,)
    inv_after_a = (weights / W) / denom_a      # (w_a/W)/(W−w_a)
    p2 = weights * (inv_after_a.sum() - inv_after_a)

    # P(i 3rd) = w_i · Σ_{a≠b, a≠i, b≠i} (w_a/W)·(w_b/(W−w_a)) / (W−w_a−w_b)
    wa = weights[:, None]
    wb = weights[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        M = (wa / W) * (wb / (W - wa)) / (W - wa - wb)
    np.fill_diagonal(M, 0.0)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0)
    S_total = M.sum()
    p3 = weights * (S_total - M.sum(axis=0) - M.sum(axis=1))

    return p1, np.clip(p1 + p2 + p3, 0.0, 1.0)


def _line_variants(
    line: LineStrength,
    z_chase: dict[int, float],
    z_rating: dict[int, float],
    params: TransitionParams,
) -> list[tuple[float, list[int]]]:
    """ライン内追い抜きの (確率, 車順) バリエーション."""
    cars = line.cars
    if len(cars) < 2:
        return [(1.0, cars)]

    leader, second = cars[0], cars[1]
    p2 = _sigmoid(
        params.gamma0
        + params.gamma1 * z_chase.get(second, 0.0)
        + params.gamma2 * (z_rating.get(second, 0.0) - z_rating.get(leader, 0.0))
    )
    variants: list[tuple[float, list[int]]] = []
    if len(cars) == 2:
        variants.append((1.0 - p2, [leader, second]))
        variants.append((p2, [second, leader]))
        return variants

    third = cars[2]
    p3 = _sigmoid(
        params.gamma0
        + params.gamma1 * z_chase.get(third, 0.0)
        + params.gamma2 * (z_rating.get(third, 0.0) - z_rating.get(leader, 0.0))
        + params.gamma3
    )
    # 正規化: 同時成立は稀なので排他近似 (p2 優先)
    p_keep = max(0.0, 1.0 - p2 - p3 * (1.0 - p2))
    rest = cars[3:]
    variants.append((p_keep, cars))
    variants.append((p2, [second, leader, third] + rest))
    variants.append((p3 * (1.0 - p2), [third, leader, second] + rest))
    return variants


def simulate_race(
    race_df: pd.DataFrame,
    params: TransitionParams = DEFAULT_PARAMS,
    *,
    n_sims: int = 2000,   # 互換用 (v0 は厳密列挙のため未使用)
    seed: int = 42,       # 同上
) -> SimResult | None:
    """1レースの展開シミュレーション。

    race_df: 1行=1車。必要列: car_no, line_id, line_pos, rating,
             b_count, nige_cnt, makuri_cnt, sashi_cnt, mark_cnt, bank_length。
    ライン情報がない (複数人ラインが1本もない) 場合は None。
    """
    df = race_df.reset_index(drop=True)
    if df.empty or len(df) < 4:
        return None

    lines = compute_line_strengths(df)
    multi = [l for l in lines if len(l.cars) >= 2]
    if not multi or len(lines) < 2:
        return None

    # レース内 z (C フェーズで使用)
    z_chase_s = _z(
        pd.to_numeric(df.get("sashi_cnt"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("mark_cnt"), errors="coerce").fillna(0)
    )
    z_rating_s = _z(df.get("rating", pd.Series(dtype=float)).reindex(df.index))
    cars_all = df["car_no"].astype(int).tolist()
    z_chase = dict(zip(cars_all, z_chase_s.tolist()))
    z_rating = dict(zip(cars_all, z_rating_s.tolist()))

    bank = pd.to_numeric(df.get("bank_length"), errors="coerce").dropna()
    bank_v = float(bank.iloc[0]) if not bank.empty else 400.0
    is_short = bank_v <= 350
    is_long = bank_v >= 500

    L = len(lines)
    fp = np.array([l.front_power for l in lines])
    tp = np.array([l.total_power for l in lines])
    solo = np.array([1.0 if l.is_solo else 0.0 for l in lines])

    # --- Phase A: 主導権確率 -------------------------------------------------
    logits = (fp - params.delta_solo * solo) / max(params.tau, 1e-6)
    logits -= logits.max()
    pA = np.exp(logits)
    pA /= pA.sum()

    # --- Phase B: 捲り成功確率 s[l, F] ---------------------------------------
    base_b = params.beta0 + (params.beta2 if is_short else 0.0) + (params.beta3 if is_long else 0.0)
    s_mat = 1.0 / (1.0 + np.exp(-(base_b + params.beta1 * (fp[:, None] - fp[None, :]))))

    # --- Phase C-1: ライン内バリエーション (静的・ライン毎) -------------------
    variants = [_line_variants(l, z_chase, z_rating, params) for l in lines]

    # 多人数ラインのインデックス (バリエーションが複数あるもののみ直積に参加)
    var_idx = [i for i in range(L) if len(variants[i]) > 1]

    p_win_acc = {c: 0.0 for c in cars_all}
    p_top3_acc = {c: 0.0 for c in cars_all}

    static_order_succ = np.argsort(-fp)   # 捲り成功ライン同士は先行力降順
    static_order_fail = np.argsort(-tp)   # 失敗ライン同士は総合力降順

    others_cache = list(range(L))

    for F in range(L):
        pF = float(pA[F])
        if pF < _PRUNE_EPS:
            continue
        others = [l for l in others_cache if l != F]
        for mask in itertools.product((0, 1), repeat=len(others)):
            p_mask = pF
            for l, bit in zip(others, mask):
                s = float(s_mat[l, F])
                p_mask *= s if bit else (1.0 - s)
                if p_mask < _PRUNE_EPS:
                    break
            if p_mask < _PRUNE_EPS:
                continue

            succ = [l for l, bit in zip(others, mask) if bit]
            fail = [l for l, bit in zip(others, mask) if not bit]
            succ.sort(key=lambda l: list(static_order_succ).index(l))
            fail.sort(key=lambda l: list(static_order_fail).index(l))
            line_order = succ + [F] + fail

            # ライン内バリエーションの直積
            for combo in itertools.product(*[range(len(variants[i])) for i in var_idx]):
                p_sc = p_mask
                chosen = {}
                for i, vi in zip(var_idx, combo):
                    pv, order = variants[i][vi]
                    p_sc *= pv
                    chosen[i] = order
                    if p_sc < _PRUNE_EPS:
                        break
                if p_sc < _PRUNE_EPS:
                    continue

                provisional: list[int] = []
                for li in line_order:
                    provisional.extend(chosen.get(li, lines[li].cars))

                ranks = {c: r + 1 for r, c in enumerate(provisional)}
                w = np.array([
                    np.exp(-params.lam * ranks[c] + params.mu * z_rating.get(c, 0.0))
                    for c in cars_all
                ])
                p1, pt3 = _pl_win_top3(w)
                for k, c in enumerate(cars_all):
                    p_win_acc[c] += p_sc * float(p1[k])
                    p_top3_acc[c] += p_sc * float(pt3[k])

    total = sum(p_win_acc.values())
    if total <= 0:
        return None
    # 枝刈り分を正規化
    p_win = {c: v / total for c, v in p_win_acc.items()}
    t3_total = sum(p_top3_acc.values())
    scale3 = (3.0 / t3_total) if t3_total > 0 else 1.0
    p_top3 = {c: min(1.0, v * scale3) for c, v in p_top3_acc.items()}

    # p_front: 自ラインの主導権確率を車単位に展開
    p_front: dict[int, float] = {}
    for i, l in enumerate(lines):
        for c in l.cars:
            p_front[c] = float(pA[i])

    return SimResult(p_win=p_win, p_top3=p_top3, p_front=p_front)
