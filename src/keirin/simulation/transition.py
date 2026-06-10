"""展開シミュレーションの遷移パラメータ.

v0: ドメイン知識による手置き定数 (DEFAULT_PARAMS)。
v1: 決まり手データが2-3ヶ月蓄積したら fit_transition_params() で弱教師推定に
    切り替える (決まり手=逃げ → 勝者ライン=主導権ライン、捲り → 非主導権ライン、
    差し/マーク → line_pos>=2 という弱ラベルを使う)。

パラメータは計14個。ライン付き+全着順付きレースが ~2,000 を超えるまでは
同時推定せず、段階別 (A/B/C) に分割して各2-4パラメータずつ推定すること。
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class TransitionParams:
    # Phase A: 主導権 (先行) 争い
    tau: float = 1.0          # softmax 温度 (小さいほど front_power 差が支配的)
    delta_solo: float = 0.5   # 単騎ラインの主導権ペナルティ

    # Phase B: 捲り成功判定 (sigmoid ロジット)
    beta0: float = -1.2       # 基準ロジット (≈23% 捲り成功)
    beta1: float = 0.8        # (自ライン先行力 − 主導権ラインの粘り) 係数
    beta2: float = -0.4       # 短バンク (<=350m) ペナルティ (捲りにくい)
    beta3: float = 0.4        # 長バンク (>=500m) ボーナス (捲りやすい)

    # Phase C-1: 番手差し判定 (sigmoid ロジット)
    gamma0: float = -0.4      # 基準ロジット (≈40% 番手差し)
    gamma1: float = 0.5       # 番手の 差し+マーク力 (z) 係数
    gamma2: float = 0.4       # (番手rating − 先行rating) z差 係数
    gamma3: float = -0.7      # 三番手のペナルティ

    # Phase C-2: 隊列 → 最終着順の Plackett-Luce
    lam: float = 0.55         # 暫定隊列順位の支配力 (大きいほど展開どおり)
    mu: float = 0.35          # 地力 (rating z) による食い込み補正


DEFAULT_PARAMS = TransitionParams()


def fit_transition_params(
    engine: Engine,
    from_date: str,
    to_date: str,
) -> TransitionParams:
    """v1: 決まり手×ライン位置の弱ラベルから遷移パラメータを段階別推定する。

    必要データ: ライン付き (line_id 2台以上) かつ kimarite 付きのレース。
    蓄積が不十分な場合は ValueError を投げる — 呼び出し側は DEFAULT_PARAMS に
    フォールバックすること。

    推定方針 (実装は決まり手の蓄積後):
      Phase A: 「1着の決まり手=逃げ」のレースで勝者ライン=主導権ラインとみなし、
               tau / delta_solo を front_power と勝者ラインの対数尤度でグリッド探索
      Phase B: 「決まり手=捲り」を捲り成功の正例、それ以外を負例として
               beta0-3 をロジスティック回帰
      Phase C: 「決まり手=差し/マーク」(line_pos>=2 の勝者) を番手差しの正例として
               gamma0-3 をロジスティック回帰、lam/mu は決まり手分布の KL 最小化
    """
    from sqlalchemy import text

    with engine.begin() as conn:
        n = conn.execute(text(
            """
            SELECT COUNT(DISTINCT r.race_id)
            FROM results r
            JOIN entries e ON e.race_id = r.race_id
            WHERE r.kimarite IS NOT NULL AND r.finish = 1
              AND e.line_id IS NOT NULL
              AND r.race_id IN (
                SELECT race_id FROM entries WHERE line_id IS NOT NULL
                GROUP BY race_id HAVING COUNT(*) >= 2)
              AND (SELECT date FROM races WHERE race_id = r.race_id)
                  BETWEEN :fd AND :td
            """
        ), {"fd": from_date, "td": to_date}).scalar()

    if not n or n < 500:
        raise ValueError(
            f"決まり手×ライン付きレースが不足 ({n or 0} < 500)。"
            " backfill_results / backfill_lines の蓄積後に再実行してください。"
        )

    # NOTE: 蓄積が条件を満たした時点で段階別推定を実装する (計画 Phase 2 v1)。
    raise NotImplementedError(
        f"データは {n} レースあります。段階別推定の実装を有効化してください。"
    )
