"""ライン力学・展開シミュレーション (独自手法).

競輪を「ライン同士の戦い」としてモデル化する:
  Phase A: どのラインが主導権 (先行) を取るか
  Phase B: 他ラインの仕掛け (捲り) が決まるか
  Phase C: 直線での番手差し + 地力による食い込み

詳細は race_sim.simulate_race を参照。
"""
from keirin.simulation.race_sim import SimResult, simulate_race  # noqa: F401
from keirin.simulation.transition import DEFAULT_PARAMS, TransitionParams  # noqa: F401

SIM_VERSION = "v0"
