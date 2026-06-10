"""脚質 (style) 文字列の正規化.

netkeirin の脚質は 逃/両/追 の3分類。過去の取り込みで UTF-8 ページを cp932 で
誤デコードした文字化け (騾�=逃, 霑ｽ=追, 荳｡=両) が entries.style / players.style に
混在しているため、特徴量で使う前に必ずこの正規化を通す。

正準値: "逃", "両", "追" (それ以外は None)
"""
from __future__ import annotations

# 完全一致マップ (文字化け値・別表記 → 正準値)
_STYLE_MAP: dict[str, str] = {
    "逃": "逃",
    "両": "両",
    "追": "追",
    # 既存DBの別表記 (ci_morning 旧 _style がマ/追を「マーク」に潰していた)
    "マーク": "追",
    "マ": "追",
    "捲": "両",   # 捲り主体は両に寄せる (netkeirin 3分類に存在しないため)
    "差": "追",
    # cp932 誤デコードの文字化け
    "騾�": "逃",
    "霑ｽ": "追",
    "荳｡": "両",
}

# 前方一致フォールバック (置換文字 � の混入パターン対応)
_PREFIX_MAP: dict[str, str] = {
    "騾": "逃",
    "霑": "追",
    "荳": "両",
}


def normalize_style(s: str | None) -> str | None:
    """脚質文字列を 逃/両/追 のいずれかに正規化。判別不能は None。"""
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    if s in _STYLE_MAP:
        return _STYLE_MAP[s]
    for prefix, canon in _PREFIX_MAP.items():
        if s.startswith(prefix):
            return canon
    # 正常な日本語が先頭に含まれるケース ("逃げ" 等)
    for canon in ("逃", "両", "追"):
        if s.startswith(canon):
            return canon
    if s.startswith("マ"):
        return "追"
    return None
