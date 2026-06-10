"""normalize_style のテスト (文字化け修復・表記ゆれ)."""
from __future__ import annotations

import pytest

from keirin.features.style_norm import normalize_style


@pytest.mark.parametrize("raw,expected", [
    # 正常値
    ("逃", "逃"), ("両", "両"), ("追", "追"),
    # cp932 誤デコードの文字化け (実DBに混在していた値)
    ("騾�", "逃"), ("霑ｽ", "追"), ("荳｡", "両"),
    # 旧 _style の別表記
    ("マーク", "追"), ("マ", "追"), ("捲", "両"), ("差", "追"),
    # 接尾辞付き
    ("逃げ", "逃"),
    # 判別不能・空
    ("ﾐｸﾒ旆", None), ("", None), (None, None), ("  ", None),
])
def test_normalize_style(raw, expected):
    assert normalize_style(raw) == expected
