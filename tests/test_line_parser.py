"""parse_line_assignments の修復版テスト.

過去の事故: line_raw が壊れた値のとき car_no=1 だけに line_id=1/line_pos=1 が
付く偽データが全レースに混入した。健全性ガード (割当 < max(2, 出走数/2) なら
全車 None) で構造的に再発を防ぐことを検証する。
"""
from __future__ import annotations

from keirin.scraper.netkeirin import parse_line_assignments


def _entries(n: int, line_raw: str | None = None) -> list[dict]:
    return [
        {"car_no": i, "line_raw": (line_raw if i == 1 else "")}
        for i in range(1, n + 1)
    ]


def _lines_of(result: list[dict]) -> dict[int, tuple]:
    return {
        e["car_no"]: (e.get("line_id"), e.get("line_pos"))
        for e in result
    }


def test_standard_format():
    res = parse_line_assignments(_entries(7, "3-1-5 / 2-7 / 4-6"))
    lines = _lines_of(res)
    assert lines[3] == (1, 1)
    assert lines[1] == (1, 2)
    assert lines[5] == (1, 3)
    assert lines[2] == (2, 1)
    assert lines[7] == (2, 2)
    assert lines[4] == (3, 1)
    assert lines[6] == (3, 2)


def test_fullwidth_and_circled_digits():
    res = parse_line_assignments(_entries(7, "③-①-⑤ ／ ２-７ ／ ４-６"))
    lines = _lines_of(res)
    assert lines[3] == (1, 1)
    assert lines[1] == (1, 2)
    assert lines[2] == (2, 1)
    assert lines[6] == (3, 2)


def test_solo_groups():
    # 単騎 (1台グループ) を含む並び
    res = parse_line_assignments(_entries(7, "3-1-5 / 2-7 / 4 / 6"))
    lines = _lines_of(res)
    assert lines[4] == (3, 1)
    assert lines[6] == (4, 1)


def test_newline_separator():
    res = parse_line_assignments(_entries(7, "3-1-5\n2-7\n4-6"))
    lines = _lines_of(res)
    assert lines[3] == (1, 1)
    assert lines[4] == (3, 1)


def test_broken_input_assigns_nothing():
    """壊れた line_raw (数字1個) では誰にも line を割り当てない (旧バグの再発防止)。"""
    res = parse_line_assignments(_entries(7, "1"))
    lines = _lines_of(res)
    assert all(v == (None, None) for v in lines.values())


def test_partial_coverage_rejected():
    """出走数の半分未満しか割当できない並びは全体を捨てる。"""
    res = parse_line_assignments(_entries(9, "1-2"))
    lines = _lines_of(res)
    assert all(v == (None, None) for v in lines.values())


def test_empty_line_raw():
    res = parse_line_assignments(_entries(7, None))
    lines = _lines_of(res)
    assert all(v == (None, None) for v in lines.values())


def test_duplicate_car_kept_first():
    """同じ車番が複数グループに現れたら最初の出現のみ有効。"""
    res = parse_line_assignments(_entries(5, "1-2-3 / 3-4-5"))
    lines = _lines_of(res)
    assert lines[3] == (1, 3)        # 最初の出現
    assert lines[4] == (2, 1)        # 2グループ目は 3 を除いた並び
    assert lines[5] == (2, 2)


def test_invalid_car_numbers_ignored():
    """出走していない車番は無視される。"""
    res = parse_line_assignments(_entries(5, "1-2-3 / 4-5-9"))
    lines = _lines_of(res)
    assert lines[4] == (2, 1)
    assert lines[5] == (2, 2)
    assert 9 not in lines
