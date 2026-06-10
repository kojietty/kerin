"""
DBデータ品質の一括修復 (ワンショット・冪等)

1. entries.style / players.style の文字化け正規化 (騾�→逃, 霑ｽ→追, 荳｡→両 等)
2. 偽ラインデータの全クリア: 旧パーサ破損により全レースで car_no=1 だけに
   line_id=1/line_pos=1 が付いた偽データを NULL に戻す
   (1台だけのライン割当は構造的にあり得ない → has_line=1 の嘘ラベル除去)
3. venues.bank_angle を静的マスタで充填 (公知のカント角・概算値)
4. player_race_log の着外導出を含む再同期

ネットワーク不要。GitHub Actions (fix_db_quality.yml) で実行して DB を commit する。

使い方:
  python scripts/fix_db_quality.py            # 実行
  python scripts/fix_db_quality.py --dry-run  # 件数確認のみ
"""
from __future__ import annotations
import sys, io, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from sqlalchemy import text

from keirin.config import load_app_config
from keirin.db.engine import get_engine, init_db
from keirin.db.repository import sync_player_race_log
from keirin.features.style_norm import normalize_style

# バンク傾斜角 (カント, 度) — 公知の固定値 (概算)。venue_id → 角度
BANK_ANGLES: dict[str, float] = {
    "11": 30.6, "12": 32.9, "13": 32.9, "21": 32.4, "22": 36.0,
    "23": 31.5, "24": 25.8, "25": 26.3, "26": 29.4, "27": 32.2,
    "28": 31.2, "31": 29.8, "32": 34.0, "34": 32.2, "35": 31.5,
    "36": 35.6, "37": 34.7, "38": 30.7, "42": 34.0, "43": 32.3,
    "44": 30.6, "45": 33.8, "46": 33.7, "47": 34.4, "48": 32.3,
    "51": 31.5, "53": 33.4, "54": 30.5, "55": 32.3, "56": 30.9,
    "61": 30.6, "62": 30.8, "63": 34.7, "71": 33.3, "73": 29.8,
    "74": 24.5, "75": 34.0, "81": 34.0, "83": 31.5, "84": 32.0,
    "85": 31.9, "86": 33.7, "87": 30.3,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_app_config()
    eng = get_engine(cfg.paths.db)
    init_db(eng)

    with eng.begin() as conn:
        # ── 1. style 正規化 ────────────────────────────────────────────────
        for table in ("entries", "players"):
            rows = conn.execute(text(
                f"SELECT DISTINCT style FROM {table} WHERE style IS NOT NULL"
            )).fetchall()
            changed = 0
            for (raw,) in rows:
                canon = normalize_style(raw)
                if canon == raw:
                    continue
                n = conn.execute(text(
                    f"SELECT COUNT(*) FROM {table} WHERE style = :raw"
                ), {"raw": raw}).scalar()
                print(f"  {table}.style {raw!r} → {canon!r} ({n}行)")
                changed += n
                if not args.dry_run:
                    conn.execute(text(
                        f"UPDATE {table} SET style = :canon WHERE style = :raw"
                    ), {"canon": canon, "raw": raw})
            print(f"{table}.style 正規化: {changed} 行")

        # ── 2. 偽ラインクリア ──────────────────────────────────────────────
        # 「そのレースで line_id を持つ車が1台だけ」は旧パーサ破損の痕跡
        fake = conn.execute(text("""
            SELECT COUNT(*) FROM entries e
            WHERE e.line_id IS NOT NULL
              AND (SELECT COUNT(*) FROM entries e2
                   WHERE e2.race_id = e.race_id AND e2.line_id IS NOT NULL) < 2
        """)).scalar()
        print(f"偽ライン行 (レース内割当1台のみ): {fake} 行")
        if not args.dry_run and fake:
            conn.execute(text("""
                UPDATE entries SET line_id = NULL, line_pos = NULL
                WHERE rowid IN (
                  SELECT e.rowid FROM entries e
                  WHERE e.line_id IS NOT NULL
                    AND (SELECT COUNT(*) FROM entries e2
                         WHERE e2.race_id = e.race_id AND e2.line_id IS NOT NULL) < 2
                )
            """))
            # player_race_log 側の line_pos も同期クリア対象だが、
            # sync_player_race_log は COALESCE 更新のため明示クリアする
            conn.execute(text("""
                UPDATE player_race_log SET line_pos = NULL
                WHERE (race_id, player_id) IN (
                  SELECT e.race_id, e.player_id FROM entries e
                  WHERE e.line_id IS NULL AND e.line_pos IS NULL
                ) AND line_pos IS NOT NULL
            """))

        # ── 3. bank_angle 充填 ─────────────────────────────────────────────
        filled = 0
        for vid, angle in BANK_ANGLES.items():
            if args.dry_run:
                continue
            r = conn.execute(text(
                "UPDATE venues SET bank_angle = :a WHERE venue_id = :v AND bank_angle IS NULL"
            ), {"a": angle, "v": vid})
            filled += r.rowcount
        print(f"venues.bank_angle 充填: {filled} 場")

    # ── 4. player_race_log 再同期 (着外導出を含む) ─────────────────────────
    if not args.dry_run:
        n = sync_player_race_log(eng)
        print(f"sync_player_race_log: {n} 行 upsert (着外 finish=9 導出込み)")

        with eng.begin() as conn:
            n9 = conn.execute(text(
                "SELECT COUNT(*) FROM player_race_log WHERE finish = 9"
            )).scalar()
            nf = conn.execute(text(
                "SELECT COUNT(*) FROM player_race_log WHERE finish IS NOT NULL"
            )).scalar()
            tot = conn.execute(text("SELECT COUNT(*) FROM player_race_log")).scalar()
            print(f"player_race_log: finish有り {nf}/{tot} (うち着外 {n9})")

    print("完了")


if __name__ == "__main__":
    main()
