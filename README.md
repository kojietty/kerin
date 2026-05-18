# 競輪予想システム

学生・少額運用向けの日次競輪予想システム。三連単で最大10点、長期回収率 +15〜30% を狙う中庸型構成。

- **券種**: 三連単（1レース最大10点）
- **1日上限**: ¥1,000（コードにハードコード、CLI上書き不可）
- **緊急停止**: 月初比 -¥3,000 で本番購入自動停止 → ペーパー記録のみ
- **自動化**: スクレイピング → 予測 → おすすめ買い目MD/HTML出力まで。**投票は手動**

## 重要な前提

控除率約25%の競輪で「月1万円→月20万円」は数学的に不可能です。本システムは年間回収率 110〜130% を目標とし、**「賭けない（見送り）」を一級市民として扱う**ことで、長期的に資金を守りながら期待値プラス側に寄せる設計です。

## クイックスタート

```powershell
cd C:\Users\pagud\競輪予想
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"

# 設定と安全装置を確認
python -m keirin doctor

# DB初期化
python -m keirin init-db

# サンプルダッシュボードを生成して開く（実データなし、デザイン確認用）
python -m keirin sample-dashboard
# → reports/predictions/sample.html をダブルクリックでブラウザ表示
```

## 段階的ロードマップ

| Phase | 内容 | 状態 |
|---|---|---|
| **P1** | スクレイパー + DB + CLI fetch | 骨格実装済み（URLパス・パーサーは実サイト確認後に微調整） |
| **P2** | 特徴量 + LightGBM + Plackett-Luce展開 + 校正 | combo.py のみ実装済み。学習パイプラインは未 |
| **P3** | バックテスト walk-forward + 買い目選定 | 選定ロジック・stake は実装済み。バックテストは未 |
| **P4** | 日次自動化 + 収支記録 + 緊急停止 | 緊急停止・PnL集計は実装済み |
| **P5** | 二車単・ワイド併用、月次再学習自動化 | 将来 |

## 日次運用CLI（Phase完了後）

```powershell
python -m keirin fetch --date today --kind schedule
python -m keirin fetch --date today --kind cards
python -m keirin fetch --date today --kind odds
python -m keirin predict --date today                # HTML + Markdown を生成
python -m keirin record-result --date yesterday      # 結果反映・収支更新
python -m keirin pnl --month current                 # 月次P&Lサマリ
python -m keirin backfill --from 2026-03-01 --to 2026-05-15   # 過去データ遡及
```

Windows タスクスケジューラに登録すれば cron 相当の自動運用が可能。

## ディレクトリ構成

```
src/keirin/
├── cli.py / __main__.py / config.py / logging_setup.py
├── scraper/    # KEIRIN.JP スクレイピング、レート制限、robots.txt 遵守、生HTMLキャッシュ
├── db/         # SQLite スキーマと repository
├── features/   # 特徴量パイプライン（Phase 2）
├── models/     # combo.py（Plackett-Luce 展開）、train/predict/calibrate は Phase 2
├── betting/    # ev.py / selector.py / stake.py（ハードキャップ・緊急停止）
├── workflow/   # 日次オーケストレーション（Phase 4）
├── reporting/  # html_renderer / markdown / Jinja2 テンプレ / CSS
└── backtest/   # walk-forward / simulator（Phase 3）
configs/      # config.yaml / betting.yaml / features.yaml
data/         # raw HTML キャッシュ + SQLite
reports/      # 日次予想 HTML + Markdown
tests/        # combo / stake / selector / renderer の単体テスト
```

## ダッシュボードデザイン: 「決戦速報」

スポーツタブロイドと金融ターミナルを掛け合わせた個性ある編集デザイン。

- **背景**: 墨色 `#0a0a0a`、テキストは温白 `#f5f0e8`
- **アクセント**: 競輪公式車番カラー（1=白, 2=黒, 3=赤, 4=青, 5=黄, 6=緑, 7=橙, 8=桃, 9=紫）
- **タイポ**: Shippori Mincho B1（明朝）+ Manrope + JetBrains Mono
- **車番ビブ**: 公式色の円形プレートで車番を視覚化
- **モーション**: ページロード時にレースカード段階表示 + オッズカウントアップ
- **収支ヘッダー**: 月初比累積を常時大型表示（依存防止の自覚装置）
- **緊急停止**: -¥3,000 到達で画面に赤バナー + 買い目グレーアウト

ビルド不要の単一HTMLファイル（CSS・JSインライン）。ブラウザでダブルクリックで開く。

## 安全装置（ギャンブル依存配慮）

| 仕組み | 実装場所 |
|---|---|
| 1日上限 ¥1,000 ハードコード | `config.py:HARD_DAILY_BUDGET_YEN` |
| 月次 -¥3,000 で自動停止 | `config.py:HARD_EMERGENCY_STOP_YEN` + `betting/stake.py` |
| マーチンゲール系拡大は実装しない | `betting/stake.py` はフラクショナルKellyのみ |
| 月初比累積を毎回先頭表示 | ダッシュボードPnLバー |
| `prob_min=0.005` で極端な穴を排除 | `configs/betting.yaml` |

## テスト

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests -v
```

11テストのスモークスイートで combo / stake / selector / renderer の正当性を確認しています。

## ライセンスと注意

- 個人研究目的のみ。スクレイピング結果の再配布禁止。
- KEIRIN.JP の利用規約と robots.txt を遵守。レート制限 1req/2sec、1日5000req上限。
- 競輪は控除率約25%。本システムを使っても損失を保証しません。

## 連絡先

`pagudaruma@gmail.com`
