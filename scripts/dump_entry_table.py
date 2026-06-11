"""
出走表テーブルの実HTML構造を診断用にダンプする (一時スクリプト)。

parse_line_assignments が cells[3] を「ライン」セルと想定していたが、
実データは {1,2,3,4,5,100} の小さい整数のみで、ライン構成("3-1-5"等)
とは一致しないことが backfill_lines.yml の結果判明した
(line_id 付与 0/5980)。実テーブル構造を直接確認するため、対象レースの
出走表テーブル行を丸ごとHTMLダンプする。

クラウド環境は 403 でスクレイプ不可 → GitHub Actions で実行すること。

使い方: python scripts/dump_entry_table.py [race_id ...]
"""
from __future__ import annotations
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from bs4 import BeautifulSoup

from keirin.scraper.session import fetch_text
from keirin.scraper.netkeirin import entry_url


def main() -> None:
    race_ids = sys.argv[1:] or ["202605087508"]
    cookie = os.environ.get("NETKEIRIN_COOKIE", "")
    headers = {
        "User-Agent": "KeirinResearchBot/0.1 (personal; pagudaruma@gmail.com)",
        "Accept-Language": "ja-JP,ja;q=0.9",
    }
    if cookie:
        headers["Cookie"] = cookie

    out: list[str] = []
    for rid in race_ids:
        url = entry_url(rid)
        html = fetch_text(url, headers=headers)
        out.append(f"===== {rid} ({url}) html_len={len(html)} =====")
        if not html:
            out.append("  (empty response)")
            continue
        soup = BeautifulSoup(html, "lxml")
        tables = soup.select("table")
        out.append(f"  tables: {len(tables)}")
        if len(tables) < 2:
            out.append("  (<2 tables, skip)")
            continue
        rows = tables[1].select("tbody tr")
        out.append(f"  rows: {len(rows)}")
        for ri, row in enumerate(rows[:3]):
            cells = row.select("td")
            out.append(f"  -- row {ri}: {len(cells)} cells --")
            for ci, cell in enumerate(cells):
                text = cell.get_text(strip=True)
                html_snip = str(cell)
                if len(html_snip) > 400:
                    html_snip = html_snip[:400] + "...(truncated)"
                out.append(f"    [{ci}] text={text!r}")
                out.append(f"         html={html_snip}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/entry_table_dump.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
