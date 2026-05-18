"""Verify the dashboard renders end-to-end with the example payload."""
from __future__ import annotations

from keirin.reporting.html_renderer import example_payload, render_dashboard
from keirin.reporting.markdown import render_markdown


def test_html_renders():
    html = render_dashboard(example_payload())
    assert "<title>決戦速報" in html
    assert "bib bib-3" in html
    assert "EV" in html
    assert "PASS" in html


def test_markdown_renders():
    md = render_markdown(example_payload())
    assert md.startswith("# 決戦速報")
    assert "| 買い目 |" in md
    assert "見送りレース" in md
