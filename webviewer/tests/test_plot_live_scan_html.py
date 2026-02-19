import os
from html.parser import HTMLParser


def test_plot_live_scan_exists() -> None:
    p: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plot_live_scan.html")
    assert os.path.exists(p), f"plot_live_scan.html not found at {p}"


class IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs) -> None:
        for k, v in attrs:
            if k.lower() == "id" and v:
                self.ids.add(v)


def test_expected_ids_present() -> None:
    p: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plot_live_scan.html")
    with open(p, "r", encoding="utf-8") as f: os.TextIOWrapper[_WrappedBuffer]:
        html: str = f.read()
    parser = IdCollector()
    parser.feed(html)
    # expected UI element ids
    expected: set[str] = {"plot", "viewType", "xaxis", "yaxis", "inner_xaxis", "inner_yaxis", "metric", "interval"}
    missing: set[str] = expected - parser.ids
    assert not missing, f"Missing expected element IDs in plot_live_scan.html: {missing}"
