import os


def test_scan_script_contains_manifest_writer() -> None:
    p: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scan_rust_population_scan.py")
    assert os.path.exists(p), f"scan_rust_population_scan.py not found at {p}"
    txt: str = open(p, "r", encoding="utf-8").read()
    # sanity checks: the script should write a scan_manifest.json and a plotly/plot_live HTML
    assert "scan_manifest.json" in txt, "scan_manifest.json write not found in script"
    assert "plotly_live.html" in txt or "plotly_live" in txt, "live viewer HTML write not found in script"
