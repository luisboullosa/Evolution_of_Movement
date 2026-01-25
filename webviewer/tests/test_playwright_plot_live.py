import json
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from playwright.sync_api import sync_playwright

ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def start_static_server(root_dir) -> tuple[ThreadingHTTPServer, int]:
    handler: partial[SimpleHTTPRequestHandler] = partial(SimpleHTTPRequestHandler, directory=str(root_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port: int = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def write_test_progress(jobdir) -> None:
    os.makedirs(jobdir, exist_ok=True)
    prog = [
        {
            "job": 0,
            "movement_cost_factor": 0.1,
            "maintenance_cost_per_node": 1e-05,
            "resource_scale": 10.0,
            "grid_width": 25,
            "final_population": 1,
            "prop_moved": 0.12,
            "params": {"movement_cost_factor": 0.1, "maintenance_cost_per_node": 1e-05},
        },
        {
            "job": 1,
            "movement_cost_factor": 1.0,
            "maintenance_cost_per_node": 0.001,
            "resource_scale": 10.0,
            "grid_width": 25,
            "final_population": 0,
            "prop_moved": 0.78,
            "params": {"movement_cost_factor": 1.0, "maintenance_cost_per_node": 0.001},
        },
    ]
    with open(os.path.join(jobdir, "progress.json"), "w", encoding="utf-8") as f: os.TextIOWrapper[_WrappedBuffer]:
        json.dump(prog, f)
    manifest = {
        "total_jobs": 2,
        "movement_vals": [0.1, 1.0],
        "maintenance_vals": [1e-05, 0.001],
        "resource_scales": [10.0],
        "grid_widths": [25],
        "movement_vals": [0.1, 1.0],
        "maintenance_vals": [1e-05, 0.001],
    }
    with open(os.path.join(jobdir, "scan_manifest.json"), "w", encoding="utf-8") as f: os.TextIOWrapper[_WrappedBuffer]:
        json.dump(manifest, f)


def test_metric_prop_moved_in_overview() -> None:
    # prepare a test jobs folder under output/scan_population
    jobs_base: str = os.path.join(ROOT, "output", "scan_population")
    test_jobs: str = os.path.join(jobs_base, "jobs_test")
    write_test_progress(test_jobs)

    server, port = start_static_server(ROOT)
    url: str = f"http://127.0.0.1:{port}/webviewer/plot_live_scan.html"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            page.goto(url, timeout=10000)
            # wait for controls to become available
            page.wait_for_selector("#metric", timeout=5000)
            assert page.locator('#metric option[value="prop_moved"]').count() == 1

            # switch to overview and select prop_moved
            page.select_option("#viewType", "overview")
            page.select_option("#metric", "prop_moved")
            # trigger an update and wait briefly for plot render
            page.click("#refresh")
            page.wait_for_timeout(1000)

            # check that plot contains title text indicating Prop moved distribution
            # Plotly places title text inside the plot; search for that string
            texts = page.locator("#plot").all_text_contents()
            joined: str = "\n".join(texts)
            assert "Prop moved" in joined or "prop_moved" in joined

            context.close()
            browser.close()
    finally:
        try:
            server.shutdown()
        except Exception:
            pass
        # cleanup test files
        try:
            os.remove(os.path.join(test_jobs, "progress.json"))
            os.remove(os.path.join(test_jobs, "scan_manifest.json"))
            os.rmdir(test_jobs)
        except Exception:
            pass
