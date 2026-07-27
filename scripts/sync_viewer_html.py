"""Sync viewer template into live output HTML, preserving embedded landscape JSON."""
import re
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
template = (repo / "rust/assets/fitness_landscape_viewer.html").read_text(encoding="utf-8")
output_path = repo / "output/evolution_hunt_staged_all/fitness_landscape_heatmap.html"
output = output_path.read_text(encoding="utf-8")

m = re.search(r'id="landscape-data"[^>]*>(.*?)</script>', output, re.S)
if not m:
    raise SystemExit("no landscape-data embed in output HTML")
embed = m.group(1)

new_html = template.replace("__LANDSCAPE_JSON__", embed, 1)
if "__LANDSCAPE_JSON__" not in new_html.split(embed, 1)[1]:
    pass  # sentinel preserved in JS
elif "raw === '__LANDSCAPE_JSON__'" not in new_html:
    raise SystemExit("JS sentinel missing after sync")

output_path.write_text(new_html, encoding="utf-8")
print("synced", output_path, "bytes", len(new_html))
