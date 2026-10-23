#!/usr/bin/env python3
"""
Render authentic Lab Bench animation from engine-streamed JSONL frames.
Shows the circular hex petri dish with amber Perlin light field and lineage-colored
organisms on the left, alongside the live BrainDiagram neural network inspector on the right.
"""

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
STREAM_FILE = ROOT_DIR / "rust" / "temp_outputs" / "lab_bench_stream.jsonl"
OUTPUT_DIR = ROOT_DIR / "temp_outputs" / "lab_bench_render"
ASSETS_DIR = ROOT_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def hex_corner(cx, cy, size, i):
    angle_deg = 60 * i + 30
    angle_rad = math.pi / 180.0 * angle_deg
    return (cx + size * math.cos(angle_rad), cy + size * math.sin(angle_rad))

def hex_polygon(cx, cy, size):
    return [hex_corner(cx, cy, size, i) for i in range(6)]

def render_frame(data, width=1020, height=640):
    img = Image.new("RGB", (width, height), (13, 17, 23)) # GitHub dark console bg
    draw = ImageDraw.Draw(img, "RGBA")
    
    # Fonts
    try:
        font_title = ImageFont.truetype("arialbd.ttf", 14)
        font_sub = ImageFont.truetype("arial.ttf", 11)
        font_mono = ImageFont.truetype("consola.ttf", 11)
        font_mono_bold = ImageFont.truetype("consolab.ttf", 12)
        font_node = ImageFont.truetype("arialbd.ttf", 9)
    except Exception:
        font_title = font_sub = font_mono = font_mono_bold = font_node = ImageFont.load_default()

    # 1. Top Header
    draw.rectangle([(0, 0), (width, 42)], fill=(22, 27, 34))
    draw.line([(0, 42), (width, 42)], fill=(48, 54, 61), width=1)
    
    t = data.get("t", 0)
    pop = data.get("population", 0)
    sel_id = data.get("selected_id", "N/A")
    sel_res = data.get("selected_resources", 0.0)
    signal = data.get("avg_move_signal", 0.0)
    
    draw.text((16, 12), "LAB BENCH", font=font_title, fill=(240, 246, 252))
    draw.text((105, 14), "·  Controlled Hex Dish Simulator (r=10)", font=font_sub, fill=(139, 148, 158))
    
    metrics_str = f"Step: {t:03d}   |   Pop: {pop:02d}   |   Tracked Org: #{sel_id} (E={sel_res:.1f})   |   Net Motility Signal: {signal:+.3f}"
    draw.text((width - 16, 14), metrics_str, font=font_mono, fill=(201, 209, 217), anchor="ra")

    # 2. Left Panel: Hex Petri Dish
    dish_center_x = 310.0
    dish_center_y = 340.0
    hex_size = 14.0
    
    cells = data.get("cells", [])
    if cells:
        xs = [c["x"] for c in cells]
        ys = [c["y"] for c in cells]
        mid_x = (min(xs) + max(xs)) / 2.0
        mid_y = (min(ys) + max(ys)) / 2.0
        dish_radius_px = (max(xs) - min(xs)) / 2.0 + hex_size * 1.5
    else:
        mid_x = mid_y = 0.0
        dish_radius_px = hex_size * 10.0 * math.sqrt(3.0) + 12.0

    # Outer dish rim
    draw.ellipse(
        [
            (dish_center_x - dish_radius_px, dish_center_y - dish_radius_px),
            (dish_center_x + dish_radius_px, dish_center_y + dish_radius_px),
        ],
        outline=(48, 56, 72),
        width=2,
    )

    # Render hexagonal cells
    for c in cells:
        px = dish_center_x + (c["x"] - mid_x)
        py = dish_center_y + (c["y"] - mid_y)
        poly = hex_polygon(px, py, hex_size * 0.96)
        rgb = tuple(c.get("rgb", [40, 40, 40]))
        draw.polygon(poly, fill=rgb, outline=(30, 36, 44, 180))

    # Render individuals
    inds = data.get("individuals", [])
    for ind in inds:
        ix = dish_center_x + (ind["x"] - mid_x)
        iy = dish_center_y + (ind["y"] - mid_y)
        color = tuple(ind.get("color", [220, 220, 220]))
        r = 6.0
        
        # Halo for selected
        if ind.get("is_selected", False):
            # Glowing cyan/white double ring
            draw.ellipse(
                [(ix - r - 6, iy - r - 6), (ix + r + 6, iy + r + 6)],
                outline=(56, 189, 248, 220),
                width=2,
            )
            draw.ellipse(
                [(ix - r - 3, iy - r - 3), (ix + r + 3, iy + r + 3)],
                outline=(255, 255, 255, 240),
                width=1,
            )

        # Body
        draw.ellipse([(ix - r, iy - r), (ix + r, iy + r)], fill=color, outline=(255, 255, 255, 220), width=1)

    # 3. Right Panel: Brain Diagram
    panel_x0 = 595
    panel_y0 = 56
    panel_w = 405
    panel_h = 560
    panel_x1 = panel_x0 + panel_w
    panel_y1 = panel_y0 + panel_h
    
    # Card Background
    draw.rounded_rectangle([(panel_x0, panel_y0), (panel_x1, panel_y1)], radius=6, fill=(22, 26, 34), outline=(70, 80, 100), width=1)
    
    # Brain Card Header
    draw.text((panel_x0 + 16, panel_y0 + 14), "BRAIN INSPECTOR", font=font_title, fill=(226, 232, 240))
    draw.text((panel_x0 + 16, panel_y0 + 34), f"Selected Individual #{sel_id}  ·  Evolved Circuit Wiring", font=font_sub, fill=(148, 163, 184))
    draw.line([(panel_x0 + 14, panel_y0 + 54), (panel_x1 - 14, panel_y0 + 54)], fill=(48, 54, 61), width=1)
    
    diagram = data.get("brain_diagram")
    if diagram:
        diag_ox = panel_x0 + 20
        diag_oy = panel_y0 + 70
        
        # Draw edges
        for edge in diagram.get("edges", []):
            x1 = diag_ox + edge["x1"]
            y1 = diag_oy + edge["y1"]
            x2 = diag_ox + edge["x2"]
            y2 = diag_oy + edge["y2"]
            c = edge.get("color", {})
            rgba = (c.get("r", 150), c.get("g", 150), c.get("b", 150), c.get("a", 200))
            w = max(1, int(round(edge.get("width", 1.0))))
            draw.line([(x1, y1), (x2, y2)], fill=rgba, width=w)
            
        # Draw nodes
        for node in diagram.get("nodes", []):
            nx = diag_ox + node["x"]
            ny = diag_oy + node["y"]
            nr = node.get("r", 9.0)
            fill_c = node.get("fill", {})
            rgb = (fill_c.get("r", 100), fill_c.get("g", 150), fill_c.get("b", 255))
            
            draw.ellipse([(nx - nr, ny - nr), (nx + nr, ny + nr)], fill=rgb, outline=(0, 0, 0), width=2)
            lbl = node.get("label", "")
            draw.text((nx, ny), lbl, font=font_node, fill=(255, 255, 255), anchor="mm")
            
            # Side label
            if node.get("side_label"):
                draw.text((nx - nr - 5, ny), node["side_label"], font=font_node, fill=(180, 190, 205), anchor="rm")
                
        # Diagram footer texts
        for text in diagram.get("texts", []):
            tx = diag_ox + text.get("x", 0)
            ty = diag_oy + text.get("y", 0)
            if ty > panel_h - 10:
                ty = panel_h - 14
            t_str = text.get("text", "")
            c = text.get("color", {})
            t_rgb = (c.get("r", 160), c.get("g", 165), c.get("b", 175))
            draw.text((tx, diag_oy + text.get("y", 0)), t_str, font=font_sub, fill=t_rgb)
    else:
        draw.text((panel_x0 + 40, panel_y0 + 120), "No active individual selected", font=font_sub, fill=(148, 163, 184))

    # 4. Global Footer
    draw.line([(0, height - 22), (width, height - 22)], fill=(30, 36, 44), width=1)
    draw.text((16, height - 17), "Hexagonal Torus Physics  ·  Heterosperma Motility Plasticity  ·  Common Garden G_move Assay", font=font_sub, fill=(100, 110, 125))
    draw.text((width - 16, height - 17), "Built with Rust + egui (evo_movement_lab)", font=font_sub, fill=(100, 110, 125), anchor="ra")

    return img

def main():
    if not STREAM_FILE.exists():
        print(f"Error: Stream file {STREAM_FILE} not found!")
        sys.exit(1)
        
    print(f"Reading stream from {STREAM_FILE}...")
    frames = []
    with open(STREAM_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    print(f"Rendering {len(lines)} Lab Bench frames...")
    rendered_images = []
    
    for idx, line in enumerate(lines):
        data = json.loads(line)
        img = render_frame(data)
        rendered_images.append(img)
        if idx == 0:
            preview_path = ASSETS_DIR / "lab_bench_preview.png"
            img.save(preview_path, format="PNG")
            print(f"Saved initial frame preview to {preview_path}")

    # Save animated GIF
    out_gif = ASSETS_DIR / "lab_bench_simulation.gif"
    print(f"Compiling animated GIF to {out_gif}...")
    rendered_images[0].save(
        out_gif,
        save_all=True,
        append_images=rendered_images[1:],
        duration=70,  # ~14 fps
        loop=0,
        optimize=True,
    )
    print(f"Animated GIF saved: {out_gif.stat().st_size / (1024 * 1024):.2f} MB")

    # Render MP4 via ffmpeg
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        temp_img_dir = OUTPUT_DIR / "frames"
        temp_img_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(rendered_images):
            img.save(temp_img_dir / f"frame_{i:04d}.png")
            
        out_mp4 = ASSETS_DIR / "lab_bench_simulation.mp4"
        cmd = [
            ffmpeg_exe,
            "-y",
            "-framerate", "14",
            "-i", str(temp_img_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "20",
            str(out_mp4),
        ]
        print(f"Compiling MP4 video to {out_mp4}...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"MP4 video saved: {out_mp4.stat().st_size / (1024 * 1024):.2f} MB")
        shutil.rmtree(temp_img_dir, ignore_errors=True)

    # Copy to artifact directory
    artifact_dir = Path("C:/Users/luisf/.gemini/antigravity-ide/brain/d9032cc4-f2ec-4bdb-8a09-a7a788854dfb")
    if artifact_dir.exists():
        shutil.copy(out_gif, artifact_dir / "lab_bench_simulation.gif")
        shutil.copy(ASSETS_DIR / "lab_bench_preview.png", artifact_dir / "lab_bench_preview.png")
        if (ASSETS_DIR / "lab_bench_simulation.mp4").exists():
            shutil.copy(ASSETS_DIR / "lab_bench_simulation.mp4", artifact_dir / "lab_bench_simulation.mp4")
        print("Copied assets to IDE artifacts directory.")

if __name__ == "__main__":
    main()
