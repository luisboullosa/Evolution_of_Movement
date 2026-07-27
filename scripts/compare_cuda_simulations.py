#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUST_DIR = ROOT / "rust"
DEFAULT_OUTPUT_BASE = RUST_DIR / "temp_outputs"


def build_map(root: Path) -> dict[str, tuple[int, str]]:
    mapping: dict[str, tuple[int, str]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        mapping[str(rel).replace("\\", "/")] = (
            path.stat().st_size,
            hash_bytes(path)
        )
    return mapping


def hash_bytes(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def diff_outputs(cpu_dir: Path, gpu_dir: Path) -> None:
    cpu_map = build_map(cpu_dir)
    gpu_map = build_map(gpu_dir)
    cpu_keys = set(cpu_map)
    gpu_keys = set(gpu_map)
    only_cpu = sorted(cpu_keys - gpu_keys)
    only_gpu = sorted(gpu_keys - cpu_keys)
    mismatched: list[str] = []
    for rel in sorted(cpu_keys & gpu_keys):
        cpu_size, cpu_hash = cpu_map[rel]
        gpu_size, gpu_hash = gpu_map[rel]
        if cpu_size != gpu_size or cpu_hash != gpu_hash:
            mismatched.append(rel)
    if only_cpu:
        print("Files only in CPU run:")
        for rel in only_cpu:
            print(f"  {rel}")
    if only_gpu:
        print("Files only in GPU run:")
        for rel in only_gpu:
            print(f"  {rel}")
    if mismatched:
        print("Files differing between runs:")
        for rel in mismatched:
            print(f"  {rel}")
    print(f"Compared {len(cpu_keys & gpu_keys)} files between runs.")


def run_simulation(label: str, args: argparse.Namespace, use_gpu: bool) -> tuple[Path, float]:
    output_dir = args.output_base / label
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_ARCH"] = args.cuda_arch
    if not use_gpu:
        env["TENSOR_KERNEL_PTX"] = args.fallback_ptx
    else:
        env.pop("TENSOR_KERNEL_PTX", None)

    cmd = ["cargo", "run", "--bin", "evo_movement_bin_old"]
    if args.release:
        cmd.append("--release")
    cmd.append("--")
    cmd.extend([
        "--config-file",
        str(args.config_file),
        "--steps",
        str(args.steps),
        "--initial-population",
        str(args.initial_population),
        "--report-interval",
        str(args.report_interval),
        "--output-run-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
    ])
    if use_gpu:
        cmd.append("--use-gpu-step")
    else:
        cmd.append("--no-use-gpu-step")

    print(f"Running {label} simulation -> {output_dir}")
    start = time.perf_counter()
    subprocess.run(cmd, cwd=RUST_DIR, env=env, check=True)
    elapsed = time.perf_counter() - start
    print(f"{label} finished in {elapsed:.2f}s")
    return output_dir, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a quick GPU vs CPU comparison using the Rust simulator and compare outputs."
    )
    parser.add_argument("--config-file", type=Path, default=RUST_DIR / "longer_config.json")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--initial-population", type=int, default=100)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--fallback-ptx", type=str, default="missing.ptx")
    parser.add_argument("--cuda-arch", default="sm_89")
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()

    args.output_base.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_ARCH"] = args.cuda_arch
    build_cmd = ["cargo", "build", "--bin", "evo_movement_bin_old"]
    if args.release:
        build_cmd.append("--release")
    subprocess.run(build_cmd, cwd=RUST_DIR, env=env, check=True)

    cpu_dir, cpu_time = run_simulation("cpu", args, use_gpu=False)
    gpu_dir, gpu_time = run_simulation("gpu", args, use_gpu=True)

    print(f"CPU fallback run time: {cpu_time:.2f}s")
    print(f"GPU run time: {gpu_time:.2f}s")

    diff_outputs(cpu_dir, gpu_dir)


if __name__ == "__main__":
    main()
