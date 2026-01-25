import json
import shutil
import subprocess
import time
from pathlib import Path

WORKSPACE: Path = Path(__file__).resolve().parents[2]


def find_rust_bin() -> str:
    rels: list[Path] = [
        WORKSPACE / "rust" / "target" / "release" / "evo_movement_rust.exe",
        WORKSPACE / "rust" / "target" / "debug" / "evo_movement_rust.exe",
    ]
    for p in rels:
        if p.exists():
            return str(p)
    raise FileNotFoundError("Rust simulator binary not found; build it first")


def run_sim_and_collect(config: dict, cli_overrides=None, timeout=60):
    if cli_overrides is None:
        cli_overrides = []

    binp: str = find_rust_bin()

    out_base: Path = WORKSPACE / "output" / "test_sim_runs"
    if out_base.exists():
        shutil.rmtree(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    # ensure sim writes to this output base so we can inspect simdir
    webviewer: Path = out_base
    config["webviewer_base"] = str(webviewer)
    # Enable config + JSONL outputs for validation. Note: frames.tsv is legacy and
    # not required for the Python tooling.
    config.setdefault("write_config", True)
    config.setdefault("write_outputs", True)
    config.setdefault("write_frames", False)
    # keep runs short and deterministic
    config.setdefault("steps", 60)
    config.setdefault("report_interval", 5)
    config.setdefault("reproduction_threshold", 1e9)

    cfg_path: Path = out_base / "job_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f)

    cmd = [binp, "--config-file", str(cfg_path), "--config-out=sim_config.json"] + cli_overrides

    proc: subprocess.CompletedProcess[str] = subprocess.run(cmd, cwd=str(WORKSPACE), capture_output=True, text=True, timeout=timeout)

    # find simdir created under webviewer_base
    sims: list[Path] = [p for p in webviewer.iterdir() if p.is_dir() and p.name.startswith(config.get("sim_prefix", "sim"))]
    if not sims:
        raise AssertionError(f"No sim directory created under {webviewer}")
    # pick the newest
    simdir: Path = max(sims, key=lambda p: p.stat().st_mtime)

    # allow small delay for atomic rename to settle
    time.sleep(0.1)

    result = {
        "proc": proc,
        "simdir": simdir,
    }
    return result


def test_max_and_initial_override() -> None:
    cfg = {
        "sim_prefix": "testsim",
        "sim_digits": 4,
        "initial_population": 5,
        "max_population": 10,
    }
    # pass CLI overrides to ensure flags override file values as well
    cli: list[str] = ["--initial-population", "5", "--max-population", "10"]
    r = run_sim_and_collect(cfg, cli_overrides=cli, timeout=120)
    simdir = r["simdir"]

    # verify config.json exists and contains our overrides
    cfgfile = simdir / "config.json"
    assert cfgfile.exists(), f"config.json not found in {simdir}"
    with open(cfgfile, "r", encoding="utf-8") as f:
        got = json.load(f)
    assert int(got.get("initial_population", 0)) == 5
    assert int(got.get("max_population", 0)) == 10

    # verify individuals.jsonl has at least one record and individuals list is present
    indf = simdir / "individuals.jsonl"
    assert indf.exists(), "individuals.jsonl missing"
    with open(indf, "r", encoding="utf-8") as f:
        lines: list[str] = [l.strip() for l in f.readlines() if l.strip()]
    assert lines, "individuals.jsonl is empty"
    last = json.loads(lines[-1])
    assert "individuals" in last
    assert isinstance(last["individuals"], list)
    assert len(last["individuals"]) <= 10


def test_env_override_and_outputs_present() -> None:
    cfg = {
        "sim_prefix": "testsim",
        "sim_digits": 4,
        "env_time_speed": 3.14,
        "steps": 40,
        "report_interval": 5,
    }
    r = run_sim_and_collect(cfg, cli_overrides=["--env-time-speed", "3.14"], timeout=120)
    simdir = r["simdir"]

    # env.jsonl should exist and have at least one record
    envf = simdir / "env.jsonl"
    assert envf.exists(), "env.jsonl missing"
    with open(envf, "r", encoding="utf-8") as f:
        lines: list[str] = [l.strip() for l in f.readlines() if l.strip()]
    assert lines, "env.jsonl is empty"

    # config.json should reflect env_time_speed
    cfgfile = simdir / "config.json"
    assert cfgfile.exists()
    with open(cfgfile, "r", encoding="utf-8") as f:
        got = json.load(f)
    # allow float tolerance
    assert abs(float(got.get("env_time_speed", 0.0)) - 3.14) < 1e-6
