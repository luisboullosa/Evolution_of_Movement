import json
import math
import os
import secrets
import statistics
import subprocess
import tempfile

NN_NODES = 10
NN_NODES2 = 10


def carrying_capacity(width, resource_scale, time_amplitude, maint, death_risk):
    mean_base = 0.5 * resource_scale
    mean_time = 0.5 * time_amplitude
    mean_res = (mean_base + mean_time) * resource_scale
    maint_total = (NN_NODES + NN_NODES2) * maint
    if maint_total <= 0 or mean_res <= 0:
        return math.inf
    k = (mean_res * width) / maint_total
    if death_risk > 0:
        k *= max(0.1, 1.0 - min(death_risk * 0.5, 0.9))
    return k


scenarios = [
    dict(name="K<100", width=25, resource_scale=1.0, maint=0.05, time_amplitude=1.0, death_risk=0.0),
    dict(name="100<K<1000", width=50, resource_scale=1.0, maint=0.01, time_amplitude=1.0, death_risk=0.0),
    dict(name="1000<K<10000", width=100, resource_scale=2.0, maint=0.01, time_amplitude=1.0, death_risk=0.0),
    dict(name="K>10000", width=200, resource_scale=5.0, maint=0.005, time_amplitude=1.0, death_risk=0.0),
]

BIN = "evo_movement_rust"
MANIFEST = os.path.normpath("rust/Cargo.toml")
RUNS = 2
STEPS = 2000

base_cfg = dict(
    movement_cost_factor=0.5,
    reproduction_threshold=20.0,
    reproduction_cost=1.0,
    env_time_speed=0.01,
    env_perlin_time_scale=1e-5,
    show_progress=False,
    write_frames=False,
    write_outputs=False,
    write_config=False,
    write_snapshot=False,
    report_interval=STEPS + 1,
    max_population=0,
)

results = []

for sc in scenarios:
    est_k = carrying_capacity(sc["width"], sc["resource_scale"], sc["time_amplitude"], sc["maint"], sc["death_risk"])
    init_pop = max(1, int(est_k * 0.5))
    cfg_base = dict(base_cfg)
    cfg_base.update(
        grid_width=int(sc["width"]),
        resource_scale=float(sc["resource_scale"]),
        env_time_amplitude=float(sc["time_amplitude"]),
        maintenance_cost_per_node=float(sc["maint"]),
        death_risk=float(sc["death_risk"]),
        steps=STEPS,
        initial_population=init_pop,
        sim_prefix=f"kcheck_{sc['name'].replace('>','gt').replace('<','lt').replace(' ','_')}",
    )

    final_pops = []
    seeds = []
    for run_idx in range(RUNS):
        cfg = dict(cfg_base)
        seed = secrets.randbelow(2**31 - 1)
        if seed == 0:
            seed = 1
        cfg["env_perlin_seed"] = seed
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
        json.dump(cfg, tf)
        tf.flush()
        tf.close()
        cmd = [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            MANIFEST,
            "--bin",
            BIN,
            "--",
            "--config-file",
            tf.name,
            "--no-write-outputs",
        ]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            out = e.stdout or ""
        finally:
            try:
                os.unlink(tf.name)
            except Exception:
                pass

        final_pop = None
        for line in (out or "").splitlines()[::-1]:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and "final_population" in obj:
                final_pop = obj.get("final_population")
                break
        final_val = final_pop if final_pop is not None else 0
        final_pops.append(final_val)
        seeds.append(seed)

    mean = statistics.mean(final_pops)
    stdev = statistics.pstdev([p for p in final_pops]) if len(final_pops) > 1 else 0.0
    stderr = stdev / math.sqrt(len(final_pops)) if len(final_pops) > 0 else float("inf")
    results.append(
        dict(name=sc["name"], est_k=est_k, init=init_pop, pops=final_pops, seeds=seeds, mean=mean, stderr=stderr)
    )

print("SCENARIO, est_K, init_pop, mean_final_pop, stderr, pops")
for r in results:
    print(f"{r['name']}, {r['est_k']:.1f}, {r['init']}, {r['mean']:.1f}, {r['stderr']:.2f}, {r['pops']}")
