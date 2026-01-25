"""Quick demo of the parallel sweep worker behavior with a tiny grid.
This runs a few simulations with --workers=2 and prints progress/ETA.
"""

from _thread import lock
from _thread import lock
import csv
import os
import queue
import subprocess
import threading
import time

EXE: str = (
    "./build/evo-movement-sim.exe"
    if os.path.exists("./build/evo-movement-sim.exe")
    else "./build/Release/evo-movement-sim.exe"
)
EXE: str = os.path.normpath(EXE)
print("Using exe:", EXE)

# tiny demo grid (4 points)
pairs: list[tuple[float, float]] = [(0.0, 0.02), (0.0, 0.025), (0.05, 0.02), (0.05, 0.025)]
job_q = queue.Queue()
for p: tuple[float, float] in pairs:
    job_q.put(p)

outpath = "demo_results.csv"
writer_lock: lock = threading.Lock()
active_procs = set()
active_lock: lock = threading.Lock()
stats = {"completed": 0, "start_time": None}


def parse_extinction(output_text) -> None | int:
    extinct = None
    for line in output_text.splitlines():
        if "population extinct at t=" in line:
            try:
                t = int(line.split("population extinct at t=")[1].split(",")[0])
                extinct: int = t
            except Exception:
                extinct = -1
    return extinct


def worker() -> None:
    while True:
        try:
            mv, mc = job_q.get_nowait()
        except queue.Empty:
            return
        cmd: list[str] = [
            EXE,
            "--steps=1000",
            f"--movement-cost-factor={mv}",
            f"--maintenance-cost-per-node={mc}",
            "--no-progress",
        ]
        start: float = time.time()
        with active_lock:
            proc: os.Popen[str] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            active_procs.add(proc)
        try:
            out, err = proc.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            with active_lock:
                try:
                    proc.kill()
                except Exception:
                    pass
                active_procs.discard(proc)
            # timed out; process killed
        with active_lock:
            active_procs.discard(proc)
        # parse extinction info if needed (not used in this demo)
        elapsed: float = time.time() - start
        stats["completed"] += 1
        completed = stats["completed"]
        elapsed_total = time.time() - stats["start_time"]
        avg = elapsed_total / completed if completed > 0 else 0
        remaining: int = max(0, len(pairs) - completed)
        eta_seconds = int(remaining * avg)
        hrs, rem = divmod(eta_seconds, 3600)
        mins, secs = divmod(rem, 60)
        rem_str: str = f"{hrs}:{mins:02d}:{secs:02d}"
        pct = completed / len(pairs) * 100
        print(
            f"demo {completed}/{len(pairs)} ({pct:.1f}%) - last {mv},{mc} took {elapsed:.1f}s - time remaining {rem_str}"
        )


def main() -> None:
    workers = 2
    csvf: os.TextIOWrapper[_WrappedBuffer] = open(outpath, "w", newline="")
    writer: csv.Writer = csv.writer(csvf)
    writer.writerow(["movement_cost_factor", "maintenance_cost_per_node", "extinction_time", "survived"])
    stats["start_time"] = time.time()
    threads = []
    for i: int in range(workers):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nInterrupted by user, terminating demo...")
        with active_lock:
            for p in list(active_procs):
                try:
                    p.terminate()
                except Exception:
                    pass
    for t in threads:
        t.join()
    csvf.close()


if __name__ == "__main__":
    main()
