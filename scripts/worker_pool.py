#!/usr/bin/env python3
import subprocess
import csv
import threading
import queue
import time


def _parse_extinction(output_text):
    extinct = None
    for line in (output_text or "").splitlines():
        if "population extinct at t=" in line:
            try:
                t = int(line.split("population extinct at t=")[1].split(",")[0])
                extinct = t
            except Exception:
                extinct = -1
    return extinct


def run_param_sweep(
    pairs,
    cmd_builder,
    outpath,
    header,
    workers=4,
    timeout=1200,
    writer_mode="w",
    result_row_builder=None,
):
    """Run a parameter sweep in parallel using worker threads.

    pairs: iterable of job parameters (e.g., (mv, mc))
    cmd_builder: callable(job) -> list(command parts)
    outpath: CSV file to write rows to
    header: list of CSV column names
    workers: number of threads
    timeout: per-process timeout in seconds
    writer_mode: file mode for CSV (default 'w')
    """
    job_q = queue.Queue()
    pairs = list(pairs)
    total = len(pairs)
    for p in pairs:
        job_q.put(p)

    writer_lock = threading.Lock()
    active_procs = set()
    active_lock = threading.Lock()
    stats = {"completed": 0, "start_time": None}

    def worker(thread_id):
        while True:
            try:
                job = job_q.get_nowait()
            except queue.Empty:
                return
            cmd = cmd_builder(job)
            start = time.time()
            with active_lock:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                active_procs.add(proc)
            try:
                out, err = proc.communicate(timeout=timeout)
                out_text = (out or "") + "\n" + (err or "")
            except subprocess.TimeoutExpired:
                with active_lock:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    active_procs.discard(proc)
                out_text = ""
            with active_lock:
                active_procs.discard(proc)

            # try to parse extinction and produce a canonical row
            extinct = _parse_extinction(out_text)
            survived = 1 if extinct is None else 0

            # ensure mv/mc exist for logging fallback
            mv = None
            mc = None
            try:
                # attempt to unpack common (mv, mc) tuple if present
                mv, mc = job
            except Exception:
                # ignore - job may be a different shape
                pass

            # If a custom row builder was provided, use it to generate the CSV row.
            row = None
            if callable(result_row_builder):
                try:
                    row = result_row_builder(job, extinct, out_text)
                except Exception:
                    row = None

            # fallback to backward-compatible rows if no custom row produced
            if row is None:
                # if header expects only survived (coarse scan), adapt
                if header == [
                    "movement_cost_factor",
                    "maintenance_cost_per_node",
                    "survived",
                ]:
                    row = [mv, mc, int(survived)]
                else:
                    # default extinction_time column assumed to exist
                    # use 50000 as default max if extinction is None
                    max_steps = 50000
                    row = [mv, mc, extinct if extinct is not None else max_steps, survived]

            elapsed = time.time() - start
            with writer_lock:
                writer.writerow(row)
                csvf.flush()
                stats["completed"] += 1
                completed = stats["completed"]
                elapsed_total = time.time() - stats["start_time"]
                avg = elapsed_total / completed if completed > 0 else 0
                remaining = max(0, total - completed)
                eta_seconds = int(remaining * avg)
                hrs, rem = divmod(eta_seconds, 3600)
                mins, secs = divmod(rem, 60)
                rem_str = f"{hrs}:{mins:02d}:{secs:02d}"
                pct = completed / total * 100
                # show last job parameters (if available)
                last_params = (
                    f"{mv},{mc}" if (mv is not None and mc is not None) else str(job)
                )
                print(
                    f"sim {completed}/{total} ({pct:.1f}%) - last {last_params} "
                    f"took {elapsed:.1f}s - time remaining {rem_str}"
                )

    # open writer and start threads
    try:
        csvf = open(outpath, writer_mode, newline="")
        writer = csv.writer(csvf)
        writer.writerow(header)
        stats["start_time"] = time.time()
        threads = []
        for i in range(max(1, workers)):
            t = threading.Thread(target=worker, args=(i,), daemon=True)
            t.start()
            threads.append(t)
        while any(t.is_alive() for t in threads):
            try:
                time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nInterrupted by user. Terminating running simulations...")
                with active_lock:
                    for p in list(active_procs):
                        try:
                            p.terminate()
                        except Exception:
                            pass
                break
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Partial results saved to", outpath)
    finally:
        try:
            csvf.close()
        except Exception:
            pass

    print("Done. Results in", outpath)
