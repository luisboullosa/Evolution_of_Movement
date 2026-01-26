#!/usr/bin/env python3
from _thread import lock
import csv
import os
import queue
import subprocess
import threading
import time
from collections.abc import Iterable


def _parse_extinction(output_text) -> None | int:
    extinct = None
    for line in (output_text or "").splitlines():
        if "population extinct at t=" in line:
            try:
                t = int(line.split("population extinct at t=")[1].split(",")[0])
                extinct: int = t
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
    progress_callback=None,
    stream_live_output=False,
):
    """Run a parameter sweep in parallel using worker threads.

    pairs: iterable of job parameters (e.g., tuples)
    cmd_builder: callable(job) -> list(command parts) where job == (idx, params)
    outpath: CSV file to write rows to
    header: list of CSV column names
    workers: number of threads
    timeout: per-process timeout in seconds
    writer_mode: file mode for CSV (default 'w')
    result_row_builder: callable to convert (job, out_text) -> csv row (flexible signatures supported)
    progress_callback: callable(completed, total, csv_row, job) called after each job completes
    """
    pairs_list = list(pairs)
    total: int = len(pairs_list)
    if total == 0:
        print("No jobs to run.")
        return

    job_q = queue.Queue()
    for idx in range(total):
        job_q.put(idx)

    writer_lock: lock = threading.Lock()
    active_procs = set()
    active_lock: lock = threading.Lock()
    stats = {"completed": 0, "start_time": time.time(), "last_params": None, "last_elapsed": 0.0}

    # open CSV file once
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    csvf: os.IO[csv.Any] = open(outpath, writer_mode, newline="", encoding="utf-8")
    writer: csv.Writer = csv.writer(csvf)
    if writer_mode == "w":
        writer.writerow(header)
        csvf.flush()

    def worker(thread_id) -> None:
        while True:
            try:
                idx = job_q.get_nowait()
            except queue.Empty:
                return
            i = idx
            row = pairs_list[idx]
            job = (i, row)
            cmd = cmd_builder(job)
            start: float = time.time()
            proc: os.Popen[str] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            with active_lock:
                active_procs.add(proc)
            try:
                if stream_live_output and proc.stdout is not None:
                    # stream stdout lines to console with job prefix but also collect
                    # the lines into `collected` so callers can inspect full output
                    # after the process ends.
                    collected = []

                    def _stream_output(rfile, jid, buf) -> None:
                        try:
                            last_print = 0.0
                            last_line = None
                            interval = 30.0  # seconds between printed live updates per job
                            for raw in rfile:
                                line = raw.rstrip("\n")
                                if not line:
                                    continue
                                now: float = time.time()
                                last_line = line
                                buf.append(line)
                                # always print important messages immediately
                                if "population extinct at t=" in line or "Done." in line:
                                    print(f"[job {i}] {line}")
                                    last_print: float = now
                                    last_line = None
                                    continue
                                if (now - last_print) >= interval:
                                    # print last observed line as a concise update
                                    print(f"[job {i}] {last_line}")
                                    last_print: float = now
                                    last_line = None
                            # on stream end, flush last observed line once
                            if last_line:
                                print(f"[job {i}] {last_line}")
                        except Exception:
                            pass

                    t_stream = threading.Thread(target=_stream_output, args=(proc.stdout, i, collected), daemon=True)
                    t_stream.start()
                    try:
                        proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    # ensure stream thread finishes
                    try:
                        t_stream.join(timeout=0.5)
                    except Exception:
                        pass
                    # use the collected buffer as the captured output
                    try:
                        out: str = "\n".join(collected)
                    except Exception:
                        out: str = ""
                    err: str = ""
                else:
                    out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
                out, err = proc.communicate()
            out_text: str = (out or "") + "\n" + (err or "")
            with active_lock:
                try:
                    active_procs.discard(proc)
                except Exception:
                    pass

            # build csv row using flexible result_row_builder
            csv_row = None
            if callable(result_row_builder):
                exc_val: None | int = _parse_extinction(out_text)
                # try common signatures, swallow and log exceptions so threads don't die
                try:
                    csv_row: object = result_row_builder(job, out_text)
                except Exception:
                    try:
                        csv_row: object = result_row_builder(job, exc_val, out_text)
                    except Exception:
                        try:
                            csv_row: object = result_row_builder(row, out_text)
                        except Exception:
                            try:
                                csv_row: object = result_row_builder(row, out_text, exc_val)
                            except Exception:
                                try:
                                    csv_row: object = result_row_builder(job)
                                except Exception:
                                    csv_row = None

            if csv_row is None:
                # fallback: flatten row
                csv_row = list(row) if isinstance(row, (list, tuple)) else [row]

            elapsed: float = time.time() - start

            # normalize csv_row to an iterable of fields (avoid passing non-iterable or string)
            if not isinstance(csv_row, Iterable) or isinstance(csv_row, (str, bytes)):
                csv_row: list[object | str | bytes] = [csv_row]
            else:
                csv_row = list(csv_row)

            # write row safely
            with writer_lock:
                writer.writerow(csv_row)
                csvf.flush()
                stats["completed"] += 1
                stats["last_params"] = str(row)
                stats["last_elapsed"] = elapsed

            # call progress callback
            try:
                if callable(progress_callback):
                    progress_callback(stats["completed"], total, csv_row, job)
            except Exception:
                pass

            job_q.task_done()

    # start threads
    threads = []
    for tid in range(min(workers, total)):
        t = threading.Thread(target=worker, args=(tid,), daemon=True)
        t.start()
        threads.append(t)

    # monitor progress
    try:
        last_print_time = 0
        last_completed = -1
        throttle = 2.0  # seconds between progress updates
        while any(t.is_alive() for t in threads):
            completed = stats.get("completed", 0)
            now: float = time.time()
            # print only when completed changed or at most every `throttle` seconds
            if completed != last_completed or (now - last_print_time) >= throttle:
                elapsed_total = now - stats["start_time"]
                avg = elapsed_total / completed if completed > 0 else 0
                remaining: int = max(0, total - completed)
                eta_seconds = int(remaining * avg)
                hrs, rem = divmod(eta_seconds, 3600)
                mins, secs = divmod(rem, 60)
                rem_str: str = f"{hrs}:{mins:02d}:{secs:02d}"
                pct = (completed / total * 100) if total > 0 else 0.0
                last = stats.get("last_params", "-")
                last_elapsed = stats.get("last_elapsed", 0.0)
                bar_len = 40
                filled: int = int(bar_len * (completed / total)) if total > 0 else 0
                bar: str = "#" * filled + "-" * (bar_len - filled)
                print(
                    f"[{bar}] {completed}/{total} ({pct:.1f}%) last={last} t={last_elapsed:.1f}s ETA={rem_str}",
                    end="\r",
                    flush=True,
                )
                last_print_time: float = now
                last_completed = completed
            try:
                time.sleep(0.25)
            except KeyboardInterrupt:
                print("\nInterrupted by user. Terminating running simulations...")
                with active_lock:
                    for p in list(active_procs):
                        try:
                            p.terminate()
                        except Exception:
                            pass
                break
        print()
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
