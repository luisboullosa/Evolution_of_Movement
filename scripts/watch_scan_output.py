#!/usr/bin/env python3
"""Simple poll-based watcher for a scan output folder.

Writes append-only events to a log and prints new files/dirs as they appear.
Use when the scan is running to observe any new per-job folders or files.
"""

import argparse
import json
import os
import time
from datetime import datetime


def now():
    return datetime.utcnow().isoformat() + "Z"


def snapshot(path):
    s = set()
    for root, dirs, files in os.walk(path):
        for d in dirs:
            s.add(os.path.join(root, d))
        for f in files:
            s.add(os.path.join(root, f))
    return s


def append_log(logpath, msg):
    try:
        with open(logpath, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="output/scan_population", help="Folder to watch")
    p.add_argument("--interval", type=float, default=1.0, help="Poll interval seconds")
    p.add_argument("--log", default=None, help="Path to append events (defaults to <path>/monitor.log)")
    args = p.parse_args()

    watch_path = os.path.normpath(args.path)
    if args.log:
        logpath = args.log
    else:
        logpath = os.path.join(watch_path, "monitor.log")

    os.makedirs(watch_path, exist_ok=True)

    seen = snapshot(watch_path)
    last_progress_count = None
    print(f"[{now()}] Watching {watch_path} (log -> {logpath})")
    append_log(logpath, f"[{now()}] START watching {watch_path}")

    try:
        while True:
            time.sleep(args.interval)
            cur = snapshot(watch_path)
            added = sorted(cur - seen)
            removed = sorted(seen - cur)

            for pth in added:
                typ = "DIR" if os.path.isdir(pth) else "FILE"
                msg = f"[{now()}] ADDED {typ} {pth}"
                print(msg)
                append_log(logpath, msg)

            for pth in removed:
                msg = f"[{now()}] REMOVED {pth}"
                print(msg)
                append_log(logpath, msg)

            # monitor progress.json length changes (concise)
            prog = os.path.join(watch_path, "progress.json")
            try:
                if os.path.exists(prog):
                    with open(prog, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    cnt = len(data) if isinstance(data, list) else None
                    if cnt is not None and cnt != last_progress_count:
                        last_progress_count = cnt
                        msg = f"[{now()}] PROGRESS entries={cnt}"
                        print(msg)
                        append_log(logpath, msg)
            except Exception:
                pass

            seen = cur
    except KeyboardInterrupt:
        print(f"[{now()}] Stopped by user")
        append_log(logpath, f"[{now()}] STOPPED")


if __name__ == "__main__":
    main()
