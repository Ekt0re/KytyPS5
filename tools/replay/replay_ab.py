#!/usr/bin/env python3
"""A/B comparison of two emulator builds or settings on one frame capture (after AnyPS5
tools/frame_replay_compare.py, re-written for kyty_emulator --replay and its replay.json).

  run      replays the capture with A and B in turns (A B A B ...), each run with
           --replay-repeat so that its "steady" numbers leave out the first, compiling pass,
           then compares them
  compare  compares replay.json files that already exist (two folders of runs, or two files)

The verdict uses the median of each run's steady median frame time (flip to flip) and, when
the capture has no flips in a pass, the steady median pass time. The 95% interval of the
difference comes from a bootstrap over the runs; a change counts only with at least 5 runs
of each, when that interval leaves out 0 and a rank-sum test agrees (p < 0.05). Two runs of
the same build can differ by a few percent, so fewer runs give no verdict. The image of every
run is checked against the game's reference (replay.json
"image"), so a faster build that breaks the picture is reported too.

Examples:
  replay_ab.py run --capture caps/gta6-intro --a old/kyty_emulator --b new/kyty_emulator
  replay_ab.py run --capture caps/x --a kyty_emulator --b kyty_emulator --b-env KYTY_ZERO_COPY=1
  replay_ab.py compare out/a out/b

Wall times follow the GPU's clocks and whatever else the machine runs: close other programs,
and on NVIDIA lock the clocks (nvidia-smi -lgc / -lmc) when the difference is small.
"""
import argparse
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path


def load(path: Path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def metrics(report):
    steady = report.get("steady") or {}
    frame = (steady.get("frame_ms") or {}).get("median")
    count = (steady.get("frame_ms") or {}).get("count", 0)
    passes = (steady.get("pass_ms") or {}).get("median")
    gpu_cpu = (steady.get("gpu_thread_cpu_ms") or {}).get("median")
    image = report.get("image")
    finished = all(p.get("finished", False) for p in report.get("passes", []))
    return {
        # Flips that come back at once (a capture with no GPU work between them) say nothing.
        "frame_ms": frame if count and frame and frame >= 0.05 else None,
        "pass_ms": passes,
        "gpu_thread_cpu_ms": gpu_cpu,
        "identical": image.get("identical") if image else None,
        "psnr_db": image.get("psnr_db") if image else None,
        "finished": finished and report.get("exit_code", 0) == 0,
    }


def bootstrap(a, b, rounds=4000, seed=1):
    """95% interval of median(b) - median(a), in % of median(a)."""
    rng = random.Random(seed)
    base = statistics.median(a)
    samples = []
    for _ in range(rounds):
        ra = [rng.choice(a) for _ in a]
        rb = [rng.choice(b) for _ in b]
        samples.append((statistics.median(rb) - statistics.median(ra)) / statistics.median(ra) * 100.0)
    samples.sort()
    return samples[int(0.025 * rounds)], samples[int(0.975 * rounds) - 1], base


MIN_RUNS = 5  # per side, before a difference is called


def rank_sum_p(a, b):
    """Two-sided p of the Mann-Whitney rank-sum test (exact for small samples)."""
    values = sorted(a + b)
    def rank(v):
        low = values.index(v)
        return low + values.count(v) / 2 + 0.5
    ranks = [rank(v) for v in a + b]
    observed = sum(ranks[len(a):])
    n, m = len(a), len(b)
    expected = m * (n + m + 1) / 2
    if math.comb(n + m, m) <= 200000:
        extreme = total = 0
        for chosen in itertools.combinations(ranks, m):
            total += 1
            if abs(sum(chosen) - expected) >= abs(observed - expected) - 1e-9:
                extreme += 1
        return extreme / total
    sigma = math.sqrt(n * m * (n + m + 1) / 12)
    z = (abs(observed - expected) - 0.5) / sigma
    return math.erfc(z / math.sqrt(2))


def compare(runs_a, runs_b, label_a="A", label_b="B"):
    ma = [metrics(r) for r in runs_a]
    mb = [metrics(r) for r in runs_b]
    key = "frame_ms" if all(m["frame_ms"] is not None for m in ma + mb) else "pass_ms"
    result = {"metric": key, "runs": [len(ma), len(mb)]}
    print(f"metric: steady median {key.replace('_ms', '')} time (ms), {len(ma)} run(s) of {label_a}, "
          f"{len(mb)} of {label_b}")
    ok = True
    for label, ms in ((label_a, ma), (label_b, mb)):
        values = [m[key] for m in ms if m[key] is not None]
        unfinished = sum(1 for m in ms if not m["finished"])
        images = [m["identical"] for m in ms if m["identical"] is not None]
        line = f"  {label}: " + ", ".join(f"{v:.3f}" for v in values)
        if values:
            line += f"  -> median {statistics.median(values):.3f}"
        if images:
            line += f"; image {min(images) * 100:.2f}% identical (worst run)"
        if unfinished:
            line += f"; {unfinished} run(s) did not finish"
            ok = False
        print(line)
    va = [m[key] for m in ma if m[key] is not None]
    vb = [m[key] for m in mb if m[key] is not None]
    if not va or not vb:
        print("no comparable runs")
        return {**result, "verdict": "no data"}
    diff = (statistics.median(vb) - statistics.median(va)) / statistics.median(va) * 100.0
    p = None
    if len(va) >= 2 and len(vb) >= 2:
        low, high, _ = bootstrap(va, vb)
        p = rank_sum_p(va, vb)
        interval = f" (95% interval {low:+.1f}% .. {high:+.1f}%, rank-sum p {p:.3f})"
        # Both the interval and the rank test must agree, on enough runs: a few runs of the same
        # build differ by a few percent from clocks and caches alone.
        if min(len(va), len(vb)) < MIN_RUNS:
            verdict = f"inconclusive: fewer than {MIN_RUNS} runs each (use --rounds {MIN_RUNS} or more)"
        elif high < 0 and p < 0.05:
            verdict = f"{label_b} is faster"
        elif low > 0 and p < 0.05:
            verdict = f"{label_b} is slower"
        else:
            verdict = "no significant difference"
    else:
        low = high = None
        interval = " (one run each: no interval; use --rounds 3 or more)"
        verdict = "inconclusive"
    for label, ms in ((label_a, ma), (label_b, mb)):
        gpu = [m["gpu_thread_cpu_ms"] for m in ms if m["gpu_thread_cpu_ms"] is not None]
        if gpu:
            print(f"  {label} GPU command thread CPU: median {statistics.median(gpu):.2f} ms per pass")
    ia = [m["identical"] for m in ma if m["identical"] is not None]
    ib = [m["identical"] for m in mb if m["identical"] is not None]
    if ia and ib and min(ib) + 0.001 < min(ia):
        print(f"  WARNING: {label_b}'s image matches the game less ({min(ib) * 100:.2f}% vs "
              f"{min(ia) * 100:.2f}% identical pixels)")
        verdict += f", but its image differs more"
    print(f"{label_b} vs {label_a}: {diff:+.2f}%{interval}: {verdict}")
    return {**result, "diff_percent": diff, "interval": [low, high], "rank_sum_p": p, "verdict": verdict,
            "ok": ok}


def reports_in(path: Path):
    if path.is_file():
        return [load(path)]
    files = sorted(path.glob("**/replay.json"))
    if not files:
        sys.exit(f"no replay.json under {path}")
    return [load(f) for f in files]


def parse_env(items):
    env = {}
    for item in items or []:
        if "=" not in item:
            sys.exit(f"--*-env expects NAME=VALUE, got {item}")
        name, value = item.split("=", 1)
        env[name] = value
    return env


def run_one(exe, env, capture, out, repeat, timeout, extra):
    out.mkdir(parents=True, exist_ok=True)
    command = [exe, "--replay", str(capture), "--replay-out", str(out), "--replay-repeat", str(repeat)]
    command += extra
    started = time.monotonic()
    with open(out / "stdout.txt", "w", encoding="utf-8", errors="replace") as log:
        try:
            code = subprocess.run(command, env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    report = out / "replay.json"
    if not report.exists():
        print(f"    {out}: no replay.json (exit {code}); see {out / 'stdout.txt'}")
        return None
    data = load(report)
    m = metrics(data)
    shown = m["frame_ms"] if m["frame_ms"] is not None else m["pass_ms"]
    print(f"    {out.name}: exit {code}, {shown:.3f} ms, {time.monotonic() - started:.0f} s")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="replay with A and B in turns and compare")
    run.add_argument("--capture", type=Path, required=True)
    run.add_argument("--a", required=True, help="emulator executable for A")
    run.add_argument("--b", required=True, help="emulator executable for B (may equal A)")
    run.add_argument("--a-env", action="append", metavar="NAME=VALUE")
    run.add_argument("--b-env", action="append", metavar="NAME=VALUE")
    run.add_argument("--rounds", type=int, default=5, help="runs of each (default 5)")
    run.add_argument("--repeat", type=int, default=4, help="--replay-repeat per run (default 4)")
    run.add_argument("--warmup", type=int, default=1,
                     help="unmeasured runs of each first, to fill the shader caches (default 1)")
    run.add_argument("--timeout", type=int, default=1800, help="seconds per run (default 1800)")
    run.add_argument("--out", type=Path, default=Path("replay_ab"))
    run.add_argument("--json", type=Path, help="write the result here too")
    run.add_argument("extra", nargs="*", help="after --: more arguments for both")
    cmp = sub.add_parser("compare", help="compare existing replay.json files")
    cmp.add_argument("a", type=Path, help="replay.json, or a folder searched for them")
    cmp.add_argument("b", type=Path)
    cmp.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.command == "compare":
        result = compare(reports_in(args.a), reports_in(args.b))
    else:
        env_a, env_b = parse_env(args.a_env), parse_env(args.b_env)
        sides = (("a", args.a, env_a), ("b", args.b, env_b))
        for index in range(args.warmup):
            for name, exe, env in sides:
                print(f"warm-up {index + 1}: {name}")
                run_one(exe, env, args.capture, args.out / f"warmup-{name}-{index + 1}", 1, args.timeout, args.extra)
        runs = {"a": [], "b": []}
        for index in range(args.rounds):
            # A B, then B A: neither side always runs right after the other's caches were warm.
            order = sides if index % 2 == 0 else sides[::-1]
            print(f"round {index + 1} of {args.rounds}")
            for name, exe, env in order:
                data = run_one(exe, env, args.capture, args.out / f"{name}-{index + 1}", args.repeat,
                               args.timeout, args.extra)
                if data is not None:
                    runs[name].append(data)
        result = compare(runs["a"], runs["b"])
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sys.exit(0 if result.get("ok", True) else 1)


if __name__ == "__main__":
    main()
