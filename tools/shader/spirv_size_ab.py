"""A/B the SPIR-V size of shader capsules offline with shader_check.

Usage: python tools/shader/spirv_size_ab.py <shader_check.exe> <capsule dir> <ENV=a> <ENV=b>
           [--jobs N] [--bench <spvbench.exe>] [--keep <dir>]
Translates every .cap twice, once per environment setting (for example
KYTY_MOVREL_LANE_BOUNDS=0 and KYTY_MOVREL_LANE_BOUNDS=1), and prints the total SPIR-V words, the
largest modules and the biggest changes. A capsule that fails in only one setting is listed.

--bench also cold-compiles both sides' modules on this machine's driver with spvbench, one module
at a time, and compares pipeline create times and select counts: the word count alone does not
predict what the driver charges. --keep writes the modules to <dir>/a and <dir>/b instead of a
temporary folder, so they can be benched again, disassembled or stripped later.
"""
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile


def run(tool, cap, env_pair, spv_path=None):
    key, value = env_pair.split("=", 1)
    env = dict(os.environ)
    env[key] = value
    if spv_path and os.path.exists(spv_path):
        os.remove(spv_path)  # a module left by an earlier run must not be benched as this one
    with tempfile.TemporaryDirectory() as tmp:
        env["SHADER_CHECK_SPIRV"] = spv_path or os.path.join(tmp, "out.spv")
        proc = subprocess.run([tool, cap], env=env, capture_output=True, text=True, timeout=600)
    match = re.search(r"SPIRV words=(\d+)", proc.stdout)
    if proc.returncode != 0 or match is None:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or ["?"]
        return None, tail[0]
    return int(match.group(1)), ""


BENCH_LINE = re.compile(r"^\s*(\S+) ms words=(\d+) selects=(\d+) stage=(\S+) (\S+) (ok|FAIL)(.*)$")


def bench(tool, folder):
    """Cold create time and select count per module name, or the failure reason."""
    proc = subprocess.run([tool, folder], capture_output=True, text=True)
    out = {}
    for line in proc.stdout.splitlines():
        match = BENCH_LINE.match(line)
        if match is None:
            continue
        name = match.group(5)[: -len(".spv")]
        if match.group(6) == "ok":
            out[name] = (float(match.group(1)), int(match.group(3)), "")
        else:
            out[name] = (None, int(match.group(3)), match.group(7).strip())
    if proc.returncode not in (0, 1) or not out:
        sys.exit("spvbench failed: %s" % ((proc.stdout + proc.stderr).strip().splitlines()[-1:] or ["?"])[0])
    header = proc.stdout.splitlines()[0] if proc.stdout else ""
    return out, header


def report_bench(caps, env_a, env_b, bench_a, bench_b):
    rows = []
    for cap in caps:
        name = os.path.basename(cap)
        a, b = bench_a.get(name), bench_b.get(name)
        if a is None or b is None:
            continue
        if (a[0] is None) != (b[0] is None):
            print("CREATE FAILS ON ONE SIDE: %s  %s: %s | %s: %s" %
                  (name, env_a, a[2] or "ok", env_b, b[2] or "ok"))
            continue
        if a[0] is None:
            print("CREATE FAILS ON BOTH SIDES: %s: %s" % (name, a[2]))
            continue
        rows.append((name, a[0], b[0], a[1], b[1]))
    ms_a = sum(r[1] for r in rows)
    ms_b = sum(r[2] for r in rows)
    sel_a = sum(r[3] for r in rows)
    sel_b = sum(r[4] for r in rows)
    print("benched: %d modules" % len(rows))
    print("total cold create: %s %.0f ms -> %s %.0f ms (%+.1f%%)" %
          (env_a, ms_a, env_b, ms_b, 100.0 * (ms_b - ms_a) / max(ms_a, 1e-9)))
    print("total OpSelect: %d -> %d (%+.1f%%)" % (sel_a, sel_b, 100.0 * (sel_b - sel_a) / max(sel_a, 1)))
    print("slowest modules (%s):" % env_a)
    for name, a, b, sa, sb in sorted(rows, key=lambda r: -r[1])[:15]:
        print("  %-40s %8.1f -> %8.1f ms  selects %6d -> %6d" % (name, a, b, sa, sb))
    print("largest create-time changes:")
    for name, a, b, sa, sb in sorted(rows, key=lambda r: -abs(r[2] - r[1]))[:10]:
        print("  %-40s %8.1f -> %8.1f ms (%+.1f%%)" % (name, a, b, 100.0 * (b - a) / max(a, 1e-9)))


def option(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def main():
    tool, folder, env_a, env_b = sys.argv[1:5]
    jobs = int(option("--jobs") or os.cpu_count())
    bench_tool = option("--bench")
    keep = option("--keep")
    caps = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".cap"))
    out_root = None
    if bench_tool or keep:
        out_root = keep or tempfile.mkdtemp(prefix="spirv_ab_")
        for side in ("a", "b"):
            os.makedirs(os.path.join(out_root, side), exist_ok=True)

    def spv_path(cap, env):
        if out_root is None:
            return None
        side = "a" if env == env_a else "b"
        return os.path.join(out_root, side, os.path.basename(cap) + ".spv")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        futures = {pool.submit(run, tool, cap, env, spv_path(cap, env)): (cap, env)
                   for cap in caps for env in (env_a, env_b)}
        for future in concurrent.futures.as_completed(futures):
            cap, env = futures[future]
            results[(cap, env)] = future.result()
    total_a = total_b = 0
    rows = []
    for cap in caps:
        (a, err_a), (b, err_b) = results[(cap, env_a)], results[(cap, env_b)]
        name = os.path.basename(cap)
        if a is None or b is None:
            if (a is None) != (b is None):
                print("ONLY ONE FAILS: %s  %s: %s | %s: %s" % (name, env_a, err_a or "ok", env_b, err_b or "ok"))
            continue
        total_a += a
        total_b += b
        rows.append((name, a, b))
    failed_both = sum(1 for cap in caps if results[(cap, env_a)][0] is None and results[(cap, env_b)][0] is None)
    print("capsules: %d, translated: %d, failed in both: %d" % (len(caps), len(rows), failed_both))
    print("total SPIR-V words: %s %d -> %s %d (%.1f%%)" %
          (env_a, total_a, env_b, total_b, 100.0 * (total_b - total_a) / max(total_a, 1)))
    print("largest modules (%s):" % env_a)
    for name, a, b in sorted(rows, key=lambda r: -r[1])[:15]:
        print("  %-40s %8d -> %8d (%+.1f%%)" % (name, a, b, 100.0 * (b - a) / a))
    changed = [r for r in rows if r[1] != r[2]]
    print("modules changed: %d; any grew: %d" % (len(changed), sum(1 for r in changed if r[2] > r[1])))
    for name, a, b in sorted(changed, key=lambda r: r[2] - r[1])[:5]:
        print("  largest saving %-30s %8d -> %8d" % (name, a, b))
    for name, a, b in sorted(changed, key=lambda r: r[1] - r[2])[:3]:
        if b > a:
            print("  GREW %-40s %8d -> %8d" % (name, a, b))

    if bench_tool:
        # Serial and after translation: concurrent compiles would share the driver's threads.
        bench_a, header = bench(bench_tool, os.path.join(out_root, "a"))
        bench_b, _ = bench(bench_tool, os.path.join(out_root, "b"))
        print(header)
        report_bench(caps, env_a, env_b, bench_a, bench_b)
    if out_root is not None:
        if keep:
            print("modules kept in %s" % out_root)
        else:
            shutil.rmtree(out_root, ignore_errors=True)


if __name__ == "__main__":
    main()
