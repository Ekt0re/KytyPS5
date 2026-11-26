"""A/B the SPIR-V size of shader capsules offline with shader_check.

Usage: python tools/shader/spirv_size_ab.py <shader_check.exe> <capsule dir> <ENV=a> <ENV=b> [--jobs N]
Translates every .cap twice, once per environment setting (for example
KYTY_MOVREL_LANE_BOUNDS=0 and KYTY_MOVREL_LANE_BOUNDS=1), and prints the total SPIR-V words, the
largest modules and the biggest changes. A capsule that fails in only one setting is listed.
"""
import concurrent.futures
import os
import re
import subprocess
import sys
import tempfile


def run(tool, cap, env_pair):
    key, value = env_pair.split("=", 1)
    env = dict(os.environ)
    env[key] = value
    with tempfile.TemporaryDirectory() as tmp:
        env["SHADER_CHECK_SPIRV"] = os.path.join(tmp, "out.spv")
        proc = subprocess.run([tool, cap], env=env, capture_output=True, text=True, timeout=600)
    match = re.search(r"SPIRV words=(\d+)", proc.stdout)
    if proc.returncode != 0 or match is None:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or ["?"]
        return None, tail[0]
    return int(match.group(1)), ""


def main():
    tool, folder, env_a, env_b = sys.argv[1:5]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else os.cpu_count()
    caps = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".cap"))
    results = {}
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        futures = {pool.submit(run, tool, cap, env): (cap, env) for cap in caps for env in (env_a, env_b)}
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


if __name__ == "__main__":
    main()
