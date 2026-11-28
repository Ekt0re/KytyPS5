#!/usr/bin/env python3
"""Ranks the pipelines in a KYTY_PIPELINE_STATS file, to find the guest shaders that compile
worst on the host GPU (after AnyPS5 APS5_PIPELINE_STATS).

The file holds the driver's own statistics (names and meaning differ between NVIDIA, AMD,
Intel and Mesa drivers), one row per statistic. This tool puts them in one row per pipeline
executable (a compiled stage) and sorts by the statistic --sort names, or by register use when
it is not given. It looks for these by name: registers (VGPRs, "Register Count", ...), spills
(spilled registers, scratch memory) and instructions (instruction count, code size).

With --trace <log> (the log of a run with KYTY_TRACE_SHADERS=1 and --printf-direction Console,
which the TRACE lines need to reach the log; for instance a frame replay) it also
counts how often the captured frames used each pipeline's shaders and ranks by uses x
instructions, which is closer to where GPU time goes than any one statistic. With --shaders
<dir> (kyty_emulator --replay <capture> --replay-shaders <dir>) it names the code file of each
guest shader, to read or disassemble.

  pipeline_stats.py kyty_pipeline_stats.csv
  pipeline_stats.py kyty_pipeline_stats.csv --sort spill --top 30
  pipeline_stats.py kyty_pipeline_stats.csv --trace replay/stdout.txt --shaders shaders/
  pipeline_stats.py kyty_pipeline_stats.csv --csv ranked.csv
"""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HASH = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{16})(?![0-9a-fA-F])")
# Per kind, the names to look for, best first: NVIDIA, for one, reports no spill count but
# "Local Memory Size" (where spilled registers go) and "Stack Size" (calls), and no instruction
# count but "Binary Size" in bytes.
KINDS = {
    "registers": [re.compile(r"vgpr|register count|registers|regs\b|gprs", re.I)],
    "spills": [re.compile(r"spill", re.I), re.compile(r"local memory|scratch|private memory", re.I),
               re.compile(r"stack", re.I)],
    "instructions": [re.compile(r"instruction", re.I), re.compile(r"code size|isa size|binary size", re.I)],
}


def number(text):
    try:
        value = float(text)
        return int(value) if value.is_integer() else value
    except ValueError:
        return None


def load(path):
    rows = defaultdict(dict)
    meta = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as file:
        for row in csv.DictReader(file):
            key = (row["pipeline"], row["executable"])
            rows[key][row["statistic"]] = number(row["value"])
            meta[key] = (row["stages"], row["subgroup_size"])
    return rows, meta


def pick(names, kind):
    """The statistic of a kind, the vector one first (VGPRs before SGPRs)."""
    for pattern in KINDS[kind]:
        matches = [n for n in names if pattern.search(n)]
        matches.sort(key=lambda n: (0 if "v" in n.lower()[:2] or "vgpr" in n.lower() else 1, len(n)))
        if matches:
            return matches[0]
    return None


def trace_uses(path):
    """Draws and dispatches per set of guest shader hashes, from KYTY_TRACE_SHADERS lines."""
    uses = Counter()
    with open(path, encoding="utf-8", errors="replace") as file:
        for line in file:
            at = line.find("TRACE f=")
            if at < 0:
                continue
            hashes = frozenset(HASH.findall(line[at:]))
            if hashes:
                uses[hashes] += 1
    return uses


def pipeline_hashes(name):
    return frozenset(HASH.findall(name))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stats", type=Path, help="KYTY_PIPELINE_STATS output (CSV)")
    parser.add_argument("--sort", help="sort by the first statistic whose name contains this (case-insensitive)")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--trace", type=Path, help="log of a run with KYTY_TRACE_SHADERS=1")
    parser.add_argument("--shaders", type=Path, help="folder written by --replay-shaders")
    parser.add_argument("--csv", type=Path, help="write every ranked row here")
    args = parser.parse_args()

    rows, meta = load(args.stats)
    if not rows:
        sys.exit(f"{args.stats}: no statistics")
    names = sorted({n for stats in rows.values() for n in stats})
    chosen = {kind: pick(names, kind) for kind in KINDS}
    if args.sort:
        sort_by = next((n for n in names if args.sort.lower() in n.lower()), None)
        if sort_by is None:
            sys.exit(f"no statistic contains {args.sort!r}; there are: {', '.join(names)}")
    else:
        sort_by = chosen["registers"] or chosen["instructions"] or names[0]

    uses = trace_uses(args.trace) if args.trace else None
    files = {}
    if args.shaders:
        for file in args.shaders.glob("*.bin"):
            match = HASH.search(file.name)
            if match:
                files[match.group(1)] = file.name

    ranked = []
    for key, stats in rows.items():
        pipeline, executable = key
        entry = {"pipeline": pipeline, "executable": executable, "stages": meta[key][0],
                 "subgroup_size": meta[key][1]}
        for kind, name in chosen.items():
            entry[kind] = stats.get(name) if name else None
        entry["sort"] = stats.get(sort_by)
        if uses is not None:
            hashes = pipeline_hashes(pipeline)
            entry["uses"] = sum(count for traced, count in uses.items() if hashes and hashes <= traced)
            weight = entry["instructions"] if isinstance(entry["instructions"], (int, float)) else 1
            entry["weight"] = entry["uses"] * weight
        if files:
            entry["files"] = " ".join(files[h] for h in sorted(pipeline_hashes(pipeline)) if h in files)
        ranked.append(entry)
    order = "weight" if uses is not None else "sort"
    ranked.sort(key=lambda e: (e[order] if isinstance(e[order], (int, float)) else -1), reverse=True)

    print(f"{len(rows)} pipeline executables, {len(names)} statistics; sorted by "
          f"{'uses x ' + str(chosen['instructions']) if uses is not None else sort_by}")
    print("statistics found: " + ", ".join(f"{kind}={name}" for kind, name in chosen.items() if name))
    columns = ["registers", "spills", "instructions"] + (["uses"] if uses is not None else [])
    # Headed by the driver's own names: "spills" may be NVIDIA's local memory size in bytes.
    titles = [(chosen.get(c) or c) if c in chosen else c for c in columns]
    header = f"{'#':>3} " + " ".join(f"{t[:12]:>12}" for t in titles) + "  executable / pipeline"
    print(header)
    for index, entry in enumerate(ranked[: args.top], 1):
        cells = " ".join(f"{(entry.get(c) if entry.get(c) is not None else '-')!s:>12}" for c in columns)
        print(f"{index:>3} {cells}  {entry['executable']} / {entry['pipeline']}")
        if entry.get("files"):
            print(f"{'':>4}{entry['files']}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as file:
            fields = ["pipeline", "executable", "stages", "subgroup_size"] + list(KINDS) + \
                     (["uses", "weight"] if uses is not None else []) + (["files"] if files else [])
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ranked)


if __name__ == "__main__":
    main()
