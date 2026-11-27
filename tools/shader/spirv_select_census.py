"""Census of the OpSelect trees in SPIR-V modules: which shapes the selects form.

Usage: python tools/shader/spirv_select_census.py <module.spv | folder> ... [--top N]
Needs spirv-dis on PATH or in SPIRV_TOOLS_DIR. Per module, one line:
  selects       OpSelect count
  const_leaf    selects whose whole subtree ends in constants (a table lookup could replace them)
  trees         select trees (selects that no other select uses as a value)
  largest       selects in the largest tree
  depth         longest chain of selects nested through their values
  nest_t/nest_f selects whose true / false value is another select
  same_cond     nestings where the inner select has the outer one's condition
  keys          distinct condition ids plus distinct compared operands, over all trees
Then the most common (outer condition, inner condition) opcode pairs over all nestings.
Definition order in a disassembly puts each select's operands first, so every pass is linear.
"""
import collections
import os
import re
import shutil
import subprocess
import sys

DEF = re.compile(r"^\s*%(\S+) = (\w+)(.*)$")
CONSTANTS = {"OpConstant", "OpConstantNull", "OpConstantTrue", "OpConstantFalse", "OpConstantComposite"}


def tool(name):
    folder = os.environ.get("SPIRV_TOOLS_DIR")
    path = os.path.join(folder, name) if folder else shutil.which(name)
    if not path or not (os.path.exists(path) or os.path.exists(path + ".exe")):
        sys.exit("%s not found: put it on PATH or set SPIRV_TOOLS_DIR" % name)
    return path


def modules(args):
    for arg in args:
        if os.path.isdir(arg):
            for name in sorted(os.listdir(arg)):
                if name.endswith(".spv"):
                    yield os.path.join(arg, name)
        else:
            yield arg


def census(dis, path, pairs):
    proc = subprocess.run([dis, path], capture_output=True, text=True)
    if proc.returncode != 0:
        return None, proc.stderr.strip().splitlines()[-1:] or ["?"]
    opcode, operands, order = {}, {}, []
    for line in proc.stdout.splitlines():
        m = DEF.match(line)
        if m:
            ident = m.group(1)
            opcode[ident] = m.group(2)
            operands[ident] = [o.lstrip("%") for o in m.group(3).split()]
            order.append(ident)
    # (condition, true value, false value) per select.
    selects = {}
    for ident in order:
        ops = operands[ident]
        if opcode[ident] == "OpSelect" and len(ops) == 4:
            selects[ident] = (ops[1], ops[2], ops[3])

    const_leaf, depth = {}, {}
    nest_t = nest_f = same = 0
    used = set()
    for ident in order:
        entry = selects.get(ident)
        if entry is None:
            continue
        cond, truth, lie = entry
        leaf = True
        d = 0
        for side, value in (("t", truth), ("f", lie)):
            inner = selects.get(value)
            if inner is not None:
                used.add(value)
                leaf = leaf and const_leaf[value]
                d = max(d, depth[value])
                if side == "t":
                    nest_t += 1
                else:
                    nest_f += 1
                same += inner[0] == cond
                pairs[(opcode.get(cond, "?"), opcode.get(inner[0], "?"))] += 1
            else:
                leaf = leaf and opcode.get(value) in CONSTANTS
        const_leaf[ident] = leaf
        depth[ident] = d + 1

    # Trees: walk down from each root; subtrees can be shared, so a visited set keeps it linear.
    largest = trees = 0
    keys = set()
    visited = set()
    for root in selects:
        if root in used or root in visited:
            continue
        trees += 1
        size = 0
        stack = [root]
        visited.add(root)
        while stack:
            node = stack.pop()
            size += 1
            cond, truth, lie = selects[node]
            keys.add(cond)
            keys.update(o for o in operands.get(cond, [])[1:] if o in opcode)
            for value in (truth, lie):
                if value in selects and value not in visited:
                    visited.add(value)
                    stack.append(value)
        largest = max(largest, size)
    return {
        "selects": len(selects),
        "const_leaf": sum(const_leaf.values()),
        "trees": trees,
        "largest": largest,
        "depth": max(depth.values(), default=0),
        "nest_t": nest_t,
        "nest_f": nest_f,
        "same_cond": same,
        "keys": len(keys),
    }, None


def main():
    args = sys.argv[1:]
    top = 0
    if "--top" in args:
        at = args.index("--top")
        top = int(args[at + 1])
        del args[at : at + 2]
    if not args:
        sys.exit(__doc__)
    dis = tool("spirv-dis")
    pairs = collections.Counter()
    rows = []
    for path in modules(args):
        row, error = census(dis, path, pairs)
        if row is None:
            print("FAIL %s: %s" % (path, error[0]))
            continue
        rows.append((os.path.basename(path), row))
    rows.sort(key=lambda r: -r[1]["selects"])
    columns = ["selects", "const_leaf", "trees", "largest", "depth", "nest_t", "nest_f", "same_cond", "keys"]
    print("%-44s " % "module" + " ".join("%10s" % c for c in columns))
    for name, row in rows[: top or None]:
        print("%-44s " % name[:44] + " ".join("%10d" % row[c] for c in columns))
    total = {c: sum(r[c] for _, r in rows) for c in ("selects", "const_leaf", "nest_t", "nest_f", "same_cond")}
    print("modules: %d, " % len(rows) + ", ".join("%s %d" % kv for kv in total.items()))
    print("nesting condition pairs (outer -> inner):")
    for (outer, inner), count in pairs.most_common(10):
        print("  %-24s -> %-24s %8d" % (outer, inner, count))


if __name__ == "__main__":
    main()
