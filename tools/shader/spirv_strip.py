"""Remove one construct from a SPIR-V module to measure what the driver charges for it.

Usage: python tools/shader/spirv_strip.py selects <in.spv> <out.spv>
       python tools/shader/spirv_strip.py flush <in.spv> <out.spv>
  selects  every OpSelect is replaced by its true value
  flush    every f32 denormal flush (FAbs, FOrdLessThan against the smallest normal, FMul by
           zero, OpSelect) is replaced by the value it guarded
The output computes wrong values; it is only for timing. Cold-compile the input and the output
with spvbench: the difference is the cost of the construct on that driver. Needs spirv-dis,
spirv-as and spirv-val on PATH or in SPIRV_TOOLS_DIR; the output must pass spirv-val.
"""
import os
import re
import shutil
import subprocess
import sys

# %id = OpX [%type] rest
DEF = re.compile(r"^\s*%(\S+) = (\w+)(?: (%\S+))?(.*)$")
USE = re.compile(r"%([A-Za-z0-9_]+)")


def tool(name):
    folder = os.environ.get("SPIRV_TOOLS_DIR")
    path = os.path.join(folder, name) if folder else shutil.which(name)
    if not path or not (os.path.exists(path) or os.path.exists(path + ".exe")):
        sys.exit("%s not found: put it on PATH or set SPIRV_TOOLS_DIR" % name)
    return path


def parse(lines):
    defs = {}
    for line in lines:
        m = DEF.match(line)
        if m:
            defs[m.group(1)] = (m.group(2), m.group(3), [o.lstrip("%") for o in m.group(4).split()])
    return defs


def strip_selects(defs):
    """Select result -> its true value."""
    return {ident: ops[1] for ident, (op, _, ops) in defs.items() if op == "OpSelect" and len(ops) == 3}, set()


def strip_flush(defs):
    """Flush result -> the guarded value, and the ids whose definitions become dead."""
    f32 = {"%" + ident for ident, (op, _, ops) in defs.items() if op == "OpTypeFloat" and ops[:1] == ["32"]}
    floats = {ident: ops[0] for ident, (op, ty, ops) in defs.items() if op == "OpConstant" and ty in f32 and ops}
    # The smallest normal (0x00800000) prints as 1.17549435e-38.
    threshold = {i for i, v in floats.items() if v.startswith("1.1754943")}
    zero = {i for i, v in floats.items() if v in ("0", "0.0", "-0", "-0.0")}
    replace, dead = {}, set()
    for ident, (op, _, ops) in defs.items():
        if op != "OpSelect" or len(ops) != 3:
            continue
        cond, truth, lie = ops
        mul, cmp_ = defs.get(truth), defs.get(cond)
        if mul is None or cmp_ is None or mul[0] != "OpFMul" or cmp_[0] != "OpFOrdLessThan":
            continue
        if len(mul[2]) != 2 or mul[2][1] not in zero or mul[2][0] != lie:
            continue
        if len(cmp_[2]) != 2 or cmp_[2][1] not in threshold:
            continue
        magnitude = defs.get(cmp_[2][0])
        if magnitude is None or magnitude[0] != "OpExtInst" or "FAbs" not in magnitude[2] or magnitude[2][-1] != lie:
            continue
        replace[ident] = lie
        dead.update((truth, cond, cmp_[2][0]))
    return replace, dead


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("selects", "flush"):
        sys.exit(__doc__)
    mode, src, out = sys.argv[1:]
    dis, asm, val = tool("spirv-dis"), tool("spirv-as"), tool("spirv-val")
    text = subprocess.run([dis, "--raw-id", src], capture_output=True, text=True, check=True).stdout
    lines = text.splitlines()
    defs = parse(lines)
    replace, dead = (strip_selects if mode == "selects" else strip_flush)(defs)

    def resolve(ident):
        # A replaced value can itself be a replaced select.
        while ident in replace:
            ident = replace[ident]
        return ident

    kept = []
    for line in lines:
        m = DEF.match(line)
        if m and m.group(1) in replace:
            continue
        kept.append(USE.sub(lambda u: "%" + resolve(u.group(1)), line))
    # Drop the helper definitions nothing reads any more; one removal can free the next.
    removed = set()
    while True:
        used = set()
        for line in kept:
            m = DEF.match(line)
            if m and m.group(1) in removed:
                continue
            used.update(USE.findall(line.split("=", 1)[1] if m else line))
        newly = {d for d in dead - removed if d not in used}
        if not newly:
            break
        removed |= newly
    kept = [l for l in kept if not ((m := DEF.match(l)) and m.group(1) in removed)]
    asm_path = out + ".spvasm"
    with open(asm_path, "w") as f:
        f.write("\n".join(kept) + "\n")
    subprocess.run([asm, "--preserve-numeric-ids", asm_path, "-o", out], check=True)
    subprocess.run([val, out], check=True)
    print("%s: replaced %d %s" % (out, len(replace), "selects" if mode == "selects" else "flushes"))


if __name__ == "__main__":
    main()
