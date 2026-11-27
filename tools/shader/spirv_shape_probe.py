"""Write two compute modules that pick one of N constants by a lane-derived index: one as a
ladder of N OpSelects, one as a single indexed load from a storage buffer. Cold-compiling both
with spvbench shows what the driver charges for the select shape against the memory shape.

Usage: python tools/shader/spirv_shape_probe.py <out dir> [N]   (N defaults to 128)
Writes <out dir>/shape_ladder_<N>.spv and shape_table_<N>.spv. Needs spirv-as and spirv-val on
PATH or in SPIRV_TOOLS_DIR.
"""
import os
import shutil
import subprocess
import sys


def tool(name):
    folder = os.environ.get("SPIRV_TOOLS_DIR")
    path = os.path.join(folder, name) if folder else shutil.which(name)
    if not path or not (os.path.exists(path) or os.path.exists(path + ".exe")):
        sys.exit("%s not found: put it on PATH or set SPIRV_TOOLS_DIR" % name)
    return path


def header(n):
    lines = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        'OpEntryPoint GLCompute %main "main" %gid %in %table %out',
        "OpExecutionMode %main LocalSize 64 1 1",
        "OpDecorate %gid BuiltIn GlobalInvocationId",
    ]
    for binding, var in enumerate(("in", "table", "out")):
        lines += ["OpDecorate %%%s DescriptorSet 0" % var, "OpDecorate %%%s Binding %d" % (var, binding)]
    lines += [
        "OpDecorate %rta ArrayStride 4",
        "OpMemberDecorate %Block 0 Offset 0",
        "OpMemberDecorate %Array 0 Offset 0",
        "OpDecorate %Block Block",
        "OpDecorate %Array Block",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%bool = OpTypeBool",
        "%uint = OpTypeInt 32 0",
        "%float = OpTypeFloat 32",
        "%v3uint = OpTypeVector %uint 3",
        "%rta = OpTypeRuntimeArray %uint",
        "%Block = OpTypeStruct %uint",
        "%Array = OpTypeStruct %rta",
        "%p_in_v3 = OpTypePointer Input %v3uint",
        "%p_block = OpTypePointer StorageBuffer %Block",
        "%p_array = OpTypePointer StorageBuffer %Array",
        "%p_uint = OpTypePointer StorageBuffer %uint",
        "%gid = OpVariable %p_in_v3 Input",
        "%in = OpVariable %p_block StorageBuffer",
        "%table = OpVariable %p_array StorageBuffer",
        "%out = OpVariable %p_block StorageBuffer",
    ]
    lines += ["%%u%d = OpConstant %%uint %d" % (i, i) for i in range(n + 1)]
    lines += ["%%f%d = OpConstant %%float %d.0" % (i, i) for i in range(1, n + 1)]
    lines += [
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "%id = OpLoad %v3uint %gid",
        "%lane = OpCompositeExtract %uint %id 0",
        "%seed_ptr = OpAccessChain %p_uint %in %u0",
        "%seed = OpLoad %uint %seed_ptr",
        "%key = OpBitwiseAnd %uint %lane %seed",
        "%out_ptr = OpAccessChain %p_uint %out %u0",
    ]
    return lines


def ladder(n):
    lines = header(n)
    lines += ["%%k%d = OpULessThan %%bool %%key %%u%d" % (i, i) for i in range(1, n + 1)]
    for i in range(n, 0, -1):
        tail = "%%f%d" % n if i == n else "%%r%d" % (i + 1)
        lines.append("%%r%d = OpSelect %%float %%k%d %%f%d %s" % (i, i, i, tail))
    lines += ["%bits = OpBitcast %uint %r1", "OpStore %out_ptr %bits"]
    return lines + ["OpReturn", "OpFunctionEnd"]


def table(n):
    lines = header(n)
    lines += [
        "%%in_range = OpULessThan %%bool %%key %%u%d" % n,
        "%slot = OpSelect %uint %in_range %key %u0",
        "%slot_ptr = OpAccessChain %p_uint %table %u0 %slot",
        "%value = OpLoad %uint %slot_ptr",
        "OpStore %out_ptr %value",
    ]
    return lines + ["OpReturn", "OpFunctionEnd"]


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) == 3 else 128
    if n < 1:
        sys.exit("N must be at least 1")
    asm, val = tool("spirv-as"), tool("spirv-val")
    os.makedirs(out, exist_ok=True)
    for name, build in (("ladder", ladder), ("table", table)):
        base = os.path.join(out, "shape_%s_%d" % (name, n))
        with open(base + ".spvasm", "w") as f:
            f.write("\n".join(build(n)) + "\n")
        subprocess.run([asm, "--target-env", "vulkan1.3", base + ".spvasm", "-o", base + ".spv"], check=True)
        subprocess.run([val, "--target-env", "vulkan1.3", base + ".spv"], check=True)
        print("%s.spv ok" % base)


if __name__ == "__main__":
    main()
