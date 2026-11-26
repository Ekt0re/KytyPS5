"""Write shader capsules (.cap, the KYTY_SHADER_CAPSULES format that shader_check reads) from
the flo/kyty recipes file (_PipelineCache/<title>.recipes), so one shader can be translated
offline without running the game.

Usage: python tools/shader/recipe_capsule.py <recipes file> <out dir> <hex hash> [<hex hash> ...]
       python tools/shader/recipe_capsule.py <recipes file> <out dir> --stage <n>
       python tools/shader/recipe_capsule.py <recipes file> --list
--stage writes every shader of one stage (ShaderType: 1 VS, 2 PS, 3 CS, 4 MS). Each capsule gets
its variant record next to it (<capsule>.variant) for shader_check's resource specialization; use
a recipes file written by the same build as shader_check, since the record layout follows the
serializer.

Record encoding (pipelineSerialization.cpp, ir/Serialize.h): a u64 magic, then frames of
{u32 compressed, u32 raw, u64 xxh3} + zstd. Scalars are raw little-endian; vectors are a u32
count and their elements. Code: u8 0, i32 stage, u64 hash, vec<u32> code, vec<u32> back code.
Variant: u8 1, i32 stage, u64 hash, u32 user_data_count, vec<u32> static_state, u32 wave_size,
u32 user_data_base, vec<u32> user_data, u32 window begin, u32 window end, specialization,
u64 module id, vec<u8> input (the stage input info struct, raw).
"""
import os
import struct
import sys
from compression import zstd

MAGIC = 0x3150435259544B59
STAGES = {0: "Unknown"}


def frames(data):
    (magic,) = struct.unpack_from("<Q", data, 0)
    if magic != MAGIC:
        sys.exit("not a recipes file")
    at = 8
    while at + 16 <= len(data):
        csize, rsize, _ = struct.unpack_from("<IIQ", data, at)
        if at + 16 + csize > len(data):
            break
        raw = zstd.decompress(data[at + 16 : at + 16 + csize])
        if len(raw) != rsize:
            break
        yield raw
        at += 16 + csize


class R:
    def __init__(self, b):
        self.b, self.at = b, 0

    def u(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.at)
        self.at += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def vec32(self):
        n = self.u("I")
        v = list(struct.unpack_from("<%dI" % n, self.b, self.at))
        self.at += 4 * n
        return v


def parse(path):
    codes, variants = {}, {}
    for raw in frames(open(path, "rb").read()):
        r = R(raw)
        kind = r.u("B")
        if kind == 0:
            stage, h = r.u("i"), r.u("Q")
            codes[(stage, h)] = (r.vec32(), r.vec32())
        elif kind == 1:
            stage, h, udc = r.u("i"), r.u("Q"), r.u("I")
            r.vec32()  # static_state
            wave, base = r.u("I"), r.u("I")
            user = r.vec32()
            # The input is the record's last field: u32 count, then the bytes.
            for n in range(len(raw) - 4, 0, -1):
                (count,) = struct.unpack_from("<I", raw, len(raw) - 4 - n) if n >= 0 else (None,)
                if count == n:
                    inp = raw[len(raw) - n :]
                    break
            else:
                continue
            variants.setdefault((stage, h), []).append((wave, base, udc, user, inp, raw))
    return codes, variants


def capsule(stage, h, code, back, wave, base, udc, user, inp):
    header = [0x5041434B, stage, wave, base, udc, len(inp), len(code), len(back),
              h & 0xFFFFFFFF, h >> 32]
    user = (user + [0] * udc)[:udc]
    pad = (-len(inp)) % 4
    return (struct.pack("<10I", *header) + inp + b"\0" * pad +
            struct.pack("<%dI" % udc, *user) + struct.pack("<%dI" % len(code), *code) +
            struct.pack("<%dI" % len(back), *back))


def main():
    codes, variants = parse(sys.argv[1])
    if sys.argv[2] == "--list":
        for (stage, h), (code, _) in sorted(codes.items(), key=lambda kv: -len(kv[1][0])):
            print("stage %d hash %016x code %d words, %d variant(s)" %
                  (stage, h, len(code), len(variants.get((stage, h), []))))
        return
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    targets = sys.argv[3:]
    if targets[:1] == ["--stage"]:
        # Every shader of one stage (e.g. --stage 3 for compute).
        targets = ["%016x" % hh for (st, hh) in codes if st == int(targets[1])]
    for text in targets:
        h = int(text, 16)
        found = [(s, hh) for (s, hh) in codes if hh == h]
        if not found:
            print("%016x: no code record" % h)
            continue
        for key in found:
            code, back = codes[key]
            for i, (wave, base, udc, user, inp, raw) in enumerate(variants.get(key, [])):
                name = os.path.join(out, "%d_%016x_%d.cap" % (key[0], h, i))
                open(name, "wb").write(capsule(key[0], h, code, back, wave, base, udc, user, inp))
                # The raw variant record: shader_check reads its resource specialization.
                open(name + ".variant", "wb").write(raw)
                print("%s: stage %d wave %d, %d code words, %d user data, input %d bytes" %
                      (name, key[0], wave, len(code), udc, len(inp)))


if __name__ == "__main__":
    main()
