# GTA VI and native ray tracing (`--native-rt`)

Status on 28 November 2026: measured and studied, nothing traces natively yet. Title PPSA38500,
01.002.001. The tree is Frozzy's main `af5e20d8` plus the `[RT]` coverage counters
(`graphics: report the guest BVH walks that ray queries leave to the game`).

## Short version

- **No GTA VI walk traces natively.** The tree's ray query pass (`RayQueryTraversal.cpp`) rejects all
  95 BVH walks of the 14 tracing programs a run records. Translated offline with ray queries on and off,
  the 2,396 compute modules `shader_check` emits are byte-identical. Nine of the 14 tracing programs are
  among them; the other five cannot be emitted offline (see Reproduce).
- **The host mirror's TLAS header check passes for GTA VI.** GTA VI's TLASes are `PSR_BVHL` version 10
  headers with properties `0x02020024`, which `Format()` in `gpu_bvh_mirror.comp` accepts. The BLAS
  headers and the instance record layout were not checked against the mirror, so they may be further
  gates.
- **simnJS/KytyPS5-GTA's recognizer would trace one program of the 14.** That is `9d22ec1f658c72a8`
  (reflections), 4 sites. Porting it means porting simnJS's whole native-forms stack, for one program.
- **Native RT cannot reduce the `CS_RaytracingResolve` compile time.** None of the 1,487 resolve
  binaries in `shaders.rpf` contains a BVH instruction.
- **sce::Psr:** 515 kernels, compiler revision `0x16957d7e`, which none of simnJS's tables lists. Its
  four BLAS refit kernels have the same code fingerprints as Marvel's Wolverine's "Base515" kernels.

## The tracing programs

`shaders.rpf` holds 18,272 binaries. 125 distinct hashes contain `IMAGE_BVH_INTERSECT_RAY` or
`IMAGE_BVH64_INTERSECT_RAY`. They come from these effects, each also with a `_bvh8` twin for the Pro:

| Effect | Programs (base / `_bvh8`) |
| --- | --- |
| `rtdirectionalshadows` | 25 / 25 |
| `indirect_diffuse_lighting` | 13 / 7 |
| `sr_tracing` | 12 / 7 |
| `indirect_light_probes` | 8 / 8 |
| `shadowmask` | 7 / 7 |
| `deferred_tiled` (`CS_LaserLightShadowRaytracing`, `CS_ComputeClosestDepth`) | 4 |
| `GpuParticleCollision` | 1 / 1 |

None of the 1,487 `CS_RaytracingResolve` binaries contains a BVH instruction. That family shades the
hits other passes found. Native ray tracing therefore leaves its 3,460 s of cold driver time
(`anyvi-raytracing-resolve-skip.md`) unchanged.

A recorded run (`build\run-anyvi` recipes, 1,212 compute programs) holds 14 of them. Below, "64-bit" is
op `0xe7` (64-bit node pointers) and "32-bit" is op `0xe6`.

| Hash | Program | Op | Walks |
| --- | --- | --- | --- |
| `0e56209edb00fb24` | `indirect_light_probes:CS_GatherAlphaVBuffer_WithFar` | 64-bit | 4 |
| `12d153e63c691415` | `indirect_diffuse_lighting:CS_ScreenProbes_TraceRays_RayBinning_BVH_Mixed_ShaderBinning_WithFar` | 64-bit | 5 |
| `55b2b9b4881a1fe4` | `sr_tracing:CS_RaytraceShadows_Mixed` | 64-bit | 3 |
| `6b1826ae54930a8b` | `indirect_light_probes:CS_IrradianceCache_GatherVBuffer_WithFar` | 64-bit | 8 |
| `95b4ea51e43c1825` | `indirect_light_probes:CS_Primary_GatherVBuffer_WithFar` | 64-bit | 8 |
| `cdc9b2f721b0bd51` | `indirect_light_probes:CS_IrradianceCache_ShadowTrace_WithFar` | 64-bit | 2 |
| `27eed256d4d993c7` | `shadowmask:CS_TraceShadowRays_Checkerboard` | 32-bit | 3 |
| `48abbbc32afc04ba` | `rtdirectionalshadows:CS_RTDistantShadowTraceLeakPlugTemporalUpsample4X32_Tiled` | 32-bit | 9 |
| `757d1ed171798eca` | `rtdirectionalshadows:CS_RTDistantShadowTraceLeakPlug32_Tiled` | 32-bit | 15 |
| `8a9372701a35ce12` | `deferred_tiled:CS_LaserLightShadowRaytracing` | 32-bit | 3 |
| `9acddb6da7cab353` | `rtdirectionalshadows:CS_RTDistantShadowTrace32_Tiled` | 32-bit | 12 |
| `9d22ec1f658c72a8` | `sr_tracing:CS_RaytraceVBuffer_RayBinning_ShaderBinning_Mixed_WithFar` | 32-bit | 14 |
| `aca98a595216007f` | `GpuParticleCollision:CS_CollisionRaytraced` | 32-bit | 3 |
| `b27252f159f856f1` | `rtdirectionalshadows:CS_RTDistantShadowTraceTemporalUpsample4X32_Tiled` | 32-bit | 6 |

All run in waves of 32 except `aca98a59` (64).

## What the walks look like

**64-bit walks** (only `12d153e6` was read; the other five were not):
- The TLAS header address is the qword at `[s[4:5] + 0x10]`. The walk reads the header's `+0x18`
  (properties), `+0x30` (instance array offset) and `+0x70` (instance count). That is the format-10
  layout.
- One loop per walk takes TLAS and BLAS nodes. The node address is a 64-bit add, with carry, of the
  node to a base that was shifted by 3 before the loop. It is not `IAdd32(node, low(header >> 3))`.
- A short per-lane stack lives in LDS and overflows to memory through FLAT.
- **The instance step reads the instance array through a buffer V#**, not by device address:
  - The V# is built before the loop: base = header + `[header + 0x30]`, stride `0x80` (`s33 | 0x00800000`),
    records = `[header + 0x70]`.
  - The instance node's qword gives index `>> 3`, and the flags are tested with `0xda000000`.
  - Four `BUFFER_LOAD_DWORDX4` read the record's first 64 bytes: the 3x4 matrix; at `+48` BLAS `>> 3` in
    54 bits, with flags in the top 10 bits of `+52`; and at `+56` the BLAS root node.
- After the loop, the hit is resolved through FLAT loads:
  - from the record (`index << 7` into the instance array): `+48`, `+112`, `+116`, `+120`;
  - from the BLAS header: `+40` and `+124`, which give the primitive record (BLAS + `[+40]` +
    4 * (primitive - `[+124]`)).

**32-bit walks** (`27eed256` read):
- The header is at `[s[4:5] + 0x30]` or `+0x10`. The TLAS walk is wave-uniform: scalar loads and SCC
  branches.
- Inside it are a BLAS loop and a loop over leaf records through `S_BUFFER_LOAD`, each with its own
  intersection.

**TLASes seen in a run** (`KYTY_RT_HEADER_PROBE=1`, run `simnrt-on1`). Each program's user data table at
`s[4:5]` (`s[2:3]` for `8a937270`) points to several TLASes:

| Program | Table offset → TLAS (instances) |
| --- | --- |
| `0e56209e` | `+0x10` → 1,227; `+0x30` → 3,243 |
| `12d153e6` | `+0x10` → 1,227; `+0x30` → 3,243; `+0x60` → 1 |
| `55b2b9b4` | `+0x10` → 1,227; `+0x40` → 3,243 |
| `6b1826ae`, `95b4ea51` | `+0x20` → 1,227; `+0x40` → 3,243; `+0x60` → 1 |
| `cdc9b2f7` | `+0x30`/`+0x98` → 1,227; `+0x40`/`+0xb8` → 3,243; `+0xd8` → 1 |
| `27eed256` | `+0x30` → 13 |
| `48abbbc3`, `757d1ed1`, `9acddb6d`, `b27252f1` | `+0x10` → 1,227; `+0x30` → 3,243; `+0x50` → 13 |
| `8a937270` | `s[2:3] + 0x10` → 3,243 |
| `9d22ec1f` | `+0x10` → 1,227; `+0x40` → 3,243; `+0x70` → 1 |

Which of them each walk traces was read only for `12d153e6` (`+0x10`) and `27eed256` (`+0x30`).
`aca98a59` did not run in this session.

| Header | Version | Properties | Instances (+0x70) | Instance array (+0x30) |
| --- | --- | --- | --- | --- |
| `0x10c03ae200` | 10 | `0x02020024` | 3,243 | `0x32bc0` |
| `0x10c0fce400` | 10 | `0x02020024` | 1,227 | `0x133c0` |
| `0x1059000000` | 10 | `0x02020024` | 13 | `0x440` |
| `0x10c15daa00` | 10 | `0x02020024` | 1 | `0x180` |

`0x02020024 & 0xa01c0000 == 0`, so the mirror accepts these headers: not Trinity, not BVH8, not
three-level, not depopulated, not GPU-sized. The BLAS headers were not probed.

## Why the tree's recognizer rejects them

These are the first failing checks, as lines of `RayQueryTraversal.cpp` at `af5e20d8`:

| Walks | Matcher | Line | Check |
| --- | --- | --- | --- |
| 64-bit `0e56`, `12d1`, `6b18`, `95b4` (25) | `Match` | 302 | node address is not `IAdd32(node, low(header >> 3))` |
| 64-bit `55b2`, `cdc9` (5) | `MatchLoop` | 208 | no header phi for the lanes or for the closest t |
| all 32-bit (65) | `MatchNested` → `MatchLoop` | 208 (518 for 7 of `9d22`) | as above |
| all 32-bit | `MatchWalk` | 795 / 807 | the loops do not nest as TLAS > BLAS > records, or the vote does not branch as expected |

## What simnJS's recognizer takes

This was run with simnJS's own code: a build of its `shader_cfg_tests` at `73ea67a15`, with a probe
that compiles each program as `CheckNativeWalks` does.

- **`IR::TraceNativeTraversals` (the native forms, before resource tracking).** Only `9d22ec1f658c72a8`
  matches: 7 of its 14 walks, as `ClosestLane`, of which 4 sites are kept. The TLAS header is at
  `[s4 + 0x10]`, root `0x25`, layout `76543210 3dc6ba98 0008018f 0`. That is the same layout simnJS's
  tests expect for GTA V's reflections, `cs 6aeb221de7ff018e`.
- **The other 13 programs match no form.** These are simnJS's reject lines at `73ea67a15`:
  - `MatchPacketWalk` 2770 and `MatchLaneMaskWalk` 2352: the BVH T#'s base is not a 64-bit address
    built from user data. This holds for most walks.
  - 32-bit walks:
    - 2875: the ray is not defined before the walk loop;
    - 1080 / 2927: the wave BLAS;
    - 2388: the first lanes are not a ballot;
    - 272 / 2413: the nested BLAS walk has no exec phi;
    - `8a93` and `aca9`: `MatchEntryWalk` 3025, not the entry template.
  - `MatchFlatWalk` → `Match(tracked = false)` takes only 64-bit walks; the 32-bit ones stop at 338.
    For the 64-bit walks:
    - `0e56`, `12d1`, `55b2` and `cdc9` pass the node-address check through its "header >> 3 in
      phis" path. They then stop at 431 / 435, because the walk does not load the instance root link
      at `+56` exactly once.
    - `6b18` and `95b4` stop at the node-address check itself (401).
    - `cdc9` also stops at 355 (a ray component with no header phi).
- **Not tested: simnJS's post-tracking ray query pass.** The probe gives no user data or flattened
  SRT, which that pass needs to find the TLAS slot, so it was not meaningfully exercised for any of the
  14. Five programs (`0e56`, `6b18`, `95b4`, `aca9`, `9d22`) stop in resource tracking. The other nine
  compile to the end with no ray query, but that does not show the pass would reject them in the game.
- **The nearest simnJS form for the 64-bit walks is its format-10 flat walk.** These walks fail it at
  431/435 because they read the instance record (BLAS `>> 3` at `+48`, root at `+56`) through a buffer
  V# over the instance array, while the flat form expects device-address loads at those offsets.
  Accepting the V# path is an edit to `Match(tracked = false)` and its rewrite, not a new form.

## sce::Psr

- The eboot holds 725 shader footers. 515 of them carry compiler revision `0x16957d7e`, at
  `0xa472020`-`0xa5ddcd0`: Psr's 515 kernels (engine doc §10).
- simnJS knows these revisions: Base460 `0x6585494c`, Base252 `0xeddd574e`, Base515 `0x5371c534` (Marvel's
  Wolverine), and disabled `0xb8263dee`.
- The eboot has the names `psr_count_nested_bvh` and `psr_create_flattened_header`, and lacks
  `psr_rebraiding_copy_srt_instance` and `psr_morton_compute_bounding_box_for_vertices`. simnJS's
  `DetectGeneration()` would therefore call it Base515, if the names were registered. In this tree
  `sceAgcDriverRegisterResource` is a stub and records no names.
- **Fingerprints.** These were computed with simnJS's `CodeFingerprint()` and its own decoder, so the
  values are comparable with its tables. The tool is a standalone build of its decoder.
  - The Base515 refit kernels all match: `copy_dispatch_args` (1 binary), `recover_aabbs` (1),
    `refit_bottom_level` (1), and `fan_refit_async_shader_mesh_0..3` (4 binaries, one shared
    fingerprint).
  - 92 binaries match Base460 entries: 72 "Other", 8 "Trinity", 2 scan descriptor
    initializations, 1 `single_pass_scan_uint32`, 1 compacted-node collapse. 4 match Base252 entries.
  - `IdentifyBase515()` still rejects all of them, on the revision alone.

## What a port would need

1. **The recognizer.** Neither the tree's pass nor simnJS's covers GTA VI's walk shapes, except simnJS
   for one program (`9d22ec1f`).
   - **64-bit walks:** simnJS's format-10 flat form is the nearest. They pass its node-address path
     (except `6b18` and `95b4`), then fail because they read the instance record through a buffer V#
     over the instance array, not by device address. Taking that path is an edit to
     `Match(tracked = false)` and its rewrite. It sits on top of the whole native-forms stack (`f138efcd`
     and the commits before it, `93d1d90c`, `6aa0a34c`).
   - **32-bit walks:** the wave-uniform TLAS > BLAS > records walk needs a form of its own.
2. **The instance record layout of Psr format 10 in GTA VI.**
   - **Read so far (64-bit walk, through the V#):** BLAS `>> 3` at `+48` (54 bits, flags in the top 10 bits
     of `+52`), root at `+56`, and the 3x4 matrix in the first 48 bytes.
   - **Read after the hit:** `+112`, `+116`, `+120`, and the BLAS header's `+40` and `+124`.
   - **Not identified:** which bits are the mask and the facing flags.
   - The mirror reads format-10 records with BLAS `>> 3` at `+48` (`gpu_bvh_mirror.comp`); whether it reads
     the root at `+56` the same way was not checked.
3. **The BLAS headers.** They were not probed. Static BLASes ship prebuilt in `pack/core_bvh4`
   (compressed in the RPF, so they could not be read offline).
4. **Psr build replacement** is not needed to trace, because the mirror reads the BVH the guest built.
   simnJS's TLAS and BLAS replacements (`4fab383e`, `54aab65a`, `0f9b3d09`) would need GTA VI's
   revision accepted and its user data layouts. Only the refit kernels are known to match.

## The `--native-rt` run

Run `simnrt-on1` used a cold `build\run-simn-rt` and this tree with `--native-rt` and the probe.
- The guest clock was held about 1,100 s for compiles, and the run never reached the world.
- At 1,193 s the game's own watchdog fired: `mov dword [0]` at `0x908bd3a40`, a 30 s main-loop stall
  (crash report `run-simn-rt\_Logs\crash_20261128_114047.txt`).
- There was no device loss and no refused indirect dispatch.
- In the same hour, three other cold runs on the same base without the flag (`simncpu-cold1`,
  `simnc-off1`, `sp-pop`) also did not reach the world within 1,200 s, but did not crash.
- With one run against three, the watchdog cannot be laid on `--native-rt`.

## Reproduce

- **Offline A/B.** Use `shader_check` with `SHADER_CHECK_RAY_QUERY=0` or `1` and `SHADER_CHECK_SPIRV`,
  over capsules from a `.recipes` (v1) file. This tree's `shader_check` cannot specialize images from
  anyvi's `.recipes2` variants. The v1 capsules of `48ab`, `757d`, `9acd`, `9d22` and `b272` exit 2,
  because the extractor misreads them.
- **Evidence and tools:** `build\research\simn-rt-gta6`. It holds the probe outputs, the simnJS study
  patch, the fingerprint tool and list, the shaders.rpf scan, and the dumps of `12d153e6` and
  `27eed256`.
- **Run.** Pass `--native-rt` and `KYTY_RT_HEADER_PROBE=1`, then read the `[RT]` lines in stdout.
