# Frame capture, replay and performance tools

A frame capture records a few frames of a running game: every committed guest memory range, then
each GPU call of the captured frames with the memory the CPU changed before it. A replay runs
those frames again without the game, as often as needed. That makes a GPU change measurable in
seconds on a fixed workload, and checkable against the picture the game itself showed.

The format and the replay follow AnyPS5's frame capture (simnJS/AnyPS5, GPL-2.0), re-implemented
for Kyty's kernel and GPU front end (`src/graphics/frameCapture.cpp`).

## Capture

Run the game with `KYTY_FRAME_CAPTURE=<dir>`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `KYTY_FRAME_CAPTURE` | | Folder to write the capture to. |
| `KYTY_FRAME_CAPTURE_AT_FLIP` | 1 | Flips to let pass before the capture starts. |
| `KYTY_FRAME_CAPTURE_AFTER_SECONDS` | 0 | Seconds to wait before the capture starts. |
| `KYTY_FRAME_CAPTURE_FRAMES` | 3 | Flips to record. |
| `KYTY_FRAME_CAPTURE_REFERENCE` | 1 | `0` skips the reference image. |
| `KYTY_FRAME_CAPTURE_SOFT_DIRTY` | 1 | Linux: find changed pages with soft-dirty bits instead of hashing every page. |
| `KYTY_FRAME_CAPTURE_FLUSH`, `KYTY_FRAME_CAPTURE_BOUNDARY_CALLS` | | See `frameCapture.cpp`. |

The folder gets `ranges.bin`, `snapshot.bin`, `calls.bin`, `header.json`, a copy of
`param.json`, and `reference.png` / `reference.rgba`: the display image the game showed when the
capture ended.

## Replay

```
kyty_emulator --replay <capture> [--replay-out <dir>] [--replay-repeat <n>]
```

The replay writes `replay.json`, `frame.png` / `frame.rgba` and, when the capture has a reference
image, `diff.png` (differing pixels in red) to `--replay-out` (default: the capture folder).

`replay.json` has, among others:

- `passes[]`: each pass with its flip times and `frames` (median, mean, p95, min, max, stdev of
  the flip-to-flip intervals).
- `steady`: the same statistics over passes 2..n, so shader compilation and first uploads in
  pass 1 are left out. Use `--replay-repeat 4` or more when comparing builds. With one pass,
  `steady` uses that pass.
- `image`: the replay's frame against the game's reference (`identical` share of pixels,
  `max_diff`, `mean_diff`, `psnr_db`, which is `null` when the images are identical).

`KYTY_REPLAY_DEBUG`, `KYTY_REPLAY_WATCH=<addr>`, `KYTY_REPLAY_MEM_LOG`, `KYTY_REPLAY_WAIT_MS` and
`KYTY_REPLAY_SHMEM_GUARD_MB` help when a replay goes wrong; see `frameCapture.cpp`.

## Shaders of a capture

```
kyty_emulator --replay <capture> --replay-shaders <dir>
```

writes every guest shader of the capture as `type<N>-<hash>.bin` (once per hash) and a
`shaders.json` line per shader (address, type, size, hash), then exits without replaying. The hash
is the one Kyty uses everywhere else (shader store, `KYTY_TRACE_SHADERS`, pipeline names), so a
dumped file can be looked up in a trace or in pipeline statistics, and fed to
`tools/shader` or the recompiler on its own.

## Pipeline statistics from the driver

`KYTY_PIPELINE_STATS=1` (or `=<file.csv>`) asks the Vulkan driver, through
`VK_KHR_pipeline_executable_properties`, for its statistics of every pipeline Kyty creates:
registers, spills, instruction counts, whatever the driver reports. They are written to
`kyty_pipeline_stats.csv` (`pipeline,executable,stages,subgroup_size,statistic,value`). The
pipeline name holds the guest shader hashes (`VS_<hash> PS_<hash>`).

AMD, NVIDIA, Intel and Mesa drivers report them, each with their own names. When the device does
not, the log says so and nothing else changes. Pipeline binaries are turned off while statistics
are recorded, because a pipeline loaded from a stored binary has no compile statistics.

Rank them with

```
python3 tools/replay/pipeline_stats.py kyty_pipeline_stats.csv [--trace kyty.log] [--sort spills]
```

With `--trace` (a log taken with `KYTY_TRACE_SHADERS`) pipelines are weighted by how many draws
and dispatches used them, so the list starts with the shaders where a fix pays off most.

## Comparing two builds or settings

```
python3 tools/replay/replay_ab.py run --capture <capture> --a old/kyty_emulator --b new/kyty_emulator
python3 tools/replay/replay_ab.py run --capture <capture> --a kyty_emulator --b kyty_emulator --b-env KYTY_X=1
python3 tools/replay/replay_ab.py compare <dir-a> <dir-b>
```

`run` replays the capture with A and B in turns (ABAB, then BABA), each run with
`--replay-repeat`, and reports the median steady frame time of each side, the difference with a
95% bootstrap interval and a rank-sum test, and a verdict. A verdict needs at least 5 runs of
each side (the default `--rounds 5`), since runs of the same build alone differ by a few percent.
A build that is faster but changes the picture is flagged.
`--json` writes the summary for a PR.

Wall times follow GPU clocks and whatever else the machine runs. Close other programs, and when
the difference is small, lock the GPU clocks if the driver allows it.

## Smoke test without a game

```
python3 tools/replay/make_test_capture.py out/capture
kyty_emulator --replay out/capture --replay-repeat 3
```

writes a tiny synthetic capture (one solid-colour display buffer and its reference), so the
replay should report 100% identical pixels. `--shader` adds one shader for `--replay-shaders`.
Needs the `zstd` command line tool or the `zstandard` Python module.
