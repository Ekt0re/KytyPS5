# Log triage

`triage.py` reads one emulator log in a single streaming pass and prints a numbered `VERDICT`
followed by one `FINDINGS` section per subject. It uses only the Python 3 standard library and
works on Windows, Linux and macOS: paths are `pathlib`, the log is opened in binary mode and
decoded with `errors='replace'`, CRLF is handled, and nothing is executed in a shell.

    python3 tools/triage/triage.py kyty.log --json triage.json
    python3 tools/triage/triage.py <run-folder> --json triage.json

Passing a folder reads `<folder>/kyty.log`. A loader crash report is picked up from
`<folder>/_Logs/crash_*.txt` (newest by name), or pass `--crash <path>` explicitly. An empty
section says `not in this log`; a bad line is counted as a parse error; one bad section prints
`section failed: ...` and the others still appear. All collectors are bounded, so log size does
not grow memory without limit.

## Sections

- `TIMELINE`: sampled wall time, PerfSummary intervals, flip presentations, intervals with no flips.
- `PERF`: the work counters of `PerfSummary` (flips, draws, dispatches, faults, lookups, …) and the
  exit totals line.
- `SYNC`: the rates of the `SyncStats` reporter (faults, readbacks, uploads, protect and GPU waits).
- `FLIPS`: flip requests/presents, out-of-order and backwards arguments, held queues and waits.
- `TRACES`: shader hashes seen in traced draws and dispatches, grouped by stage.
- `SHADERS`: module compiles and reasons, the compile banner, guest-clock hold, skipped shaders.
- `CACHE`: shader cache / recipes / precompile / Vulkan pipeline cache lines and load totals.
- `DEVICE LOST`: the breadcrumb and device-fault report, in-flight operations and BDA attribution.
- `CRASH`: the loader's guest fault context, registers, modules, guest threads and report path.
- `FATAL`: assertion and fatal blocks; `[RUN-ENDER]` is always last.

The verdict leads with the frame-related numbers, then the worst compile, cache, device-loss,
fault and fatal findings. The same numbers are in `--json`.

## What is parsed

Every regular expression in `triage.py` carries the upstream `file:line` of the format string it
reads: `perfSummary.cpp:28-34,164-226,237,269,289`, `emulatorConfig.cpp:56-72`,
`shaderTrace.h:20-21` with `renderDraw.cpp:1263` and `renderCompute.cpp:726,940`,
`videoOut.cpp:932,1144,1153,1246`, `gpuBreadcrumbs.cpp:58,64,94,119-139,157-168`,
`bdaGuard.cpp:241-245`, `runtimeLinker.cpp:680-686,704-757,813-816`, `pthread.cpp:3079-3096`,
`assert.cpp:16-39`, `shaderCompileStatus.cpp:60-73`, `compilePause.h:93-97`,
`pipelineCache.cpp:385,571,685-687,746-750,1121,1199,1242,1354-1507` and
`pipelineSerialization.cpp:344-514,650-666,1055-1060,1317-1480`. A `@@T <seconds>` marker from a
time-stamped log is accepted but not required; upstream's own `t=` anchors are used otherwise.

## Adding a parser

1. Add a compiled pattern with a comment naming the upstream `file:line` of the format string.
2. Add a bounded collector to `Triage.__init__` and match the line in `Triage._line`.
3. Add the value to `Triage.findings()` and a `def _section(self, f)` returning a list of lines;
   an empty list is printed as `not in this log`.
4. Register `('NAME', self._section)` in `Triage.sections()` and add one synthetic test in
   `tests/test_triage.py` built from the same format string. Tests need no emulator:

       python3 -m unittest discover -s tools/triage/tests

Never attach program files, logs, dumps, stored modules or shader hashes to an issue; quote a
short line or a named number from this output instead.
