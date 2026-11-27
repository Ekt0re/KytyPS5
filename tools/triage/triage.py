#!/usr/bin/env python3
"""Triage one emulator log in a single streaming pass.

Usage:
    python3 tools/triage/triage.py <log> [--json out.json] [--crash path] [--run-dir dir]

Reads an upstream log (or a run folder containing kyty.log), optionally reads a crash report
written by the loader, and prints a numbered VERDICT followed by one FINDINGS section per subject.
Only the Python 3 standard library is used. The log is opened in binary mode and decoded with
errors='replace', so CRLF logs and stray bytes cannot stop the pass. Every collector is bounded.

Each parser below cites the upstream file and line of the format string it reads. Line types that
upstream does not print yet are kept in isolated sections: an absent type means "not in this log"
in that section and does not affect the others.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

# --- format strings of the log we parse -------------------------------------------------------
# PerfSummary: names src/common/perfSummary.cpp:28-34, line src/common/perfSummary.cpp:164-226,
# periodic tag and exit tag src/common/perfSummary.cpp:237,269, GPU thread line :289.
PERF_RE = re.compile(
    r'^PerfSummary\s+(?P<tag>(?:total\s+)?t=\d+)\s+secs=(?P<secs>[0-9.]+)\s*(?P<body>.*)$'
)
PERF_GPU_RE = re.compile(r'^PerfSummary: GPU thread tid (\d+), instruction counters: user (\S+), kernel (.+)$')
PERF_COUNTERS = (
    'flips', 'draws', 'dispatches', 'indirect_dispatches', 'write_faults', 'read_faults',
    'buf_readbacks', 'tex_readbacks', 'tex_lookups', 'buf_lookups', 'desc_writes',
    'heap_desc_writes', 'pipe_lookups', 'pipe_created', 'amm_maps', 'amm_unmaps', 'gpu_unmaps',
    'heap_rebinds', 'heap_lookups', 'heap_range_skips', 'depth_recreates',
)
# SyncStats: src/common/emulatorConfig.cpp:56-72.
SYNC_RE = re.compile(r'^SyncStats\s+(?P<tag>t=\d+)\s+(?P<body>.*)$')
# TRACE f=.. n=.. D/C: shaderTrace.h:20-21, renderDraw.cpp:1263-1268, renderCompute.cpp:726-730,940-944.
TRACE_RE = re.compile(r'^TRACE f=(\d+) n=(\d+) ([DC])\s+(.*)$')
TRACE_HASH_RE = re.compile(r'\b([a-z]{2})=0x([0-9a-fA-F]+)')
# FLIPTRACE: videoOut.cpp:932-936 (req), :1144-1150 (held), :1153 (wait), :1246-1256 (present).
FLIP_RE = re.compile(r'^FLIPTRACE (req|present|held|wait)\s+(.*)$')
# GPU breadcrumbs / device-lost report: gpuBreadcrumbs.cpp:58,64,94,119-139,157-168 and
# bdaGuard.cpp:241-245 (the [bda-guard] lines appended to the device-fault report).
DEVICE_PREFIXES = (
    'GPU device lost (', 'GPU breadcrumbs:', '  in flight:', '  last completed:',
    'Device fault:', '  address type ', '  vendor ', '  [bda-guard]',
)
BREADCRUMB_RE = re.compile(r'^GPU breadcrumbs: recorded (\d+), started (\d+), completed (\d+)$')
# Crash report: runtimeLinker.cpp:704-757, guest threads pthread.cpp:3079-3096, crash file :750.
CRASH_START = '--- Guest fault context ---'
CRASH_FAULT_RE = re.compile(
    r'^type=(\d+) code=(\d+) pc=0x([0-9a-fA-F]+) access=(\d+) address=0x([0-9a-fA-F]+)$'
)
CRASH_THREAD_RE = re.compile(r'^  \[(\d+)\]\s+(.*?)\s+host=(\d+)(.*)$')
CRASH_MODULE_RE = re.compile(r'^\s*\*\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s+(.+)$')
CRASH_MODULE_ANY_RE = re.compile(r'^\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s+(.+)$')
CRASH_FILE_RE = re.compile(r'^crash report: (.+)$')
# Fatal blocks: assert.cpp:16-39, runtimeLinker.cpp:680-686,813-816,873, pipelineCache.cpp:723.
FATAL_PREFIXES = (
    '--- Build ---', '--- Error ---', '--- Fatal Error ---', 'Error: condition (',
    'Not implemented (', 'Unhandled host exception:', 'exit!!!',
    '!!! KYTY_RAGE_ASSERT_BYPASS:', CRASH_START,
)
FATAL_RESOURCE_RE = re.compile(r'^shader 0x[0-9a-fA-F]+ \([^)]*\): resources could not be materialized$')
# Shader compile / store: shaderCompileStatus.cpp:60-64,70-73, compilePause.h:93-97,
# pipelineCache.cpp:385,571,685-687,746-750,1121,1199,1242 and pipelineSerialization.cpp:
# 344-347,411-416,425-428,482-485,507-514,650-654,664-666,1055-1060,1317-1321,1355-1379,
# 1415-1425,1438-1480.
COMPILING_RE = re.compile(r'^\[shaders\] compiling: (\d+) queued, (\d+) active, (\d+) done$')
BURST_RE = re.compile(r'^\[shaders\] burst done: (\d+) shaders in ([0-9.]+) s$')
CLOCK_HELD_RE = re.compile(r'^Guest clock: held ([0-9.]+) s in total while GPU queues waited for compiles$')
SHADER_COMPILE_RE = re.compile(
    r'^Shader compile: (.+?) \(translate ([0-9.]+) ms( ahead)?, SPIR-V ([0-9.]+) ms\)$'
)
SHADER_COUNT_RE = re.compile(
    r'^Shaders: VS (\d+) \| PS (\d+) \| CS (\d+) \| GS (\d+) \| LS (\d+) \| HS (\d+) \| TES (\d+)$'
)
SKIP_SHADER_RE = re.compile(r'^KYTY_SKIP shader=0x([0-9a-fA-F]+) stage=(\w+)$')
TEMP_SKIP_RE = re.compile(
    r'^!!! TEMPORARY COMPUTE DISPATCH SKIP !!! shader=0x([0-9a-fA-F]+) '
    r'pc=0x([0-9a-fA-F]+) reason=(.*?); rendering is incomplete$'
)
AHEAD_FAILED_RE = re.compile(r'^Shader .*: translation ahead failed; .*$')
CACHE_PREFIXES = ('Shader cache:', 'Shader recipes:', 'Shader precompile:', 'Shader restamp:',
                  'Vulkan pipeline cache:')
CACHE_LOADED_RE = re.compile(r'^Shader cache: loaded (\d+) programs, (\d+) modules, (\d+) pipelines in ([0-9.]+) ms')
# Optional time marker of a time-stamped log wrapper. Not emitted by upstream; isolated so a log
# without it still triages. Upstream's own anchors are the PerfSummary/SyncStats t= fields above.
AT_RE = re.compile(r'^@@T ([0-9.]+)$')
ANSI_RE = re.compile(r'^\x1b\[[0-9;]*m')
KV_RE = re.compile(r'([\w/.-]+)=(\S+)')

MAX_PERF = 2000
MAX_SYNC = 500
MAX_IDS = 2000
MAX_LINES = 300


def kv(text: str) -> dict[str, str]:
    """Parse the space-separated key=value fields of an upstream report line."""
    return {m.group(1): m.group(2) for m in KV_RE.finditer(text)}


def add_id(collection: set[str], value: str) -> None:
    if value not in collection and len(collection) < MAX_IDS:
        collection.add(value)


class Triage:
    def __init__(self) -> None:
        self.lines = 0
        self.t = 0.0
        self.first_t: float | None = None
        self.parse_errors = 0
        self.perf: collections.deque[dict] = collections.deque(maxlen=MAX_PERF)
        self.perf_lines = 0
        self.perf_totals: dict[str, str] = {}
        self.sync: collections.deque[dict] = collections.deque(maxlen=MAX_SYNC)
        self.traced_frames: set[str] = set()
        self.draws = 0
        self.dispatches = 0
        self.indirect_gpu = 0
        self.trace_ids: dict[str, set[str]] = collections.defaultdict(set)
        self.flip_present = 0
        self.flip_ooo = 0
        self.flip_argback = 0
        self.flip_held = 0
        self.longest_hold_ms = 0.0
        self.longest_wait_ms = 0.0
        self.flip_recent: collections.deque[str] = collections.deque(maxlen=50)
        self.compiles = 0
        self.translate_ms = 0.0
        self.worst_translate_ms = 0.0
        self.worst_translate_at = 0.0
        self.reasons: collections.Counter[str] = collections.Counter()
        self.queued_peak = 0
        self.active_peak = 0
        self.shader_done = 0
        self.bursts = 0
        self.burst_shaders = 0
        self.burst_seconds = 0.0
        self.clock_held = 0.0
        self.shader_counts: dict[str, int] = {}
        self.skip_shaders: set[str] = set()
        self.skip_lines = 0
        self.skip_stages: collections.Counter[str] = collections.Counter()
        self.temp_skips: collections.deque[str] = collections.deque(maxlen=50)
        self.ahead_failed = 0
        self.cache: collections.deque[tuple[float, str]] = collections.deque(maxlen=MAX_LINES)
        self.cache_loaded: dict[str, float | int] = {}
        self.device: collections.deque[str] = collections.deque(maxlen=MAX_LINES)
        self.breadcrumb: dict[str, int] = {}
        self.crash: collections.deque[str] = collections.deque(maxlen=MAX_LINES)
        self.in_crash = False
        self.crash_fault: dict[str, str] = {}
        self.crash_modules: collections.deque[str] = collections.deque(maxlen=50)
        self.crash_threads: collections.deque[str] = collections.deque(maxlen=50)
        self.crash_files: list[str] = []
        self.fatal: collections.deque[str] = collections.deque(maxlen=MAX_LINES)
        self.first_fatal_line: int | None = None

    # -- one line -----------------------------------------------------------------------------
    def line(self, raw: str) -> None:
        self.lines += 1
        line = ANSI_RE.sub('', raw.rstrip('\r\n'))
        try:
            self._line(line)
        except Exception as exc:  # one bad line must not hide the rest of the log
            self.parse_errors += 1

    def _line(self, line: str) -> None:
        at = AT_RE.match(line)
        if at:
            self.t = float(at.group(1))
            if self.first_t is None:
                self.first_t = self.t
            return

        if line.startswith('PerfSummary'):
            m = PERF_RE.match(line)
            if m:
                tag = m.group('tag')
                body = kv(m.group('body'))
                fields = {k: v for k, v in body.items() if k in PERF_COUNTERS}
                val = {
                    't': self.t, 'line': self.lines, 'tag': tag, 'secs': float(m.group('secs')),
                    'counters': fields,
                    'fps': body.get('fps'), 'gpu_cpu_ms_per_frame': body.get('gpu_cpu_ms/f'),
                    'gpu_cpu_us_per_op': body.get('gpu_cpu_us/op'),
                    'gpu_instr_user': body.get('gpu_instr_user'),
                    'gpu_instr_kernel': body.get('gpu_instr_kernel'),
                }
                self.perf.append(val)
                self.perf_lines += 1
                if tag.startswith('total'):
                    self.perf_totals = fields
                    self.t = float(m.group('secs'))
                else:
                    self.t = float(tag.split('=')[1])
                if self.first_t is None:
                    self.first_t = self.t
                return
            if PERF_GPU_RE.match(line):
                return

        if line.startswith('SyncStats '):
            m = SYNC_RE.match(line)
            if m:
                self.t = float(m.group('tag').split('=')[1])
                if self.first_t is None:
                    self.first_t = self.t
                self.sync.append({'t': self.t, 'line': self.lines, 'fields': kv(m.group('body'))})
                return

        m = TRACE_RE.match(line)
        if m:
            self._trace(m)
            return

        m = FLIP_RE.match(line)
        if m:
            self._flip(m)
            return

        if line.startswith(DEVICE_PREFIXES):
            self.device.append(line)
            m = BREADCRUMB_RE.match(line)
            if m:
                self.breadcrumb = {'recorded': int(m.group(1)), 'started': int(m.group(2)),
                                   'completed': int(m.group(3))}
            return

        if line.startswith(FATAL_PREFIXES) or FATAL_RESOURCE_RE.match(line):
            self.fatal.append(line)
            if self.first_fatal_line is None:
                self.first_fatal_line = self.lines
            if line.startswith(CRASH_START):
                self.in_crash = True
                self.crash.append(line)
            return

        if self.in_crash:
            m = CRASH_FAULT_RE.match(line)
            if m:
                self.crash_fault = {'type': m.group(1), 'code': m.group(2), 'pc': '0x' + m.group(3),
                                    'access': m.group(4), 'address': '0x' + m.group(5),
                                    't': self.t, 'line': self.lines}
            elif CRASH_MODULE_RE.match(line) or CRASH_MODULE_ANY_RE.match(line):
                if len(self.crash_modules) < 50:
                    self.crash_modules.append(line.strip())
            elif CRASH_THREAD_RE.match(line):
                if len(self.crash_threads) < 50:
                    self.crash_threads.append(line.strip())
            elif CRASH_FILE_RE.match(line):
                self.crash_files.append(CRASH_FILE_RE.match(line).group(1))
                self.in_crash = False
            elif not (line.startswith(('version:', 'thread:', 'code (pc-48', 'stack:',
                                        'modules (', '--- Guest threads ---')) or
                      re.match(r'^[a-z][a-z0-9 ]*=', line) or
                      re.match(r'^\s+0x[0-9a-fA-F]+(\s+0x[0-9a-fA-F]+)+$', line) or
                      re.match(r'^[0-9a-f ]{16,}$', line)):
                self.in_crash = False
            if len(self.crash) < MAX_LINES:
                self.crash.append(line)
            if self.in_crash:
                return

        m = COMPILING_RE.match(line)
        if m:
            self.queued_peak = max(self.queued_peak, int(m.group(1)))
            self.active_peak = max(self.active_peak, int(m.group(2)))
            self.shader_done = int(m.group(3))
            return
        m = BURST_RE.match(line)
        if m:
            self.bursts += 1
            self.burst_shaders += int(m.group(1))
            self.burst_seconds += float(m.group(2))
            return
        m = CLOCK_HELD_RE.match(line)
        if m:
            self.clock_held = float(m.group(1))
            return
        m = SHADER_COMPILE_RE.match(line)
        if m:
            self.compiles += 1
            ms = float(m.group(2))
            self.translate_ms += ms
            if ms > self.worst_translate_ms:
                self.worst_translate_ms = ms
                self.worst_translate_at = self.t
            reason = m.group(1) + (' ahead' if m.group(3) else '')
            if reason in self.reasons or len(self.reasons) < 500:
                self.reasons[reason] += 1
            return
        m = SHADER_COUNT_RE.match(line)
        if m:
            names = ('vs', 'ps', 'cs', 'gs', 'ls', 'hs', 'tes')
            self.shader_counts = dict(zip(names, (int(x) for x in m.groups())))
            return
        m = SKIP_SHADER_RE.match(line)
        if m:
            self.skip_lines += 1
            add_id(self.skip_shaders, m.group(1))
            self.skip_stages[m.group(2)] += 1
            return
        m = TEMP_SKIP_RE.match(line)
        if m:
            self.temp_skips.append('shader=0x%s pc=0x%s reason=%s' % m.groups())
            return
        if AHEAD_FAILED_RE.match(line):
            self.ahead_failed += 1
            return
        if line.startswith(CACHE_PREFIXES):
            self.cache.append((self.t, line))
            m = CACHE_LOADED_RE.match(line)
            if m:
                self.cache_loaded = {'programs': int(m.group(1)), 'modules': int(m.group(2)),
                                     'pipelines': int(m.group(3)), 'ms': float(m.group(4)),
                                     't': self.t}
            return

    def _trace(self, m: re.Match[str]) -> None:
        frame, _seq, kind, body = m.groups()
        add_id(self.traced_frames, frame)
        if kind == 'D':
            self.draws += 1
            for stage, sid in TRACE_HASH_RE.findall(body):
                if stage == 'ps':
                    continue
                add_id(self.trace_ids[stage], sid)
            ps = re.search(r'\bps=0x([0-9a-fA-F]+)', body)
            if ps:
                add_id(self.trace_ids['ps'], ps.group(1))
        else:
            self.dispatches += 1
            if 'indirect=gpu' in body:
                self.indirect_gpu += 1
            cs = re.search(r'\bcs=0x([0-9a-fA-F]+)', body)
            if cs:
                add_id(self.trace_ids['cs'], cs.group(1))

    def _flip(self, m: re.Match[str]) -> None:
        kind, body = m.groups()
        f = kv(body)
        if kind == 'req':
            return
        if kind == 'present':
            self.flip_present += 1
            self.flip_ooo += int(f.get('ooo', 0))
            self.flip_argback += int(f.get('argback', 0))
        elif kind == 'held':
            self.flip_held += 1
            self.longest_hold_ms = max(self.longest_hold_ms, float(f.get('waited_ms', 0)))
        elif kind == 'wait':
            self.longest_wait_ms = max(self.longest_wait_ms, float(f.get('waited_ms', 0)))
        if len(self.flip_recent) < MAX_LINES:
            self.flip_recent.append(m.group(0)[:300])

    # -- crash report from a file or a run folder ---------------------------------------------
    def read_crash_file(self, path: Path) -> None:
        """Read the loader's crash report as extra crash-section lines (runtimeLinker.cpp:750)."""
        with path.open('rb') as handle:
            for raw in handle:
                line = ANSI_RE.sub('', raw.decode('utf-8', errors='replace').rstrip('\r\n'))
                m = CRASH_FAULT_RE.match(line)
                if m:
                    self.crash_fault = {'type': m.group(1), 'code': m.group(2),
                                        'pc': '0x' + m.group(3), 'access': m.group(4),
                                        'address': '0x' + m.group(5), 'source': path.name}
                elif CRASH_MODULE_RE.match(line):
                    if len(self.crash_modules) < 50:
                        self.crash_modules.append(line.strip())
                elif CRASH_THREAD_RE.match(line):
                    if len(self.crash_threads) < 50:
                        self.crash_threads.append(line.strip())
                elif len(self.crash) < MAX_LINES:
                    self.crash.append(line)

    # -- findings -----------------------------------------------------------------------------
    def findings(self) -> dict:
        no_flip_secs = sum(float(p['secs']) for p in self.perf if p['counters'].get('flips') == '0')
        fps = [float(p['fps']) for p in self.perf if p.get('fps') not in (None, 'na')]
        return {
            'timeline': {
                'first_t': self.first_t, 'last_t': self.t, 'lines': self.lines,
                'perf_intervals': self.perf_lines, 'perf_seconds_without_flips': round(no_flip_secs, 1),
                'fps_min': min(fps) if fps else None, 'fps_max': max(fps) if fps else None,
                'flip_presents': self.flip_present, 'flip_out_of_order': self.flip_ooo,
                'flip_arg_backward': self.flip_argback,
            },
            'perf': {
                'intervals': [dict(p) for p in list(self.perf)[-50:]],
                'interval_count': self.perf_lines, 'totals': self.perf_totals,
            },
            'sync': {'samples': [dict(s) for s in list(self.sync)[-20:]],
                     'sample_count': len(self.sync)},
            'flips': {'presented': self.flip_present, 'out_of_order': self.flip_ooo,
                      'arg_backward': self.flip_argback, 'held': self.flip_held,
                      'longest_hold_ms': self.longest_hold_ms,
                      'longest_wait_ms': self.longest_wait_ms,
                      'recent': list(self.flip_recent)},
            'traces': {'frames': len(self.traced_frames), 'draws': self.draws,
                       'dispatches': self.dispatches, 'gpu_indirect': self.indirect_gpu,
                       'ids_by_stage': {k: len(v) for k, v in sorted(self.trace_ids.items())}},
            'shaders': {
                'compiles': self.compiles, 'translate_ms_total': round(self.translate_ms, 1),
                'worst_translate_ms': self.worst_translate_ms, 'worst_translate_at': self.worst_translate_at,
                'reasons': dict(self.reasons.most_common(20)), 'compiling_peak_queued': self.queued_peak,
                'compiling_peak_active': self.active_peak, 'compiling_done': self.shader_done,
                'bursts': self.bursts, 'burst_shaders': self.burst_shaders,
                'burst_seconds': round(self.burst_seconds, 1), 'clock_held_s': self.clock_held,
                'shader_counts': self.shader_counts, 'skipped_shaders': len(self.skip_shaders),
                'skip_lines': self.skip_lines, 'skip_stages': dict(self.skip_stages),
                'temporary_skips': list(self.temp_skips), 'ahead_failures': self.ahead_failed,
            },
            'cache': {'lines': [dict(t=t, line=s) for t, s in self.cache], 'loaded': self.cache_loaded},
            'device_lost': {'lines': list(self.device), 'breadcrumbs': self.breadcrumb},
            'crash': {'fault': self.crash_fault, 'modules': list(self.crash_modules),
                      'guest_threads': list(self.crash_threads), 'reports': self.crash_files,
                      'lines': list(self.crash)},
            'fatal': {'first_line': self.first_fatal_line, 'lines': list(self.fatal)},
        }

    # -- verdict ------------------------------------------------------------------------------
    def verdict(self, f: dict) -> list[str]:
        v: list[str] = []
        tl, fl, sh, dev, cr, cache = (f['timeline'], f['flips'], f['shaders'],
                                     f['device_lost'], f['crash'], f['cache'])
        if tl['perf_seconds_without_flips']:
            v.append('no flips for %s s of the %s sampled intervals' %
                     (tl['perf_seconds_without_flips'], tl['perf_intervals']))
        if tl['flip_presents']:
            v.append('flips presented=%d, out-of-order=%d, backwards argument=%d, longest hold %s ms' %
                     (fl['presented'], fl['out_of_order'], fl['arg_backward'], fl['longest_hold_ms']))
        if tl['last_t'] is not None and tl['first_t'] is not None and tl['last_t'] > tl['first_t']:
            v.append('log covers %.1f s (%d lines); sampled fps %s..%s' %
                     (tl['last_t'] - tl['first_t'], tl['lines'], tl['fps_min'], tl['fps_max']))
        if sh['compiles']:
            v.append('shader compiles=%d, translate %s ms total, worst %s ms @%s s; top reason %s' %
                     (sh['compiles'], sh['translate_ms_total'], sh['worst_translate_ms'],
                      sh['worst_translate_at'], next(iter(sh['reasons']), '?')))
        if sh['clock_held_s']:
            v.append('guest clock held %s s while GPU queues waited for compiles' % sh['clock_held_s'])
        if sh['skipped_shaders'] or sh['temporary_skips']:
            v.append('skipped shaders=%d (%d lines), temporary dispatch skips=%d' %
                     (sh['skipped_shaders'], sh['skip_lines'], len(sh['temporary_skips'])))
        if cache['loaded']:
            c = cache['loaded']
            v.append('shader cache: %s programs / %s pipelines loaded in %s ms @%s s' %
                     (c['programs'], c['pipelines'], c['ms'], c['t']))
        failed = [s for _, s in cache['lines'] if 'failed' in s or 'could not be decoded' in s]
        if failed:
            v.append('shader cache: %s problem lines (last: %s)' % (len(failed), failed[-1][:120]))
        if dev['lines']:
            v.append('device lost: %s' % dev['lines'][0][:160])
        if cr['fault']:
            v.append('guest fault: pc=%s type=%s code=%s address=%s' %
                     (cr['fault'].get('pc'), cr['fault'].get('type'), cr['fault'].get('code'),
                      cr['fault'].get('address')))
        if f['fatal']['lines']:
            v.append('fatal: %s' % f['fatal']['lines'][-1][:160])
        if not v:
            v.append('no upstream diagnostic lines found in this log')
        return v

    # -- text sections ------------------------------------------------------------------------
    def sections(self, f: dict) -> list[tuple[str, list[str]]]:
        return [
            ('TIMELINE', self._timeline),
            ('PERF', self._perf),
            ('SYNC', self._sync),
            ('FLIPS', self._flips),
            ('TRACES', self._traces),
            ('SHADERS', self._shaders),
            ('CACHE', self._cache),
            ('DEVICE LOST', self._device),
            ('CRASH', self._crash),
            ('FATAL', self._fatal),
        ]

    def _timeline(self, f: dict) -> list[str]:
        tl = f['timeline']
        if not tl['perf_intervals'] and not tl['flip_presents']:
            return []
        return ['@%s..%s s: %d lines, %d PerfSummary intervals, %d flip presentations' %
                (tl['first_t'], tl['last_t'], tl['lines'], tl['perf_intervals'], tl['flip_presents']),
                'sampled fps %s..%s; %s s in intervals with no flips' %
                (tl['fps_min'], tl['fps_max'], tl['perf_seconds_without_flips'])]

    def _perf(self, f: dict) -> list[str]:
        p = f['perf']
        if not p['interval_count']:
            return []
        out = ['%d intervals; last %d:' % (p['interval_count'], len(p['intervals']))]
        for i in p['intervals'][-10:]:
            c = i['counters']
            out.append('@%s s %s secs=%s flips=%s draws=%s dispatches=%s faults %s/%s fps=%s' %
                       (i['t'], i['tag'], i['secs'], c.get('flips', '?'), c.get('draws', '?'),
                        c.get('dispatches', '?'), c.get('write_faults', '?'),
                        c.get('read_faults', '?'), i['fps']))
        if p['totals']:
            out.append('exit totals: %s' % ' '.join('%s=%s' % kv for kv in sorted(p['totals'].items())))
        return out

    def _sync(self, f: dict) -> list[str]:
        s = f['sync']
        if not s['sample_count']:
            return []
        out = ['%d samples; last %d:' % (s['sample_count'], len(s['samples']))]
        for x in s['samples'][-6:]:
            d = x['fields']
            out.append('@%s s flips/s=%s subs/s=%s draws/s=%s faults r/w=%s/%s fault_ms/s=%s '
                       'readbacks/s=%s gpuwait_ms/s=%s' %
                       (x['t'], d.get('flips/s', '?'), d.get('subs/s', '?'), d.get('draws/s', '?'),
                        d.get('r/s', '?'), d.get('w/s', '?'), d.get('fault_ms/s', '?'),
                        d.get('readbacks/s', '?'), d.get('gpuwait_ms/s', '?')))
        return out

    def _flips(self, f: dict) -> list[str]:
        fl = f['flips']
        if not (fl['presented'] or fl['held'] or fl['longest_wait_ms']):
            return []
        return ['presented=%d out-of-order=%d backwards=%d held=%d longest hold=%s ms wait=%s ms' %
                (fl['presented'], fl['out_of_order'], fl['arg_backward'], fl['held'],
                 fl['longest_hold_ms'], fl['longest_wait_ms']),
                'recent: %s' % ('; '.join(fl['recent'][-5:]) or 'none')]

    def _traces(self, f: dict) -> list[str]:
        tr = f['traces']
        if not (tr['draws'] or tr['dispatches']):
            return []
        return ['%d frames: %d draws, %d dispatches (%d with GPU-side arguments)' %
                (tr['frames'], tr['draws'], tr['dispatches'], tr['gpu_indirect']),
                'distinct ids by stage: %s' % ' '.join('%s=%s' % kv for kv in sorted(tr['ids_by_stage'].items()))]

    def _shaders(self, f: dict) -> list[str]:
        sh = f['shaders']
        if not any((sh['compiles'], sh['bursts'], sh['skipped_shaders'], sh['shader_counts'],
                    sh['ahead_failures'], sh['clock_held_s'])):
            return []
        out = ['compiles=%d translate total=%s ms worst=%s ms @%s s; reasons: %s' %
               (sh['compiles'], sh['translate_ms_total'], sh['worst_translate_ms'],
                sh['worst_translate_at'], ' | '.join('%s=%s' % kv for kv in list(sh['reasons'].items())[:8]) or 'none')]
        if sh['shader_counts']:
            out.append('compiled modules by stage: %s' %
                       ' '.join('%s=%s' % kv for kv in sorted(sh['shader_counts'].items())))
        out.append('banner: peak queued=%d active=%d done=%d; %d bursts, %s shaders in %s s; '
                   'guest clock held=%s s' %
                   (sh['compiling_peak_queued'], sh['compiling_peak_active'], sh['compiling_done'],
                    sh['bursts'], sh['burst_shaders'], sh['burst_seconds'], sh['clock_held_s']))
        if sh['skipped_shaders'] or sh['ahead_failures'] or sh['temporary_skips']:
            out.append('skipped=%d shaders (%d lines, stages %s); ahead failures=%d; temporary skips=%d' %
                       (sh['skipped_shaders'], sh['skip_lines'], dict(sh['skip_stages']),
                        sh['ahead_failures'], len(sh['temporary_skips'])))
        return out

    def _cache(self, f: dict) -> list[str]:
        c = f['cache']
        if not c['lines']:
            return []
        out = ['@%s s: %s' % (x['t'], x['line']) for x in c['lines'][-30:]]
        if c['loaded']:
            out.append('loaded: %s' % ' '.join('%s=%s' % kv for kv in sorted(c['loaded'].items())))
        return out

    def _device(self, f: dict) -> list[str]:
        return list(f['device_lost']['lines'])

    def _crash(self, f: dict) -> list[str]:
        c = f['crash']
        if not c['lines']:
            return []
        out = []
        if c['fault']:
            out.append('fault: %s' % ' '.join('%s=%s' % kv for kv in sorted(c['fault'].items())))
        out.extend(c['modules'][:20])
        out.extend(c['guest_threads'][:20])
        out.extend('report: %s' % path for path in c['reports'])
        if len(c['lines']) <= 30:
            out.extend(c['lines'])
        else:
            out.append('(%d crash-report lines; see --json)' % len(c['lines']))
        return out

    def _fatal(self, f: dict) -> list[str]:
        return list(f['fatal']['lines'])


def discover_crash(run_dir: Path) -> Path | None:
    """Newest crash report in a run folder (runtimeLinker.cpp:750)."""
    reports = sorted((run_dir / '_Logs').glob('crash_*.txt'))
    return reports[-1] if reports else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='One-pass triage of an upstream emulator log.')
    ap.add_argument('log', type=Path, help='log file, or a run folder containing kyty.log')
    ap.add_argument('--json', type=Path, help='write the findings as JSON to this path')
    ap.add_argument('--crash', type=Path, help='also read this _Logs/crash_*.txt report')
    ap.add_argument('--run-dir', type=Path, help='run folder used to discover a crash report')
    args = ap.parse_args(argv)

    log = args.log
    if log.is_dir():
        log = log / 'kyty.log'
    triage = Triage()
    try:
        with log.open('rb') as handle:
            for raw in handle:
                triage.line(raw.decode('utf-8', errors='replace'))
    except OSError as exc:
        print('triage: cannot read %s: %s' % (log, exc), file=sys.stderr)
        return 2

    crash = args.crash
    if crash is None and args.run_dir is not None:
        crash = discover_crash(args.run_dir)
    if crash is not None:
        try:
            triage.read_crash_file(crash)
        except OSError as exc:
            print('triage: cannot read crash report %s: %s' % (crash, exc), file=sys.stderr)

    f = triage.findings()
    try:
        verdict = triage.verdict(f)
    except Exception as exc:
        verdict = ['section failed: %s' % exc]
    out = {'log': str(log), 'lines': triage.lines, 'parse_errors': triage.parse_errors,
           'verdict': verdict, 'findings': f}

    def emit(name: str, builder) -> None:
        print('[%s]' % name)
        body = []
        try:
            body = builder(f)
        except Exception as exc:
            body = ['section failed: %s' % exc]
        if not body:
            body = ['not in this log']
        for line in body:
            print('  ' + str(line))

    print('VERDICT:')
    for i, line in enumerate(out['verdict'], 1):
        print('  %d. %s' % (i, line))
    print('FINDINGS:')
    for name, builder in triage.sections(f):
        emit(name, builder)
    print('[RUN-ENDER]')
    fatal = f['fatal']['lines']
    print('FATAL:' if fatal else 'no fatal block')
    for line in fatal:
        print('  ' + line)

    if args.json is not None:
        try:
            args.json.write_text(json.dumps(out, indent=1), encoding='utf-8')
        except OSError as exc:
            print('triage: cannot write %s: %s' % (args.json, exc), file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
