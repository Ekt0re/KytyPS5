#!/usr/bin/env python3
"""Deterministic tests for tools/triage/triage.py.

Every log line here is synthetic and built from an upstream format string cited in triage.py.
Shader ids are generated from small integers by H(); no real identifier is written in this file.
No emulator, run data or network access is needed.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRIAGE = HERE.parent / 'triage.py'

ALL_SECTIONS = ('TIMELINE', 'PERF', 'SYNC', 'FLIPS', 'TRACES', 'SHADERS', 'CACHE',
                'DEVICE LOST', 'CRASH', 'FATAL')


def H(n):
    """A synthetic 16-hex-digit id made from a small integer."""
    return '0x%016x' % n


def sections(text):
    result = {}
    current = None
    for line in text.splitlines():
        m = re.match(r'^\[([A-Z][A-Z -]*)\]$', line)
        if m:
            current = m.group(1)
            result[current] = []
        elif current and line.startswith('  ') and line.strip():
            result[current].append(line[2:])
    return result


def run_log(text, *args, crlf=False, extra_files=None):
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / 'snippet.log'
        data = text.replace('\n', '\r\n').encode() if crlf else text.encode()
        log.write_bytes(data)
        out = Path(d) / 'triage.json'
        cmd = [sys.executable, str(TRIAGE), str(log), '--json', str(out), *args]
        for name, content in (extra_files or {}).items():
            (Path(d) / name).write_bytes(content.encode())
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
        if p.returncode or not p.stdout.startswith('VERDICT:'):
            raise AssertionError('triage failed: %d %s %s' % (p.returncode, p.stderr, p.stdout))
        return p.stdout, json.loads(out.read_text(encoding='utf-8'))


# Upstream formats, as plain Python strings. Sources are named in triage.py.
def perf_line(tag, secs, flips, draws=10, dispatches=2, fps='1.00'):
    # src/common/perfSummary.cpp:164-226, counters in the order of g_names (:28-34).
    counters = dict(flips=flips, draws=draws, dispatches=dispatches, indirect_dispatches=0,
                    write_faults=3, read_faults=1, buf_readbacks=0, tex_readbacks=0,
                    tex_lookups=5, buf_lookups=7, desc_writes=9, heap_desc_writes=0,
                    pipe_lookups=4, pipe_created=1, amm_maps=0, amm_unmaps=0, gpu_unmaps=0,
                    heap_rebinds=2, heap_lookups=6, heap_range_skips=0, depth_recreates=0)
    body = ' '.join('%s=%d' % kv for kv in counters.items())
    return ('PerfSummary %s secs=%s %s gpu_cpu_ms=12 gpu_instr_user=na gpu_instr_kernel=na '
            'gpu_csw_vol=4 gpu_csw_invol=1 proc_cpu_ms=30 proc_minflt=2 proc_majflt=0 '
            'proc_csw_vol=9 proc_csw_invol=3 | fps=%s gpu_cpu_ms/f=12.00 '
            'flips/f=1.0 draws/f=10.0 | gpu_cpu_us/op=1000.0' % (tag, secs, body, fps))


SYNC_LINE = (
    'SyncStats t=10 flips/s=1.0 subs/s=9 draws/s=10 | faults r/s=1 w/s=3 all/s=4 fault_ms/s=2.5 '
    '(buf 1.0 tex 0.5 heap 1.0) | readbacks/s=2 detached/s=1 zc_releases/s=0 readback_ms/s=3.0 '
    'bufdown_MB/s=1.0 texdown_MB/s=0.5 | bufup_MB/s=2.0 reup_MB/s=0.1 unchanged_MB/s=0.0 '
    'stream_MB/s=0.0 bda_MB/s=0.0 texup_MB/s=0.5 | protect/s=4 protect_MB/s=0.2 '
    'protect_ms/s=0.5 | gpuwait_ms/s=12 gputhread_ms/s=30'
)

CRASH_BLOCK = '\n'.join([
    '--- Guest fault context ---',
    'version: local',
    'thread: worker',
    'type=1 code=4 pc=%s access=2 address=%s' % (H(0x1234), H(0)),
    'rax=%s rbx=%s rcx=%s rdx=%s' % (H(1), H(2), H(3), H(4)),
    'code (pc-48 .. pc+48, fault at byte 48):',
    ' 48 8b 05 01 02 03 04',
    'stack:',
    ' %s %s' % (H(5), H(6)),
    'modules (base / size / name; * holds pc):',
    '  * %s 0x00001000 main' % H(0x1000),
    '    %s 0x00002000 other' % H(0x2000),
    '--- Guest threads ---',
    '  [7] worker                           host=11 waits in wait object=0x20 value=3 for 1.5 s',
    'crash report: _Logs/crash_stamp.txt',
])


class PerfTest(unittest.TestCase):
    def test_perf_intervals_and_totals(self):
        text = ('@@T 1\n' + perf_line('t=10', '10.0', 5) + '\n@@T 11\n' +
                perf_line('total t=11', '11.0', 6) + '\n')
        out, data = run_log(text)
        p = data['findings']['perf']
        self.assertEqual(p['interval_count'], 2)
        self.assertEqual(p['intervals'][0]['counters']['flips'], '5')
        self.assertEqual(p['intervals'][0]['fps'], '1.00')
        self.assertEqual(p['intervals'][0]['counters']['flips'], '5')
        self.assertEqual(p['totals']['draws'], '10')
        self.assertIn('fps=1.00', '\n'.join(sections(out)['PERF']))

    def test_no_flips_in_intervals_leads_verdict(self):
        text = perf_line('t=10', '10.0', 0, fps='0.00') + '\n' + perf_line('t=20', '10.0', 0, fps='0.00') + '\n'
        _, data = run_log(text)
        self.assertEqual(data['findings']['timeline']['perf_seconds_without_flips'], 20.0)
        self.assertTrue(data['verdict'][0].startswith('no flips for 20.0 s'))

    def test_sync_stats_fields(self):
        out, data = run_log('@@T 10\n' + SYNC_LINE + '\n')
        s = data['findings']['sync']
        self.assertEqual(s['sample_count'], 1)
        self.assertEqual(s['samples'][0]['fields']['fault_ms/s'], '2.5')
        self.assertEqual(s['samples'][0]['fields']['gpuwait_ms/s'], '12')
        self.assertIn('faults r/w=1/3', '\n'.join(sections(out)['SYNC']))


class TraceTest(unittest.TestCase):
    def test_draw_and_dispatch_traces(self):
        # shaderTrace.h:20-21, renderDraw.cpp:1263, renderCompute.cpp:726,940.
        text = ('TRACE f=3 n=0 D vs=%s ps=%s idx=30 inst=1 indirect=0 mesh_groups=0 task=0 '
                'prim=4 colors=1 depth=0\n' % (H(1), H(2)) +
                'TRACE f=3 n=1 C cs=%s groups=4x1x1 local=8x8x1 buffers=2 images=1\n' % H(3) +
                'TRACE f=4 n=0 C cs=%s indirect=gpu groups=1x1x1 local=8x8x1 buffers=1 images=0\n' % H(3))
        out, data = run_log(text)
        tr = data['findings']['traces']
        self.assertEqual((tr['draws'], tr['dispatches'], tr['gpu_indirect'], tr['frames']), (1, 2, 1, 2))
        self.assertEqual(tr['ids_by_stage'], {'cs': 1, 'ps': 1, 'vs': 1})
        self.assertIn('1 draws, 2 dispatches (1 with GPU-side arguments)',
                      '\n'.join(sections(out)['TRACES']))


class FlipTest(unittest.TestCase):
    def test_fliptrace_events(self):
        # videoOut.cpp:932,1144,1153,1246.
        text = ('FLIPTRACE req id=9 bus=0 idx=1 arg=5 src=cpu pending=2 ptc=100\n'
                'FLIPTRACE present id=9 bus=0 idx=1 arg=5 src=cpu prev_id=8 prev_arg=4 ooo=0 '
                'argback=0 vblank=3 pending=1 totals: presented=1 ooo=0 argback=0 held=0\n'
                'FLIPTRACE held bus=0 idx=1 waited_ms=1500 eop_queue=2 cpu_queue=0 head_id=9 '
                'head_idx=1 head_state=2 held_total=1\n'
                'FLIPTRACE wait bus=0 idx=1 waited_ms=1600.5\n')
        out, data = run_log(text)
        fl = data['findings']['flips']
        self.assertEqual((fl['presented'], fl['held'], fl['longest_hold_ms'], fl['longest_wait_ms']),
                         (1, 1, 1500.0, 1600.5))
        self.assertIn('presented=1', '\n'.join(sections(out)['FLIPS']))


class ShaderTest(unittest.TestCase):
    def test_compile_banner_and_clock_hold(self):
        # shaderCompileStatus.cpp:60,70; compilePause.h:95; pipelineCache.cpp:746.
        text = ('@@T 5\n[shaders] compiling: 3 queued, 2 active, 7 done\n'
                'Shader compile: new-code (translate 12.5 ms, SPIR-V 4.0 ms)\n'
                'Shader compile: specialization (translate 20.0 ms ahead, SPIR-V 1.0 ms)\n'
                '@@T 9\n[shaders] burst done: 4 shaders in 3.5 s\n'
                'Guest clock: held 15.0 s in total while GPU queues waited for compiles\n'
                'Shaders: VS 1 | PS 2 | CS 3 | GS 0 | LS 0 | HS 0 | TES 0\n')
        out, data = run_log(text)
        sh = data['findings']['shaders']
        self.assertEqual((sh['compiles'], sh['compiling_peak_queued'], sh['compiling_peak_active']),
                         (2, 3, 2))
        self.assertEqual((sh['bursts'], sh['burst_shaders'], sh['clock_held_s']), (1, 4, 15.0))
        self.assertEqual(sh['shader_counts']['cs'], 3)
        self.assertEqual(sh['worst_translate_ms'], 20.0)
        joined = '\n'.join(sections(out)['SHADERS'])
        self.assertIn('new-code', joined)
        self.assertIn('guest clock held=15.0 s', joined)

    def test_skipped_shader_lines(self):
        # pipelineCache.cpp:571,685.
        text = ('KYTY_SKIP shader=%s stage=cs\n' % H(9) +
                '!!! TEMPORARY COMPUTE DISPATCH SKIP !!! shader=%s pc=0x00000020 '
                'reason=unsupported GPU-selected callable target; rendering is incomplete\n' % H(10))
        out, data = run_log(text)
        sh = data['findings']['shaders']
        self.assertEqual((sh['skipped_shaders'], sh['skip_lines'], len(sh['temporary_skips'])), (1, 1, 1))
        self.assertEqual(sh['skip_stages']['cs'], 1)
        self.assertIn('temporary skips=1', '\n'.join(sections(out)['SHADERS']))


class CacheTest(unittest.TestCase):
    def test_cache_lines_and_load_totals(self):
        # pipelineSerialization.cpp:344,411,507; pipelineCache.cpp:1354,1507.
        text = ('@@T 2\nShader cache: compiling shaders and pipelines from cache.bin (4.0 MB) with 8 threads\n'
                'Shader cache: loaded 5 programs, 6 modules, 7 pipelines in 9 ms (read 2 ms, '
                'records 3 ms, pipelines 4 ms)\n'
                'Vulkan pipeline cache: loaded 1.20 MB from driver.bin\n'
                'Shader precompile: translation failed: bad module\n')
        out, data = run_log(text)
        c = data['findings']['cache']
        self.assertEqual(len(c['lines']), 4)
        self.assertEqual((c['loaded']['programs'], c['loaded']['pipelines']), (5, 7))
        self.assertTrue(data['verdict'][-1].startswith('shader cache:') or
                        any(x.startswith('shader cache:') for x in data['verdict']))
        self.assertIn('Shader cache: loaded 5 programs', '\n'.join(sections(out)['CACHE']))


class DeviceLostTest(unittest.TestCase):
    def test_device_lost_block(self):
        # gpuBreadcrumbs.cpp:119-139,157-168; bdaGuard.cpp:241-245.
        text = '\n'.join([
            'GPU device lost (device wait)',
            'GPU breadcrumbs: recorded 12, started 11, completed 9',
            '  in flight: draw #11 vs=%s ps=%s' % (H(1), H(2)),
            '  in flight: dispatch #10 cs=%s' % H(3),
            '  last completed: dispatch #9 cs=%s' % H(3),
            'Device fault: a fault was reported',
            '  address type 0 %s (precision 4)' % H(0x2000),
            '  [bda-guard] 0x2000: live buffer #3 0x2000+0x100 guest 0x3000',
            '  [bda-guard] 4 live buffers, 2 kept (1 MiB), 3 findings',
            '  vendor 0x1 data 0x2: detail',
        ])
        out, data = run_log(text)
        d = data['findings']['device_lost']
        self.assertEqual(d['breadcrumbs'], {'recorded': 12, 'started': 11, 'completed': 9})
        self.assertEqual(len(d['lines']), 10)
        joined = '\n'.join(sections(out)['DEVICE LOST'])
        self.assertIn('in flight: draw', joined)
        self.assertIn('Device fault:', joined)
        self.assertTrue(any(x.startswith('device lost:') for x in data['verdict']))

    def test_breadcrumbs_unavailable(self):
        _, data = run_log('GPU breadcrumbs: requested but unavailable (no VK_AMD_buffer_marker)\n')
        self.assertEqual(data['findings']['device_lost']['lines'],
                         ['GPU breadcrumbs: requested but unavailable (no VK_AMD_buffer_marker)'])


class CrashTest(unittest.TestCase):
    def test_crash_block(self):
        out, data = run_log('@@T 4\n' + CRASH_BLOCK + '\n')
        c = data['findings']['crash']
        self.assertEqual(c['fault']['pc'], H(0x1234))
        self.assertEqual(c['fault']['address'], H(0))
        self.assertEqual(len(c['modules']), 2)
        self.assertEqual(len(c['guest_threads']), 1)
        self.assertEqual(c['reports'], ['_Logs/crash_stamp.txt'])
        self.assertTrue(any(x.startswith('guest fault:') for x in data['verdict']))
        self.assertIn('host=11', '\n'.join(sections(out)['CRASH']))

    def test_crash_file_option(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'snippet.log'
            log.write_text('@@T 1\nnothing here\n', encoding='utf-8')
            report = Path(d) / 'crash_extra.txt'
            report.write_text('type=1 code=2 pc=%s access=0 address=%s\n' % (H(7), H(0)),
                              encoding='utf-8')
            out = Path(d) / 'triage.json'
            p = subprocess.run([sys.executable, str(TRIAGE), str(log), '--crash', str(report),
                                '--json', str(out)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            data = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(data['findings']['crash']['fault']['pc'], H(7))
            self.assertIn('CRASH', sections(p.stdout))

    def test_run_folder_discovers_crash(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'run'
            (run / '_Logs').mkdir(parents=True)
            (run / 'kyty.log').write_text('@@T 1\nnothing here\n', encoding='utf-8')
            (run / '_Logs' / 'crash_1.txt').write_text(
                'type=1 code=2 pc=%s access=0 address=%s\n' % (H(8), H(0)), encoding='utf-8')
            p = subprocess.run([sys.executable, str(TRIAGE), str(run)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('CRASH', sections(p.stdout))


class RobustnessTest(unittest.TestCase):
    def test_absent_sections_say_not_in_this_log(self):
        out, data = run_log('@@T 1\nnothing interesting here\n')
        found = sections(out)
        for name in ALL_SECTIONS:
            self.assertEqual(found[name], ['not in this log'], name)
        self.assertIn('no upstream diagnostic lines found', data['verdict'][0])
        self.assertEqual(data['parse_errors'], 0)
        self.assertEqual(out.strip().splitlines()[-1], 'no fatal block')

    def test_fatal_block_and_run_ender_last(self):
        # assert.cpp:16-39.
        text = ('--- Build ---\nlocal build\n--- Error ---\n'
                'Error: condition (x) is true in src/a.cpp:12\n')
        out, data = run_log(text)
        f = data['findings']['fatal']
        self.assertEqual(len(f['lines']), 3)
        self.assertTrue(data['verdict'][-1].startswith('fatal:'))
        names = re.findall(r'^\[([A-Z][A-Z -]*)\]$', out, re.M)
        self.assertEqual(names[-1], 'RUN-ENDER')

    def test_one_section_failure_does_not_hide_the_rest(self):
        source = TRIAGE.read_text(encoding='utf-8')
        needle = '    def _device(self, f: dict) -> list[str]:\n'
        broken = source.replace(needle, needle + "        raise ValueError('boom')\n", 1)
        self.assertNotEqual(broken, source)
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / 'broken.py'
            log = Path(d) / 's.log'
            script.write_text(broken, encoding='utf-8')
            log.write_text('GPU device lost (test)\n' + SYNC_LINE + '\n', encoding='utf-8')
            p = subprocess.run([sys.executable, str(script), str(log)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            found = sections(p.stdout)
            self.assertEqual(found['DEVICE LOST'], ['section failed: boom'])
            self.assertTrue(found['SYNC'])
            self.assertTrue(found['TIMELINE'])

    def test_garbage_and_windows_line_endings(self):
        raw = ('@@T 1\r\n' + perf_line('t=10', '10.0', 1) + '\r\n' +
               SYNC_LINE + '\r\n' + 'garbage \xff\xfe bytes\r\n')
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 's.log'
            log.write_bytes(raw.encode('utf-8', errors='surrogateescape'))
            out = Path(d) / 'triage.json'
            p = subprocess.run([sys.executable, str(TRIAGE), str(log), '--json', str(out)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            data = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(data['findings']['perf']['interval_count'], 1)
            self.assertEqual(data['findings']['sync']['sample_count'], 1)
            for name in ALL_SECTIONS:
                self.assertTrue(name in sections(p.stdout), name)

    def test_crlf_matches_lf(self):
        text = perf_line('t=10', '10.0', 4) + '\n' + SYNC_LINE + '\n'
        _, lf = run_log(text)
        _, crlf = run_log(text, crlf=True)
        self.assertEqual(lf['findings']['perf'], crlf['findings']['perf'])
        self.assertEqual(lf['findings']['sync'], crlf['findings']['sync'])


class RunWrapperTest(unittest.TestCase):
    def test_run_folder_manifest_log_and_triage(self):
        # run.py starts the child with subprocess only; a second Python is a portable stand-in.
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'run'
            code = "print('PerfSummary t=10 secs=10.0 flips=1 draws=2 dispatches=0')"
            p = subprocess.run([sys.executable, str(HERE.parent / 'run.py'), '--out', str(run),
                                '--emulator', sys.executable, '-e', 'KYTY_TEST_SWITCH=1',
                                '--', '-c', code], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            manifest = (run / 'run.txt').read_text(encoding='utf-8')
            self.assertIn('KYTY_TEST_SWITCH=1', manifest)
            self.assertIn('exit code: 0', manifest)
            self.assertIn('PerfSummary', (run / 'kyty.log').read_text(encoding='utf-8'))
            data = json.loads((run / 'triage.json').read_text(encoding='utf-8'))
            self.assertEqual(data['findings']['perf']['intervals'][0]['counters']['flips'], '1')
            self.assertIn('VERDICT:', (run / 'triage.txt').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
