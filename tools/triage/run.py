#!/usr/bin/env python3
"""Run the emulator once in a fresh folder and triage its log at the end.

Usage:
    python3 tools/triage/run.py --out RUN_DIR [-e NAME=VALUE ...] [--emulator PATH] [-- ARGS...]

Only subprocess is used to start the child; there is no shell, signal or platform-specific
call, so this works on Windows, Linux and macOS. The child runs with RUN_DIR as its working
folder, which is where the loader writes its crash report.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRIAGE = HERE / 'triage.py'


def emulator_path(given: str | None) -> str:
    if given is not None:
        return str(Path(given).resolve())
    name = 'kyty_emulator.exe' if os.name == 'nt' else 'kyty_emulator'
    local = Path.cwd() / name
    if local.is_file():
        return str(local.resolve())
    found = shutil.which('kyty_emulator')
    if found is None:
        raise SystemExit('run.py: no emulator given and %s was not found' % name)
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Run the emulator once and triage its log.')
    ap.add_argument('--out', type=Path, required=True, help='new run folder')
    ap.add_argument('--emulator', help='emulator executable (default: kyty_emulator on PATH/cwd)')
    ap.add_argument('-e', '--env', action='append', default=[], metavar='NAME=VALUE',
                    help='environment variable for the child (repeatable)')
    ap.add_argument('args', nargs=argparse.REMAINDER, help='arguments passed to the emulator')
    args = ap.parse_args(argv)

    extra: dict[str, str] = {}
    for item in args.env:
        if '=' not in item or not item.split('=', 1)[0]:
            raise SystemExit('run.py: -e expects NAME=VALUE, got %r' % item)
        name, value = item.split('=', 1)
        extra[name] = value

    emu_args = list(args.args)
    if emu_args and emu_args[0] == '--':
        emu_args.pop(0)

    run_dir: Path = args.out
    if (run_dir / 'run.txt').exists():
        raise SystemExit('run.py: %s already has a run.txt; use a new folder' % run_dir)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit('run.py: cannot create %s: %s' % (run_dir, exc))

    exe = emulator_path(args.emulator)
    argv_child = [exe, *emu_args]
    env = os.environ.copy()
    env.update(extra)
    switches = sorted((k, v) for k, v in env.items() if k.startswith('KYTY_'))

    def quoted(items: list[str]) -> str:
        return ' '.join(repr(x) if any(c.isspace() for c in x) else x for x in items)

    manifest = [
        'start: %s' % datetime.now(timezone.utc).isoformat(),
        'command: %s' % quoted(argv_child),
        'emulator: %s' % exe,
        'platform: %s' % sys.platform,
        'environment switches:',
    ]
    manifest += ['  %s=%s' % kv for kv in switches] or ['  (none)']
    (run_dir / 'run.txt').write_text('\n'.join(manifest) + '\n', encoding='utf-8')

    log = run_dir / 'kyty.log'
    with log.open('wb') as handle:
        rc = subprocess.run(argv_child, cwd=run_dir, env=env, stdout=handle,
                            stderr=subprocess.STDOUT, check=False).returncode
    with (run_dir / 'run.txt').open('a', encoding='utf-8') as handle:
        handle.write('exit code: %d\n' % rc)

    triage_txt = run_dir / 'triage.txt'
    triage_json = run_dir / 'triage.json'
    with triage_txt.open('w', encoding='utf-8') as handle:
        subprocess.run([sys.executable, str(TRIAGE), str(log), '--json', str(triage_json),
                        '--run-dir', str(run_dir)], stdout=handle, stderr=subprocess.STDOUT,
                       check=False)

    print('run.py: exit code %d' % rc)
    print('run.py: %s' % (run_dir / 'run.txt'))
    print('run.py: %s' % log)
    print('run.py: %s' % triage_txt)
    print('run.py: %s' % triage_json)
    return rc


if __name__ == '__main__':
    sys.exit(main())
