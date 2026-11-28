"""xbox_save_import.py - play a GTA VI Xbox save in Kyty.

GTA VI saves from the Xbox app on PC or an Xbox Series console (GDK "victory" format) load on the
PS5 build unchanged once they are laid out the way Kyty stores PS5 saves. This tool does that.

USAGE
  python tools/savedata/xbox_save_import.py <xbox save> <save folder> [options]

  <xbox save>     The save as a .zip, or a folder anywhere above its containers, e.g. a copy of
                  %LOCALAPPDATA%\\victory. Containers look like Player~<hash>\\v<N>\\{0,1,manifest}.
  <save folder>   Where Kyty reads saves from: `_SaveData` in the folder the emulator runs from,
                  or any folder you then pass as KYTY_SAVE_DIR. It must be empty or missing.

OPTIONS
  --base <folder>   Copy an existing save folder first (keeps your PS5 settings and system data);
                    the Xbox saves are added on top. Must be another folder than <save folder>.
  --with-settings   Also import the Xbox menu settings (SettingsProfileSettings); skipped by default.
  --title-id <id>   Title ID folder to write (default PPSA38500).
  --force           Delete a non-empty <save folder> first.

WHAT TO DO
  1. Convert, either straight into the run folder's saves:
       python tools/savedata/xbox_save_import.py Mandy_Save.zip <run folder>\\_SaveData
     or into a separate folder, keeping the old saves as they are:
       python tools/savedata/xbox_save_import.py Mandy_Save.zip D:\\saves\\mandy --base <run folder>\\_SaveData
       set KYTY_SAVE_DIR=D:\\saves\\mandy      (needs a build with KYTY_SAVE_DIR)
  2. Run the emulator as usual (kyty_emulator --game <game folder>) and press Cross at the start
     screen. The game continues the newest story save instead of starting a new game. With
     KYTY_SAVE_DIR the emulator prints "Save data folder: <folder>".
  3. The game writes into that folder (autosaves). Convert again to get the original save back.

WHAT IT WRITES
  For the newest v<N> of each container: blob 0 ("Save Game Data", the same ASAV file the PS5
  build writes) copied unchanged to <title id>\\SAVEDATA<Name>\\<Name>, plus sce_sys\\param.bin and
  sce_sys\\blocks.bin (Kyty lists only folders that have them). Files get the save's own time,
  because the game continues from the slot Kyty reports as newest. It stops if a header checksum
  does not match. Format notes: docs/xbox-saves.md.
"""
import argparse
import datetime
import os
import re
import shutil
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

HEADER_SIZE = 0x170
CHECKSUM_OFFSET = 0x160
CHECKSUM_SEED = 0x4BF672ED
PARAM_SIZE = 0x530  # title[128] sub_title[128] detail[1024] user_param pad mtime reserved[32]
BLOCKS = 48


def header_checksum(data: bytes) -> int:
    """Jenkins one-at-a-time over the 0x170-byte ASAV header with the checksum field zeroed.
    The game sign-extends each byte and seeds the hash with 0x4bf672ed."""
    header = bytearray(data[:HEADER_SIZE])
    header[CHECKSUM_OFFSET:CHECKSUM_OFFSET + 4] = b"\0\0\0\0"
    h = CHECKSUM_SEED
    for byte in header:
        h = (h + (byte - 256 if byte > 127 else byte)) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    return (h + (h << 15)) & 0xFFFFFFFF


def describe(data: bytes) -> dict:
    if len(data) < HEADER_SIZE or data[:4] != b"ASAV":
        raise ValueError("not an ASAV save file")
    version, kind = struct.unpack_from("<II", data, 4)
    (stamp,) = struct.unpack_from("<Q", data, 0x20)
    title = data[0x28:0x128].decode("utf-16-le", "replace").split("\0", 1)[0]
    (stored,) = struct.unpack_from("<I", data, CHECKSUM_OFFSET)
    return {"version": version, "type": kind, "time": stamp, "title": title,
            "checksum_ok": stored == header_checksum(data)}


def read_manifest(path: Path) -> list:
    data, names, pos = path.read_bytes(), [], 0
    while pos + 4 <= len(data):
        (length,) = struct.unpack_from("<I", data, pos)
        names.append(data[pos + 4:pos + 4 + length].decode("utf-8", "replace"))
        pos += 4 + length
    return names


def find_containers(root: Path) -> dict:
    """Newest version folder of every <Name>~<hash> container under root."""
    found = {}
    for folder in root.rglob("*~*"):
        match = re.fullmatch(r"(.+)~[0-9a-fA-F]+", folder.name)
        if not folder.is_dir() or not match:
            continue
        versions = [(int(v.name[1:]), v) for v in folder.iterdir()
                    if v.is_dir() and re.fullmatch(r"v\d+", v.name)]
        if versions:
            found[match.group(1)] = max(versions)[1]
    return found


def param_bin(subtitle: str, mtime: int) -> bytes:
    param = bytearray(PARAM_SIZE)
    param[0:6] = b"GTA VI"
    encoded = subtitle.encode("utf-8")[:127]
    param[0x80:0x80 + len(encoded)] = encoded
    struct.pack_into("<q", param, 0x508, mtime)
    return bytes(param)


def import_saves(source: Path, output: Path, title_id: str, with_settings: bool) -> int:
    containers = find_containers(source)
    if not containers:
        print(f"no <Name>~<hash>\\v<N> containers under {source}", file=sys.stderr)
        return 1
    title_dir = output / title_id
    title_dir.mkdir(parents=True, exist_ok=True)
    for name, version_dir in sorted(containers.items()):
        if name.startswith("Settings") and not with_settings:
            print(f"skip   {name} (--with-settings imports the Xbox menu settings)")
            continue
        manifest = read_manifest(version_dir / "manifest") if (version_dir / "manifest").exists() else []
        blobs = manifest[2:] if len(manifest) >= 3 else []
        index = blobs.index("Save Game Data") if "Save Game Data" in blobs else 0
        data = (version_dir / str(index)).read_bytes()
        info = describe(data)
        if not info["checksum_ok"]:
            print(f"FAIL   {name}: header checksum does not match", file=sys.stderr)
            return 1
        save_dir = title_dir / f"SAVEDATA{name}"
        (save_dir / "sce_sys").mkdir(parents=True, exist_ok=True)
        files = {save_dir / name: data,
                 save_dir / "sce_sys" / "param.bin": param_bin(info["title"] or name, info["time"]),
                 save_dir / "sce_sys" / "blocks.bin": struct.pack("<Q", BLOCKS)}
        for path, content in files.items():
            path.write_bytes(content)
            os.utime(path, (info["time"], info["time"]))
        when = datetime.datetime.fromtimestamp(info["time"], datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"import {name:<24} {len(data):>7} bytes  ASAV v{info['version']} type {info['type']}  "
              f"{when}  {info['title']!r}  (from {version_dir.parent.name}\\{version_dir.name})")
    return 0


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help", "/?"):
        print(__doc__)
        return 0 if len(sys.argv) > 1 else 2
    parser = argparse.ArgumentParser(usage="see `python tools/savedata/xbox_save_import.py` with no arguments")
    parser.add_argument("source", type=Path, help="Xbox save zip or folder")
    parser.add_argument("output", type=Path, help="save root (_SaveData or the KYTY_SAVE_DIR folder)")
    parser.add_argument("--base", type=Path, help="save root to copy first (keeps PS5 settings)")
    parser.add_argument("--title-id", default="PPSA38500")
    parser.add_argument("--with-settings", action="store_true",
                        help="also import SettingsProfileSettings (Xbox menu settings)")
    parser.add_argument("--force", action="store_true", help="replace an existing output folder")
    args = parser.parse_args()

    if args.base and args.base.resolve() == args.output.resolve():
        print("--base must be another folder than the output (import into a copy instead)", file=sys.stderr)
        return 1
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        print(f"{args.output} is not empty; pass --force to replace it", file=sys.stderr)
        return 1
    # The set is built next to the output and swapped in only when the import succeeded, so a
    # missing source or a bad checksum leaves an existing save folder as it was.
    staging = args.output.with_name(args.output.name + ".importing")
    if staging.exists():
        shutil.rmtree(staging)
    if args.base:
        shutil.copytree(args.base, staging)
    if args.source.is_file() and zipfile.is_zipfile(args.source):
        with tempfile.TemporaryDirectory() as temp:
            with zipfile.ZipFile(args.source) as archive:
                archive.extractall(temp)
            result = import_saves(Path(temp), staging, args.title_id, args.with_settings)
    else:
        result = import_saves(args.source, staging, args.title_id, args.with_settings)
    if result != 0:
        shutil.rmtree(staging, ignore_errors=True)
        return result
    if args.output.exists():
        shutil.rmtree(args.output)
    staging.rename(args.output)
    print(f"done: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
