# GTA VI: playing an Xbox save

GTA VI's Xbox saves (the PC Xbox app and Series consoles, GDK "victory" format) load on the PS5 build unchanged once they are laid out the way Kyty stores PS5 saves. `tools/savedata/xbox_save_import.py` does the layout. The game then continues the newest story save after the first Cross press at the start screen.

## Steps

You need Python 3 and the save as a zip or folder. Any folder above the containers works, for example a copy of `%LOCALAPPDATA%\victory`.

1. Convert it into the `_SaveData` folder of the folder you run the emulator from:
   ```powershell
   python tools\savedata\xbox_save_import.py Mandy_Save.zip <run folder>\_SaveData
   ```
   The tool refuses a folder that already has saves. To keep your PS5 settings and system data, convert into a new folder on top of a copy of the old one, then swap the folders:
   ```powershell
   python tools\savedata\xbox_save_import.py Mandy_Save.zip <run folder>\_SaveData.xbox --base <run folder>\_SaveData
   ```
2. Or keep both save sets and choose per run with `KYTY_SAVE_DIR`. It needs a build with that variable; it is not in older builds.
   ```powershell
   python tools\savedata\xbox_save_import.py Mandy_Save.zip D:\saves\mandy
   $env:KYTY_SAVE_DIR = 'D:\saves\mandy'
   .\kyty_emulator.exe --game D:\GTA6_PS5
   ```
   The emulator prints `Save data folder: D:\saves\mandy` when the game first touches its saves.
3. Start the game and press Cross at the start screen. It loads the newest story slot, not a new game.

The game writes into the folder it loaded (autosaves, `SAVEDATA00\SPACECHECK00.dat`). Convert again to go back to the original save.

Xbox menu settings (`SettingsProfileSettings`) are skipped unless you pass `--with-settings`, because the PS5 build has its own.

## What the tool does

Each Xbox container is `<Name>~<hash>\v<N>\` with blobs `0` ("Save Game Data"), `1` ("Save Game Header", Xbox only) and `manifest`. For the newest `v<N>` of each container, the tool:
- copies blob `0` unchanged to `<title id>\SAVEDATA<Name>\<Name>`;
- writes `sce_sys\param.bin` (title "GTA VI", the save's title as subtitle, its time) and `sce_sys\blocks.bin`, without which Kyty does not list the folder;
- gives the files the save's own time. Kyty sorts the game's newest-first slot search by file time, and the game continues from the first slot.

It stops if a header checksum does not match.

## Format notes

Every save file starts with a 0x170-byte `ASAV` header:

| Offset | Field |
|---|---|
| 0x00 | `ASAV` |
| 0x04 | version: the PS5 build writes 1, Xbox saves carry 2; the PS5 build does not check it |
| 0x08 | type: 1 story slot (`SARC`), 2 settings, 3 player |
| 0x0C | flags, fixed per type (3 for story slots, 1 for settings and player) |
| 0x20 | save time (Unix seconds) |
| 0x28 | title, UTF-16 (mission name for story slots) |
| 0x128 | container name, ASCII |
| 0x160 | checksum |
| 0x168 | file size |

The checksum is Jenkins one-at-a-time over the header with the checksum field zeroed, seeded with `0x4bf672ed`, with each byte sign-extended; it is the same on both platforms. The PS5 build (01.002.001) checks magic, type, flags, name fields and checksum at `0x9084047f0`. Story slot data after the header is encrypted the same way on both platforms. Story slots are `SARC0000` to `SARC0014`, and `SARC0015` is the autosave.

## Tested (2026-11-27, PPSA38500 01.002.001)

Two Xbox saves, "Mandy" (`SARC0000`–`SARC0002`) and "Manual" (`SARC0000` and autosave `SARC0015`), loaded into the world from the newest slot. The game awarded the progress trophy "Just Like Paradise", which new-game runs have never logged.
