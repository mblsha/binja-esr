#!/usr/bin/env python3
"""Wrap the compiled native frontend in a local, discoverable macOS app."""

import argparse
import plistlib
import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path, default=root / "target/release/oz9600-window"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "target/release/OZ-9600 Emulator.app"
    )
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error("build the release binary first")
    contents = args.output / "Contents"
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.binary, contents / "MacOS/oz9600-window")
    (contents / "Info.plist").write_bytes(
        plistlib.dumps(
            {
                "CFBundleExecutable": "oz9600-window",
                "CFBundleIdentifier": "org.binjaesr.oz9600-window",
                "CFBundleName": "OZ-9600 Emulator",
                "CFBundleDisplayName": "OZ-9600 Emulator",
                "CFBundlePackageType": "APPL",
                "CFBundleShortVersionString": "0.1.0",
                "CFBundleVersion": "1",
                "NSHighResolutionCapable": True,
            }
        )
    )
    print(args.output)


if __name__ == "__main__":
    main()
