#!/usr/bin/env python3
"""Zip the local .ragdb directory into ragdb.zip for distribution.

Usage:
  python tools/zip_ragdb.py [.ragdb] [output.zip]
"""
import sys
from pathlib import Path
import zipfile


def zip_dir(src_dir: Path, out_zip: Path):
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir))


def main():
    db_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".ragdb")
    out_zip = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("ragdb.zip")

    if not db_dir.exists():
        print(f"Source DB not found: {db_dir}")
        sys.exit(1)
    zip_dir(db_dir, out_zip)
    print(f"Created {out_zip} from {db_dir}")


if __name__ == "__main__":
    main()
