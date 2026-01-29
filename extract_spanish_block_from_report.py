#!/usr/bin/env python3
"""
ES: Extrae del reporte del escáner las coincidencias de un archivo y un rango de líneas.
EN: Extract from the scanner report the matches for a file and a line-range.
JP: スキャナーレポートから、指定ファイル＋行範囲の一致項目を抽出します。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


RE_MATCH_LINE = re.compile(r"\s+L(?P<line>\d+)\s+\[(?P<kind>[^\]]+)\]\s+(?P<msg>.*)$")


def extract_block(
    report_lines: list[str],
    *,
    target_file: str,
    start_line: int,
    end_line: int,
) -> list[tuple[int, str, str]]:
    inside = False
    out: list[tuple[int, str, str]] = []

    for raw in report_lines:
        line = raw.rstrip("\n")

        if line.strip() == target_file:
            inside = True
            continue

        if inside:
            # End of section: next file header or end of matches block.
            if line and not line.startswith("  "):
                break

            m = RE_MATCH_LINE.match(line)
            if not m:
                continue

            ln = int(m.group("line"))
            if start_line <= ln <= end_line:
                out.append((ln, m.group("kind"), m.group("msg")))

    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract a line-range block from scan_spanish_console_messages report."
    )
    ap.add_argument("--report", required=True, help="Path to report txt")
    ap.add_argument("--file", required=True, help="Target file section header (e.g. 0sec.py)")
    ap.add_argument("--start", type=int, required=True, help="Start line (inclusive)")
    ap.add_argument("--end", type=int, required=True, help="End line (inclusive)")
    ap.add_argument("--out", required=True, help="Output txt path")
    args = ap.parse_args()

    report_path = Path(args.report)
    report_lines = report_path.read_text(encoding="utf-8", errors="replace").splitlines()

    items = extract_block(
        report_lines,
        target_file=args.file,
        start_line=args.start,
        end_line=args.end,
    )

    out_path = Path(args.out)
    out_path.write_text(
        "\n".join(f"L{ln} [{kind}] {msg}" for ln, kind, msg in items),
        encoding="utf-8",
    )

    print(len(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

