#!/usr/bin/env python3
"""
Scan code files for Spanish comments that do NOT start with "# ES:".

Heuristics:
- Python: use tokenize to extract COMMENT tokens (robust vs strings).
- Other files: simple regex for // and # inline comments (best-effort).

Outputs grouped matches: file, line number, and comment text.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


DEFAULT_EXTS = [
    ".py",
    ".ps1",
    ".sh",
    ".bat",
    ".cmd",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
]

DEFAULT_EXCLUDE_DIRS = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "Lib",          # common venv layout on Windows
    "Scripts",      # common venv layout on Windows
    "share",        # often large (node / misc assets)
    "installer",
    "build_logs",
    "dist",
    "build",
    "node_modules",
]


LANG_MARKER_RE = re.compile(r"^(ES|EN|JP|JA)\s*:\s*", re.IGNORECASE)
SPANISH_CHARS_RE = re.compile(r"[áéíóúüñ¿¡ÁÉÍÓÚÜÑ]")
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")

# Common Spanish function words + app-domain verbs/nouns.
SPANISH_STRONG_WORDS_RE = re.compile(
    r"\b("
    r"archivo|carpeta|ruta|leer|cargar|guardar|crear|borrar|limpiar|buscar|copiar|obtener|"
    r"cerrar|abrir|agregar|quitar|seleccionar|validar|verificar|"
    r"nota|metodo|método|"
    r"prediccion|predicción|analisis|análisis|resultado|configuracion|configuración|modelo|"
    r"habilitar|deshabilitar|"
    r"advertencia|critico|crítico"
    r")\b",
    re.IGNORECASE,
)

# Weak/common words (very prone to false positives in English), used only as support.
SPANISH_WEAK_WORDS_RE = re.compile(
    r"\b("
    r"el|la|los|las|un|una|unos|unas|pero|porque|"
    r"por|para|con|sin|"
    r"si|sino|solo|sólo|"
    r"desde|hasta|mientras|cuando|donde"
    r")\b",
    re.IGNORECASE,
)


def looks_like_spanish(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    # Filter out Japanese/Chinese/Korean comments early.
    if CJK_RE.search(t):
        return False
    # Filter out likely mojibake/replacement characters (common when decoding mismatch).
    if "�" in t:
        return False
    # Quick accept if it has Spanish punctuation/accents.
    if re.search(r"[¿¡]", t):
        return True

    strong = SPANISH_STRONG_WORDS_RE.findall(t)
    weak = SPANISH_WEAK_WORDS_RE.findall(t)

    # Require at least one strong indicator; weak words alone are too noisy.
    if len(strong) >= 2:
        return True
    if len(strong) == 1:
        # One strong word is enough if the comment is short-ish, or also has weak support, or has accents.
        if len(t.split()) <= 10:
            return True
        if len(weak) >= 1:
            return True
        if SPANISH_CHARS_RE.search(t):
            return True

    # Fallback: allow multiple weak Spanish function words to qualify.
    # This catches short, accent-less Spanish notes like "por si cambia ..." that otherwise get missed.
    words = t.split()
    if len(strong) == 0:
        if len(weak) >= 3:
            return True
        if len(weak) >= 2 and len(words) <= 8:
            return True

    return False


def is_language_marker_comment(comment_text: str) -> bool:
    # comment_text should be WITHOUT the leading comment token (# or //)
    return bool(LANG_MARKER_RE.match(comment_text.strip()))


@dataclass(frozen=True)
class Match:
    path: Path
    line: int
    comment: str  # comment content including marker token, e.g. "# ..."


def iter_files(root: Path, exts: Sequence[str], exclude_dirs: Sequence[str]) -> Iterator[Path]:
    exclude = set(exclude_dirs)
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.name == "scan_spanish_comments.py":
                continue
            if p.suffix.lower() in exts:
                yield p


def extract_python_comments(path: Path) -> Iterator[Match]:
    try:
        # Use tokenize.open() to respect PEP-263 encoding cookies.
        with tokenize.open(str(path)) as f:
            for tok in tokenize.generate_tokens(f.readline):
                if tok.type == tokenize.COMMENT:
                    yield Match(path=path, line=tok.start[0], comment=tok.string)
    except (SyntaxError, tokenize.TokenError, OSError, UnicodeError):
        return


LINE_COMMENT_RE = re.compile(r"(?P<prefix>^[ \t]*|[^'\"])(?P<tok>//|#)\s*(?P<text>.*)$")


def extract_generic_line_comments(path: Path) -> Iterator[Match]:
    # Best-effort for non-Python; for .py we use tokenize instead.
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return
    for i, line in enumerate(raw, start=1):
        m = LINE_COMMENT_RE.search(line)
        if not m:
            continue
        tok = m.group("tok")
        text = m.group("text")
        yield Match(path=path, line=i, comment=f"{tok} {text}".rstrip())


def scan(root: Path, exts: Sequence[str], exclude_dirs: Sequence[str]) -> list[Match]:
    matches: list[Match] = []
    for p in iter_files(root, exts=exts, exclude_dirs=exclude_dirs):
        if p.suffix.lower() == ".py":
            comments = extract_python_comments(p)
        else:
            comments = extract_generic_line_comments(p)

        for m in comments:
            c = m.comment.strip()
            # Normalize token
            if c.startswith("#"):
                body = c[1:].lstrip()
            elif c.startswith("//"):
                body = c[2:].lstrip()
            else:
                body = c.lstrip()

            # Skip already-localized markers (# ES:, # EN:, # JP:, # JA:)
            if is_language_marker_comment(body):
                continue

            # Only report if it looks like Spanish.
            if looks_like_spanish(body):
                matches.append(m)

    return matches


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Root folder to scan")
    ap.add_argument("--ext", action="append", default=None, help="File extension to include (repeatable)")
    ap.add_argument("--exclude-dir", action="append", default=None, help="Directory name to exclude (repeatable)")
    ap.add_argument("--out", default="spanish_comments_without_es_marker.txt", help="Write report to this file (UTF-8)")
    ap.add_argument("--max", type=int, default=5000, help="Max matches to include in the report (total)")
    ap.add_argument("--print", action="store_true", help="Also print full report to stdout (can be huge)")
    args = ap.parse_args(list(argv))

    # Ensure we can print Unicode on Windows consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

    root = Path(args.root).resolve()
    exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in (args.ext or DEFAULT_EXTS)]
    exclude_dirs = args.exclude_dir or DEFAULT_EXCLUDE_DIRS

    matches = scan(root=root, exts=exts, exclude_dirs=exclude_dirs)

    # Group by file
    by_file: dict[Path, list[Match]] = {}
    for m in matches:
        by_file.setdefault(m.path, []).append(m)

    total = len(matches)
    out_path = Path(args.out).resolve()

    def format_report() -> str:
        lines: list[str] = []
        lines.append(f"Total Spanish comments without ES/EN/JP marker: {total}")
        if total == 0:
            return "\n".join(lines) + "\n"

        printed = 0
        for path in sorted(by_file.keys(), key=lambda p: str(p).lower()):
            file_matches = sorted(by_file[path], key=lambda m: m.line)
            lines.append("")
            lines.append(str(path))
            for m in file_matches:
                c = m.comment.replace("\t", "    ")
                if len(c) > 240:
                    c = c[:237] + "..."
                lines.append(f"  L{m.line}: {c}")
                printed += 1
                if printed >= args.max:
                    remaining = total - printed
                    if remaining > 0:
                        lines.append("")
                        lines.append(
                            f"... truncated, {remaining} more matches not shown (use --max to increase) ..."
                        )
                    return "\n".join(lines) + "\n"

        return "\n".join(lines) + "\n"

    report = format_report()
    out_path.write_text(report, encoding="utf-8", errors="backslashreplace")

    if args.print:
        print(report)
    else:
        # Keep stdout ASCII-ish to avoid Windows console encoding issues.
        print(f"Wrote report: {out_path.name} (total: {total})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

