#!/usr/bin/env python3
"""
ES: Escanea archivos de código para encontrar mensajes de consola en castellano (print/logs/Write-Host) y genera un reporte.
EN: Scan code files to find Spanish console messages (print/logs/Write-Host) and generate a report.
JP: コード内のコンソール出力（print/log/Write-Host）からスペイン語メッセージを抽出し、レポートを生成します。
"""

from __future__ import annotations

import argparse
import ast
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
]

DEFAULT_EXCLUDE_DIRS = [
    ".git",
    ".cursor",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    "installer",
    "build_logs",
    # Common venv layout on Windows
    "Lib",
    "Scripts",
    "share",
]

# Spanish indicators
SPANISH_CHARS_RE = re.compile(r"[áéíóúüñ¿¡ÁÉÍÓÚÜÑ]")
SPANISH_STRONG_WORDS_RE = re.compile(
    r"\b("
    r"importando|iniciando|creando|mostrando|leyendo|buscando|encontrad[oa]|"
    r"validando|validaci[oó]n|convirtiendo|eliminando|limpiando|"
    r"archivo|carpeta|ruta|directorio|proyecto|"
    r"predicci[oó]n|an[aá]lisis|configuraci[oó]n|resultad[oa]s|modelo|datos|"
    r"usuario|cancel[oó]|cancelando|"
    r"reporte|revis[ae]|m[aá]s|"
    r"error|errores|advertencia|aviso|"
    r"no se|no hay|no existe|no encontrado"
    r")\b",
    re.IGNORECASE,
)
SPANISH_WEAK_WORDS_RE = re.compile(
    r"\b("
    r"el|la|los|las|un|una|unos|unas|"
    r"no|si|sino|pero|porque|"
    r"por|para|con|sin|"
    r"de|del|al|"
    r"se|ha|han"
    r")\b",
    re.IGNORECASE,
)

# Avoid treating mostly CJK-only strings as Spanish
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def looks_like_spanish(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if "�" in t:
        return False
    if SPANISH_CHARS_RE.search(t) or re.search(r"[¿¡]", t):
        return True

    strong = [s for s in SPANISH_STRONG_WORDS_RE.findall(t) if s]
    weak = [w for w in SPANISH_WEAK_WORDS_RE.findall(t) if w]

    strong_norm = {s.strip().lower() for s in strong}
    if strong_norm and strong_norm.issubset({"error", "errores"}):
        return False

    # If it is mostly CJK and has no strong Spanish signals, ignore
    if CJK_RE.search(t) and len(strong) == 0:
        return False

    if len(strong) >= 1:
        return True
    if len(weak) >= 3 and len(t.split()) <= 12:
        return True

    return False


def iter_files(root: Path, exts: Sequence[str], exclude_dirs: Sequence[str]) -> Iterator[Path]:
    exclude = set(exclude_dirs)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in exts:
                yield p


@dataclass(frozen=True)
class Match:
    path: Path
    line: int
    kind: str
    message: str
    code: str


def _string_from_expr(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.JoinedStr):
        parts: list[str] = []
        for v in expr.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            else:
                parts.append("{...}")
        return "".join(parts)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left = _string_from_expr(expr.left)
        right = _string_from_expr(expr.right)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        # "x {}".format(y) -> capture template
        if expr.func.attr == "format":
            base = _string_from_expr(expr.func.value)
            return base
    return None


def _call_kind(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name) and f.id == "print":
        return "print"
    if isinstance(f, ast.Attribute):
        # sys.stdout.write(...)
        if f.attr == "write" and isinstance(f.value, ast.Attribute):
            if isinstance(f.value.value, ast.Name) and f.value.value.id == "sys":
                if f.value.attr in ("stdout", "stderr"):
                    return f"sys.{f.value.attr}.write"
        # logger.info(...) / logging.info(...)
        if f.attr in ("debug", "info", "warning", "error", "critical", "exception"):
            if isinstance(f.value, ast.Name) and f.value.id == "logging":
                return f"logging.{f.attr}"
            # any_logger.info(...)
            return f"logger.{f.attr}"
    return None


def scan_python_file(path: Path) -> list[Match]:
    try:
        with tokenize.open(str(path)) as f:
            src = f.read()
    except (OSError, UnicodeError):
        return []

    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []

    lines = src.splitlines()
    matches: list[Match] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kind = _call_kind(node)
        if not kind:
            continue
        if not node.args:
            continue

        msg = _string_from_expr(node.args[0])
        if msg is None:
            continue
        if not looks_like_spanish(msg):
            continue

        lineno = getattr(node, "lineno", 1)
        code_line = lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else ""
        matches.append(
            Match(
                path=path,
                line=lineno,
                kind=kind,
                message=msg.replace("\t", "    "),
                code=code_line.replace("\t", "    "),
            )
        )

    return matches


PS1_CALL_RE = re.compile(
    r"^\s*(?P<cmd>Write-Host|Write-Output|Write-Error|Write-Warning)\s+(?P<arg>.+?)\s*$",
    re.IGNORECASE,
)
PS1_STRING_RE = re.compile(r"""(?P<q>['"])(?P<s>.*?)(?P=q)""")


def scan_ps1_file(path: Path) -> list[Match]:
    try:
        raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []

    matches: list[Match] = []
    for i, line in enumerate(raw_lines, start=1):
        m = PS1_CALL_RE.match(line)
        if not m:
            continue
        cmd = m.group("cmd")
        arg = m.group("arg")
        sm = PS1_STRING_RE.search(arg)
        if not sm:
            continue
        msg = sm.group("s")
        if not looks_like_spanish(msg):
            continue
        matches.append(
            Match(
                path=path,
                line=i,
                kind=cmd,
                message=msg.replace("\t", "    "),
                code=line.strip().replace("\t", "    "),
            )
        )
    return matches


def scan(root: Path, exts: Sequence[str], exclude_dirs: Sequence[str]) -> list[Match]:
    matches: list[Match] = []
    for p in iter_files(root=root, exts=exts, exclude_dirs=exclude_dirs):
        if p.suffix.lower() == ".py":
            matches.extend(scan_python_file(p))
        elif p.suffix.lower() == ".ps1":
            matches.extend(scan_ps1_file(p))
        else:
            continue
    return matches


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Root folder to scan")
    ap.add_argument("--ext", action="append", default=None, help="File extension to include (repeatable)")
    ap.add_argument("--exclude-dir", action="append", default=None, help="Directory name to exclude (repeatable)")
    ap.add_argument("--out", default="spanish_console_messages_es.txt", help="Write report to this file (UTF-8)")
    args = ap.parse_args(list(argv))

    # Ensure we can print Unicode on Windows consoles
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

    root = Path(args.root).resolve()
    exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in (args.ext or DEFAULT_EXTS)]
    exclude_dirs = args.exclude_dir or DEFAULT_EXCLUDE_DIRS

    matches = scan(root=root, exts=exts, exclude_dirs=exclude_dirs)

    by_file: dict[Path, list[Match]] = {}
    for m in matches:
        by_file.setdefault(m.path, []).append(m)

    unique_messages = sorted({m.message for m in matches}, key=lambda s: s.lower())

    out_path = Path(args.out).resolve()

    def format_report() -> str:
        lines: list[str] = []
        lines.append(f"Root: {root}")
        lines.append(f"Total matches: {len(matches)}")
        lines.append(f"Unique messages: {len(unique_messages)}")
        lines.append("")
        lines.append("== Matches (by file) ==")

        for path in sorted(by_file.keys(), key=lambda p: str(p).lower()):
            file_matches = sorted(by_file[path], key=lambda m: (m.line, m.kind))
            rel = path.relative_to(root) if path.is_absolute() and str(path).startswith(str(root)) else path
            lines.append("")
            lines.append(str(rel))
            for m in file_matches:
                msg = m.message.replace("\r", "")
                code = m.code.replace("\r", "")
                if len(msg) > 240:
                    msg = msg[:237] + "..."
                if len(code) > 260:
                    code = code[:257] + "..."
                lines.append(f"  L{m.line} [{m.kind}] {msg}")
                lines.append(f"    {code}")

        lines.append("")
        lines.append("== Unique messages (sorted) ==")
        for s in unique_messages:
            t = s.replace("\r", "")
            if len(t) > 300:
                t = t[:297] + "..."
            lines.append(f"- {t}")

        lines.append("")
        return "\n".join(lines)

    out_path.write_text(format_report(), encoding="utf-8", errors="backslashreplace")
    print(f"Wrote report: {out_path.name} (matches: {len(matches)}, unique: {len(unique_messages)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

