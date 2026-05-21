#!/usr/bin/env python3
"""Split a combined Q&A/DEF file into separate ex<N> and def<N> files.

Usage:
    python3 split_qa.py path/to/whole [output_dir]

A new block starts at any line beginning with "Q:" or "DEF:".
"Q:" blocks -> ex1, ex2, ...   "DEF:" blocks -> def1, def2, ...
Numbered in order of appearance. Output dir defaults to the input's dir.
"""
import sys
from pathlib import Path


def split(text):
    blocks = []          # list of (kind, lines)
    cur = None
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("Q:"):
            cur = ("ex", [])
            blocks.append(cur)
        elif stripped.startswith("DEF:"):
            cur = ("def", [])
            blocks.append(cur)
        if cur is not None:
            cur[1].append(line)
    return blocks


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: split_qa.py path/to/whole [output_dir]")
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = split(src.read_text())
    counters = {"ex": 0, "def": 0}
    written = []
    for kind, lines in blocks:
        counters[kind] += 1
        name = f"{kind}{counters[kind]}"
        body = "\n".join(lines).strip() + "\n"
        (out_dir / name).write_text(body)
        written.append(name)

    print(f"{src} -> {len(written)} files in {out_dir}: {', '.join(written)}")


if __name__ == "__main__":
    main()
