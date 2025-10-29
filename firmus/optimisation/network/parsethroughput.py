#!/usr/bin/env python3
"""
Parse SGLang offline throughput from .out logs and build a table.
Extracts ONLY "Total token throughput (tok/s)" for each file.

Usage:
  python parse_sglang_throughput.py /path/to/net_sweep

Outputs:
  - throughput_table.csv
  - throughput_table.md
Prints the Markdown table to stdout.
"""

import sys
import re
from pathlib import Path
import csv

THR_PATTERNS = [
    re.compile(r"Total token throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)"),  # primary
]

# Example filename: t1_ARing_PLL_T64_C4_M1_R0_S1_G0.out
NAME_RE = re.compile(
    r"""^t(?P<trial>\d+)_A(?P<algo>[^_]+)_P(?P<proto>[^_]+)_T(?P<T>\d+)_C(?P<C>\d+)_M(?P<M>\d+)_R(?P<R>\d+)_S(?P<S>\d+)_G(?P<G>\d+)\.out$"""
)

HEADERS = [
    "sweep",
    "trial",
    "algo",
    "proto",
    "T",
    "C",
    "M",
    "R",
    "S",
    "G",
    "total_token_throughput_tok_s",
    "logfile",
]

def extract_total_thr(text: str) -> float | None:
    for pat in THR_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None

def parse_name(fname: str):
    m = NAME_RE.match(fname)
    if not m:
        return None
    d = m.groupdict()
    return {
        "trial": int(d["trial"]),
        "algo": d["algo"],
        "proto": d["proto"],
        "T": int(d["T"]),
        "C": int(d["C"]),
        "M": int(d["M"]),
        "R": int(d["R"]),
        "S": int(d["S"]),
        "G": int(d["G"]),
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_sglang_throughput.py /path/to/net_sweep", file=sys.stderr)
        sys.exit(2)

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.exists():
        print(f"Path not found: {root}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for sweep_dir in sorted(root.glob("net_sweep_*")):
        logs_dir = sweep_dir / "logs"
        if not logs_dir.is_dir():
            continue
        for out_file in sorted(logs_dir.glob("*.out")):
            parsed = parse_name(out_file.name)
            if not parsed:
                continue
            try:
                text = out_file.read_text(errors="ignore")
            except Exception:
                continue

            thr = extract_total_thr(text)
            # Only include entries where the "Total token throughput" line exists
            if thr is None:
                continue

            rows.append({
                "sweep": sweep_dir.name,
                **parsed,
                "total_token_throughput_tok_s": thr,
                "logfile": str(out_file),
            })

    # Sort by sweep then trial
    rows.sort(key=lambda r: (r["sweep"], r["trial"]))

    # Write CSV
    csv_path = Path("throughput_table.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in HEADERS})

    # Write Markdown
    md_path = Path("throughput_table.md")
    with md_path.open("w") as f:
        # header
        f.write("| " + " | ".join(HEADERS) + " |\n")
        f.write("|" + "|".join(["---"] * len(HEADERS)) + "|\n")
        # rows
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k, "")) for k in HEADERS) + " |\n")

    # Print Markdown table to stdout for quick copy
    with md_path.open("r") as f:
        print(f.read())

    print(f"\nWrote CSV: {csv_path.resolve()}")
    print(f"Wrote Markdown: {md_path.resolve()}")

if __name__ == "__main__":
    main()

