#!/usr/bin/env python3
"""Summarize observed ViT graph adapter boundaries from a UART log.

The replacement report identifies each unique adapter. Its ordinal is a
label, not a claim about LLVM execution order. A missing END means the return
marker was not observed; it does not diagnose a stalled Triton kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


MARKER = re.compile(
    r"\[vit-op\] ([0-9]+)/([0-9]+) symbol=([A-Za-z0-9_]+) "
    r"case=([A-Za-z0-9_]+) (BEGIN|END)"
)
CYCLE_MARKER = re.compile(
    r"\[vit-op-cycle\] \[vit-op\] ([0-9]+)/([0-9]+) symbol=([A-Za-z0-9_]+) "
    r"case=([A-Za-z0-9_]+) (BEGIN|END) cycles=0x([0-9A-Fa-f]+)"
)


def load_calls(path: Path) -> dict[int, dict]:
    report = json.loads(path.read_text())
    records = report.get("replaced")
    if report.get("status") != "PASS" or not isinstance(records, list) or not records:
        raise ValueError("replacement report must contain a complete graph")
    count = len(records)
    if report.get("graphs", {}).get("forward", {}).get("external_calls") != count:
        raise ValueError("replacement report graph call count differs from its records")
    if report.get("operator_trace", {}).get("call_count") != count:
        raise ValueError("replacement report has no matching operator trace inventory")
    calls = {}
    symbols = set()
    for ordinal, record in enumerate(records, 1):
        if (record.get("ordinal") != ordinal or
                not isinstance(record.get("symbol"), str) or
                not isinstance(record.get("case"), str) or
                record["symbol"] in symbols):
            raise ValueError(f"invalid or duplicate replacement record {ordinal}")
        symbols.add(record["symbol"])
        calls[ordinal] = {key: record[key] for key in
                          ("ordinal", "node", "family", "symbol", "case")}
    return calls


def summarize(log: str, calls: dict[int, dict], cycle_hz: float | None = None) -> dict:
    """Parse complete UART lines only; accept any graph call order."""
    complete = log if log.endswith("\n") else log.rpartition("\n")[0]
    pending = []
    unpaired_ends = []
    errors = []
    started = set()
    completed_ordinals = set()
    last_begin = last_end = last_completed = None
    begin_count = end_count = completed_count = 0
    cycle_events = {}
    timing_errors = []
    for line_number, raw_line in enumerate(complete.splitlines(), 1):
        # The UART relay can place an ASCII framing byte immediately before a
        # line (currently ``0x1f`` before some END markers).  It is transport
        # metadata, not part of the marker, so discard leading C0 controls
        # before applying the strict marker grammar.
        line = re.sub(r"^[\x00-\x1f]+", "", raw_line.rstrip("\r"))
        cycle = CYCLE_MARKER.fullmatch(line)
        if cycle is not None:
            ordinal, total = int(cycle[1]), int(cycle[2])
            expected = calls.get(ordinal)
            if (total != len(calls) or expected is None or
                    (cycle[3], cycle[4]) != (expected["symbol"], expected["case"])):
                timing_errors.append(f"cycle marker disagrees with replacement report at UART line {line_number}")
                continue
            key = (ordinal, cycle[5])
            if key in cycle_events:
                timing_errors.append(f"duplicate cycle marker for ordinal {ordinal} phase {cycle[5]} at UART line {line_number}")
            cycle_events[key] = int(cycle[6], 16)
            continue
        match = MARKER.fullmatch(line)
        if match is None:
            if line.startswith("[vit-op]"):
                errors.append(f"malformed operator marker at UART line {line_number}")
            continue
        ordinal, total = int(match[1]), int(match[2])
        expected = calls.get(ordinal)
        if (total != len(calls) or expected is None or
                (match[3], match[4]) != (expected["symbol"], expected["case"])):
            errors.append(f"marker disagrees with replacement report at UART line {line_number}")
            continue
        event = {**expected, "uart_line": line_number}
        if match[5] == "BEGIN":
            begin_count += 1
            last_begin = event
            if pending:
                errors.append(f"new BEGIN before previous END at UART line {line_number}")
            if ordinal in started:
                errors.append(f"duplicate BEGIN for ordinal {ordinal} at UART line {line_number}")
            started.add(ordinal)
            pending.append(event)
        else:
            end_count += 1
            last_end = event
            matching = next((index for index in range(len(pending) - 1, -1, -1)
                             if pending[index]["ordinal"] == ordinal), None)
            if matching is None:
                unpaired_ends.append(event)
                errors.append(f"END without matching BEGIN for {expected['symbol']} "
                              f"({expected['case']}) at UART line {line_number}")
                continue
            if matching != len(pending) - 1:
                errors.append(f"END out of adapter call order at UART line {line_number}")
            pending.pop(matching)
            completed_count += 1
            completed_ordinals.add(ordinal)
            last_completed = event
    timings = []
    for ordinal in sorted(calls):
        begin = cycle_events.get((ordinal, "BEGIN"))
        end = cycle_events.get((ordinal, "END"))
        if begin is None or end is None:
            continue
        cycles = (end - begin) & ((1 << 64) - 1)
        item = {**calls[ordinal], "begin_cycles": begin, "end_cycles": end,
                "elapsed_cycles": cycles}
        if cycle_hz is not None:
            item["elapsed_seconds"] = cycles / cycle_hz
        timings.append(item)
    errors.extend(timing_errors)
    status = ("TRACE_SEQUENCE_ERROR" if errors else
              "NO_MARKERS" if not begin_count and not end_count else
              "ALL_CALLS_RETURNED" if len(completed_ordinals) == len(calls) and
              completed_count == len(calls) and not pending else
              "PROGRESS_VALID_SO_FAR")
    return {
        "status": status,
        "expected_calls": len(calls),
        "observed_begin_markers": begin_count,
        "observed_end_markers": end_count,
        "completed_calls": completed_count,
        "distinct_completed_calls": len(completed_ordinals),
        "last_begin": last_begin,
        "last_end": last_end,
        "last_completed": last_completed,
        "unpaired_begin": pending[-1] if pending else None,
        "unpaired_begins": pending,
        "unpaired_ends": unpaired_ends,
        "errors": errors,
        "cycle_markers": len(cycle_events),
        "operator_timings": timings,
        "cycle_hz": cycle_hz,
        "limits": [
            "Only complete UART lines are counted; a trailing partial line is ignored.",
            "An unpaired BEGIN means its END was not observed. External UVHS interruption or UART loss can cause this; it does not prove an operator deadlock.",
            "Ordinal values identify report entries; observed UART order determines progress.",
            "Operator markers do not prove numerical correctness or full model completion.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uart-log", type=Path, required=True)
    parser.add_argument("--replacement-report", "--replacement", dest="replacement_report",
                        type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cycle-hz", type=float,
                        default=float(os.environ.get("UV_RUN_SYS_CLK_HZ", "5529600")),
                        help="RA cycle frequency used for seconds (default: UV_RUN_SYS_CLK_HZ or 5529600)")
    args = parser.parse_args()
    try:
        calls = load_calls(args.replacement_report)
        raw = args.uart_log.read_bytes()
        result = summarize(raw.decode(errors="replace"), calls, args.cycle_hz)
        result.update({
            "uart_log": str(args.uart_log),
            "uart_bytes": len(raw),
            "uart_sha256": hashlib.sha256(raw).hexdigest(),
            "replacement_report": str(args.replacement_report),
            "replacement_sha256": hashlib.sha256(args.replacement_report.read_bytes()).hexdigest(),
            "trailing_partial_line_ignored": bool(raw and not raw.endswith(b"\n")),
        })
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
