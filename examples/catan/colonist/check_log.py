#!/usr/bin/env python3
"""Validate resource consistency in a colonist game log.

Reads the game log output (the printed timeline) and tracks each player's
resources from StartingResources, GotResources, builds, trades, and steals.
Reports the first inconsistency or confirms totals match.

Usage: cargo run ... 2>&1 | python3 check_log.py
   or: python3 check_log.py < game_output.txt
"""

import sys
import re

RESOURCES = ["L", "B", "W", "G", "O"]
BUILD_COSTS = {
    "road": {"L": 1, "B": 1},
    "settlement": {"L": 1, "B": 1, "W": 1, "G": 1},
    "city": {"G": 2, "O": 3},
    "dev card": {"W": 1, "G": 1, "O": 1},
}


def parse_resources(s):
    res = {r: 0 for r in RESOURCES}
    for token in s.strip().split():
        if token in res:
            res[token] += 1
    return res


def fmt(hand):
    parts = []
    for r in RESOURCES:
        if hand[r] > 0:
            parts.append(f"{hand[r]}{r}")
    return " ".join(parts) if parts else "(empty)"


def main():
    lines = sys.stdin.readlines()
    players = {}
    current_turn = 0
    trace_player = "Green"
    trace_resource = "W"

    for line in lines:
        line = line.rstrip()

        m = re.match(r"--- Turn (\d+): (\w+) rolls", line)
        if m:
            current_turn = int(m.group(1))
            continue

        m = re.match(r"\s+(\w+) starting: (.+)", line)
        if m:
            name, res = m.group(1), parse_resources(m.group(2))
            if name not in players:
                players[name] = {r: 0 for r in RESOURCES}
            for r, v in res.items():
                players[name][r] += v
            continue

        m = re.match(r"\s+(\w+) got (.+)", line)
        if m and "Longest" not in m.group(2):
            name, res = m.group(1), parse_resources(m.group(2))
            if name in players:
                for r, v in res.items():
                    players[name][r] += v
                if name == trace_player and res.get(trace_resource, 0) > 0:
                    print(
                        f"  T{current_turn}: +{res[trace_resource]}{trace_resource} → {players[name][trace_resource]}{trace_resource}"
                    )
            continue

        m = re.match(r"\s+(\w+) built (\w+)", line)
        if m:
            name, building = m.group(1), m.group(2)
            if name in players and building in BUILD_COSTS:
                cost_w = BUILD_COSTS[building].get(trace_resource, 0)
                for r, v in BUILD_COSTS[building].items():
                    players[name][r] -= v
                    if players[name][r] < 0:
                        print(
                            f"ERROR Turn {current_turn}: {name} negative {r}={players[name][r]} after {building}"
                        )
                        print(f"  Hand: {fmt(players[name])}")
                        return
                if name == trace_player and cost_w > 0:
                    print(
                        f"  T{current_turn}: -{cost_w}{trace_resource} ({building}) → {players[name][trace_resource]}{trace_resource}"
                    )
            continue

        m = re.match(r"\s+(\w+) bought dev card", line)
        if m:
            name = m.group(1)
            if name in players:
                for r, v in BUILD_COSTS["dev card"].items():
                    players[name][r] -= v
                    if players[name][r] < 0:
                        print(
                            f"ERROR Turn {current_turn}: {name} negative {r}={players[name][r]} after dev card"
                        )
                        print(f"  Hand: {fmt(players[name])}")
                        return
            continue

        m = re.match(r"\s+(\w+) bank trade: (.+?) (?:→|→) (.+)", line)
        if m:
            name = m.group(1)
            gave, got = parse_resources(m.group(2)), parse_resources(m.group(3))
            if name in players:
                for r, v in gave.items():
                    players[name][r] -= v
                for r, v in got.items():
                    players[name][r] += v
                if name == trace_player and (
                    gave.get(trace_resource, 0) > 0 or got.get(trace_resource, 0) > 0
                ):
                    print(
                        f"  T{current_turn}: trade {trace_resource} → {players[name][trace_resource]}{trace_resource}"
                    )
                for r in RESOURCES:
                    if players[name][r] < 0:
                        print(
                            f"ERROR Turn {current_turn}: {name} negative {r}={players[name][r]} after trade"
                        )
                        print(f"  Hand: {fmt(players[name])}")
                        return
            continue

        # Debug: check if line looks like a bank trade but didn't match
        if "bank trade" in line:
            print(f"  UNMATCHED TRADE: {repr(line)}")
            continue

        m = re.match(r"\s+(\w+) stole (.+) from (\w+)", line)
        if m:
            stealer, res, victim = m.group(1), parse_resources(m.group(2)), m.group(3)
            if stealer in players and victim in players:
                for r, v in res.items():
                    players[stealer][r] += v
                    players[victim][r] -= v
                    if players[victim][r] < 0:
                        print(
                            f"ERROR Turn {current_turn}: {victim} negative {r}={players[victim][r]} after steal"
                        )
                        print(f"  {victim} hand: {fmt(players[victim])}")
                        return
            continue

        m = re.match(r"\s+(\w+): (.+) \(total: (\d+)\)", line)
        if m:
            name = m.group(1)
            expected_total = int(m.group(3))
            if name in players:
                actual = sum(players[name].values())
                status = "OK" if actual == expected_total else "MISMATCH"
                print(
                    f"{status} {name}: tracked={actual} reported={expected_total} {fmt(players[name])}"
                )

    if not players:
        print("No player data found in input")


if __name__ == "__main__":
    main()
