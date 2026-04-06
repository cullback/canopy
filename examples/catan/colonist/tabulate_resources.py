#!/usr/bin/env python3
"""Extract raw colonist game log via CDP, dump each entry with resource annotation.

Connects to colonist.io, extracts the full game log, and for each entry:
- Shows the type, player, and key fields
- For resource events (type 47): shows raw cards and parsed resources
- Tracks running hand totals per player
- Flags when a player goes negative

Usage: python3 tabulate_resources.py [--port 9223]
"""

import json, http.client, sys, random

CARD = {1: "L", 2: "B", 3: "W", 4: "G", 5: "O"}
COLOR = {1: "Red", 2: "Blue", 3: "Orange", 4: "White", 5: "Green"}


def cards_str(arr):
    return " ".join(CARD.get(c, f"?{c}") for c in arr) if arr else "(none)"


def hand_str(h):
    parts = [f"{v}{r}" for r, v in sorted(h.items()) if v != 0]
    return " ".join(parts) if parts else "(empty)"


def get_log(port):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request("GET", "/json")
    tabs = json.loads(conn.getresponse().read())
    ws_url = next(
        t["webSocketDebuggerUrl"] for t in tabs if "colonist.io" in t.get("url", "")
    )

    import websocket

    ws = websocket.create_connection(ws_url, suppress_origin=True)
    js = r"""(() => {
        function isLogArray(v) {
            return Array.isArray(v) && v.length > 0 && v[0]?.text?.type != null;
        }
        function searchProps(p) {
            if (!p) return null;
            for (let key of Object.keys(p)) {
                if (isLogArray(p[key])) return p[key];
            }
            return null;
        }
        let vs = document.querySelector('[class*="virtualScroller"]');
        if (vs) {
            let fk = Object.keys(vs).find(k => k.startsWith('__reactFiber'));
            if (fk) {
                let node = vs[fk];
                for (let d = 0; d < 30 && node; d++) {
                    let found = searchProps(node.memoizedProps);
                    if (found) return JSON.stringify(found);
                    node = node.return;
                }
            }
        }
        let seen = new Set();
        for (let el of document.querySelectorAll('*')) {
            let fk = Object.keys(el).find(k => k.startsWith('__reactFiber'));
            if (!fk) continue;
            let node = el[fk];
            for (let d = 0; d < 50 && node; d++) {
                if (seen.has(node)) { node = node.return; continue; }
                seen.add(node);
                let p = node.memoizedProps;
                if (p?.gameValidator) {
                    let gv = p.gameValidator;
                    let found = searchProps(gv);
                    if (found) return JSON.stringify(found);
                    if (gv.gameState) {
                        found = searchProps(gv.gameState);
                        if (found) return JSON.stringify(found);
                        for (let key of Object.keys(gv.gameState)) {
                            let v = gv.gameState[key];
                            if (v && typeof v === 'object' && !Array.isArray(v)) {
                                found = searchProps(v);
                                if (found) return JSON.stringify(found);
                            }
                        }
                    }
                    break;
                }
                node = node.return;
            }
        }
        if (vs) {
            let fk = Object.keys(vs).find(k => k.startsWith('__reactFiber'));
            if (fk) {
                let n = vs[fk].return?.return;
                let children = n?.memoizedProps?.children;
                if (Array.isArray(children)) {
                    return JSON.stringify(children.map(c => c?.props?.gameLogData).filter(Boolean));
                }
            }
        }
        return '[]';
    })()"""
    mid = random.randint(1, 999999)
    ws.send(
        json.dumps(
            {
                "id": mid,
                "method": "Runtime.evaluate",
                "params": {"expression": js, "returnByValue": True},
            }
        )
    )
    while True:
        resp = json.loads(ws.recv())
        if resp.get("id") == mid:
            break
    ws.close()
    return json.loads(resp["result"]["result"]["value"])


def main():
    port = 9223
    for i, a in enumerate(sys.argv[1:]):
        if a == "--port":
            port = int(sys.argv[i + 2])

    entries = get_log(port)
    print(f"# {len(entries)} entries\n")

    hands = {}  # color -> {resource_char: count}
    turn = 0

    def ensure(c):
        if c and c not in hands:
            hands[c] = {r: 0 for r in "LBWGO"}

    def add(c, res_dict):
        ensure(c)
        for r, v in res_dict.items():
            if r in hands[c]:
                hands[c][r] += v

    def sub(c, res_dict):
        ensure(c)
        for r, v in res_dict.items():
            if r in hands[c]:
                hands[c][r] -= v

    def parse_cards(arr):
        """Convert card enum array to resource dict."""
        d = {r: 0 for r in "LBWGO"}
        for c in arr or []:
            r = CARD.get(c)
            if r:
                d[r] += 1
        return d

    for idx, entry in enumerate(entries):
        text = entry.get("text", {})
        typ = text.get("type")
        pc = text.get("playerColor", entry.get("from"))
        ensure(pc)
        pname = COLOR.get(pc, f"c{pc}") if pc else "?"

        # Type 47: Resources received
        if typ == 47:
            cards_raw = text.get("cardsToBroadcast", [])
            dist = text.get("distributionType", 1)
            res = parse_cards(cards_raw)
            kind = "STARTING" if dist == 0 else "GOT"
            add(pc, res)
            res_s = " ".join(f"{v}{r}" for r, v in res.items() if v > 0)
            print(
                f"[{idx:3d}] t={typ:3d} {pname:>6s} {kind}: {res_s}  cards={cards_raw}  hand={hand_str(hands.get(pc, {}))}"
            )

        # Type 5: Build
        elif typ == 5:
            piece = text.get("pieceEnum")
            costs = {
                0: ("road", {"L": 1, "B": 1}),
                2: ("settle", {"L": 1, "B": 1, "W": 1, "G": 1}),
                3: ("city", {"G": 2, "O": 3}),
            }
            name, cost = costs.get(piece, (f"p{piece}", {}))
            sub(pc, cost)
            h = hands.get(pc, {})
            flag = " ***NEG***" if any(v < 0 for v in h.values()) else ""
            print(
                f"[{idx:3d}] t={typ:3d} {pname:>6s} BUILD {name} (-{hand_str(cost)})  hand={hand_str(h)}{flag}"
            )

        # Type 10: Roll
        elif typ == 10:
            d1 = text.get("firstDice", 0)
            d2 = text.get("secondDice", 0)
            turn += 1
            print(
                f"\n[{idx:3d}] t={typ:3d} {pname:>6s} ROLL {d1}+{d2}={d1 + d2}  --- Turn {turn} ---"
            )

        # Type 4: Setup placement
        elif typ == 4:
            piece = text.get("pieceEnum")
            name = {0: "road", 2: "settlement"}.get(piece, f"p{piece}")
            print(f"[{idx:3d}] t={typ:3d} {pname:>6s} PLACE {name}")

        # Type 116: Bank trade
        elif typ == 116:
            gave_raw = text.get("givenCardEnums", [])
            recv_raw = text.get("receivedCardEnums", [])
            gave = parse_cards(gave_raw)
            recv = parse_cards(recv_raw)
            sub(pc, gave)
            add(pc, recv)
            h = hands.get(pc, {})
            flag = " ***NEG***" if any(v < 0 for v in h.values()) else ""
            print(
                f"[{idx:3d}] t={typ:3d} {pname:>6s} TRADE gave={cards_str(gave_raw)} got={cards_str(recv_raw)}  hand={hand_str(h)}{flag}"
            )

        # Type 14: Stole (victim perspective)
        elif typ == 14:
            cards_raw = text.get("cardEnums", [])
            victim = text.get("victimColor")
            res = parse_cards(cards_raw)
            if victim:
                ensure(victim)
            add(pc, res)
            if victim:
                sub(victim, res)
            vname = COLOR.get(victim, f"c{victim}") if victim else "?"
            print(
                f"[{idx:3d}] t={typ:3d} {pname:>6s} STOLE {cards_str(cards_raw)} from {vname}  thief={hand_str(hands.get(pc, {}))}  victim={hand_str(hands.get(victim, {}))}"
            )

        # Type 15/139: Stole (robber perspective)
        elif typ in (15, 139):
            robber = entry.get("from")
            recs = entry.get("specificRecipients", [])
            victim = recs[0] if recs else None
            cards_raw = text.get("cardEnums", [])
            res = parse_cards(cards_raw)
            if robber:
                ensure(robber)
            if victim:
                ensure(victim)
            if robber:
                add(robber, res)
            if victim:
                sub(victim, res)
            rname = COLOR.get(robber, f"c{robber}")
            vname = COLOR.get(victim, f"c{victim}") if victim else "?"
            print(
                f"[{idx:3d}] t={typ:3d} {rname:>6s} STOLE {cards_str(cards_raw)} from {vname}  thief={hand_str(hands.get(robber, {}))}  victim={hand_str(hands.get(victim, {}))}"
            )

        # Type 11: Move robber
        elif typ == 11:
            print(f"[{idx:3d}] t={typ:3d} {pname:>6s} MOVE_ROBBER")

        # Type 74: Stole nothing
        elif typ == 74:
            who = entry.get("from")
            wname = COLOR.get(who, f"c{who}") if who else pname
            print(f"[{idx:3d}] t={typ:3d} {wname:>6s} STOLE_NOTHING")

        # Type 60: Rolled seven
        elif typ == 60:
            print(f"[{idx:3d}] t={typ:3d} {pname:>6s} ROLLED_SEVEN")

        # Type 49: Tile blocked
        elif typ == 49:
            ti = text.get("tileInfo", {})
            print(
                f"[{idx:3d}] t={typ:3d}        BLOCKED dice={ti.get('diceNumber')} res={ti.get('resourceType')}"
            )

        # Type 1: Buy dev card
        elif typ == 1:
            cost = {"W": 1, "G": 1, "O": 1}
            sub(pc, cost)
            h = hands.get(pc, {})
            flag = " ***NEG***" if any(v < 0 for v in h.values()) else ""
            print(
                f"[{idx:3d}] t={typ:3d} {pname:>6s} BUY_DEV  hand={hand_str(h)}{flag}"
            )

        # Type 113: Embargo
        elif typ == 113:
            target = text.get("embargoedPlayerColor")
            tname = COLOR.get(target, f"c{target}") if target else "?"
            print(f"[{idx:3d}] t={typ:3d} {pname:>6s} EMBARGO on {tname}")

        # Discard (type varies — check for cardEnums with resource deduction)
        elif typ == 999:  # placeholder — need to find actual discard type
            cards_raw = text.get("cardEnums", [])
            res = parse_cards(cards_raw)
            sub(pc, res)
            print(
                f"[{idx:3d}] t={typ:3d} {pname:>6s} DISCARD {cards_str(cards_raw)}  hand={hand_str(hands.get(pc, {}))}"
            )

        # Type 118: Trade offer
        elif typ == 118:
            print(f"[{idx:3d}] t={typ:3d} {pname:>6s} TRADE_OFFER")

        # Separators
        elif typ in (2, 44):
            pass  # skip

        # Longest road
        elif typ == 66:
            print(f"[{idx:3d}] t={typ:3d} {pname:>6s} LONGEST_ROAD")

        else:
            print(f"[{idx:3d}] t={typ:3d} {pname:>6s} UNKNOWN keys={list(text.keys())}")

    print(f"\n# Final hands:")
    for c in sorted(hands.keys()):
        total = sum(hands[c].values())
        neg = " ***NEG***" if any(v < 0 for v in hands[c].values()) else ""
        print(f"#   {COLOR.get(c, f'c{c}')}: {hand_str(hands[c])} (total={total}){neg}")


if __name__ == "__main__":
    main()
