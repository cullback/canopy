#!/usr/bin/env python3
"""
Connect to Chrome via CDP on port 9223 and extract
mechanicDevelopmentCardsState player object keys from colonist.io.
"""

import json
import sys
import urllib.request

import websocket

CDP_PORT = 9223

JS_EXPR = r"""
(() => {
    let debug = {};

    // Step 1: find local player color from React props
    let localColor = null;
    try {
        let raw = localStorage.getItem('userState');
        let me = raw ? JSON.parse(raw)?.username : null;
        debug.username = me;
        if (me) {
            let seen = new Set();
            outer:
            for (let el of document.querySelectorAll('*')) {
                let fk = Object.keys(el).find(k => k.startsWith('__reactFiber'));
                if (!fk) continue;
                let node = el[fk];
                for (let d = 0; d < 50 && node; d++) {
                    if (seen.has(node)) { node = node.return; continue; }
                    seen.add(node);
                    let p = node.memoizedProps;
                    if (p && p.players) {
                        // players could be array or object
                        let arr = Array.isArray(p.players) ? p.players : Object.values(p.players);
                        for (let pl of arr) {
                            if (pl && pl.username === me) {
                                localColor = pl.color;
                                debug.matchedColor = localColor;
                                break outer;
                            }
                        }
                    }
                    // Also check for localPlayer prop directly
                    if (p && p.localPlayer && p.localPlayer.username === me) {
                        localColor = p.localPlayer.color;
                        debug.matchedColor = localColor;
                        break outer;
                    }
                    node = node.return;
                }
            }
        }
    } catch(e) { debug.colorError = e.message; }
    debug.localColor = localColor;

    // Step 2: find mechanicDevelopmentCardsState
    let seen2 = new Set();
    for (let el of document.querySelectorAll('*')) {
        let fk = Object.keys(el).find(k => k.startsWith('__reactFiber'));
        if (!fk) continue;
        let node = el[fk];
        for (let d = 0; d < 50 && node; d++) {
            if (seen2.has(node)) { node = node.return; continue; }
            seen2.add(node);
            let ms = node.memoizedState;
            for (let i = 0; i < 30 && ms; i++) {
                let v = ms.memoizedState;
                if (v && v.mechanicDevelopmentCardsState) {
                    debug.foundDevCardState = true;
                    let ps = v.mechanicDevelopmentCardsState.players;
                    debug.devCardPlayersKeys = ps ? Object.keys(ps) : null;

                    // Try localColor, then string version, then first available
                    let playerData = null;
                    let usedKey = null;
                    if (localColor != null && ps[localColor]) {
                        playerData = ps[localColor];
                        usedKey = String(localColor);
                    } else if (localColor != null && ps[String(localColor)]) {
                        playerData = ps[String(localColor)];
                        usedKey = String(localColor);
                    } else if (ps) {
                        let firstKey = Object.keys(ps)[0];
                        playerData = ps[firstKey];
                        usedKey = firstKey;
                        debug.fallback = "used first player key";
                    }

                    // Also grab all players for comparison
                    let allPlayers = {};
                    if (ps) {
                        for (let [k, pv] of Object.entries(ps)) {
                            allPlayers[k] = {
                                keys: Object.keys(pv),
                                summary: pv
                            };
                        }
                    }

                    return JSON.stringify({
                        debug: debug,
                        usedKey: usedKey,
                        keys: playerData ? Object.keys(playerData) : [],
                        sample: playerData,
                        allPlayers: allPlayers
                    });
                }
                ms = ms.next;
            }
            node = node.return;
        }
    }

    debug.foundDevCardState = false;
    return JSON.stringify({debug: debug});
})()
"""


def main():
    # 1. Discover the colonist.io tab
    url = f"http://localhost:{CDP_PORT}/json"
    print(f"Fetching tab list from {url} ...")
    with urllib.request.urlopen(url, timeout=5) as resp:
        tabs = json.loads(resp.read())

    ws_url = None
    for tab in tabs:
        title = tab.get("title", "")
        tab_url = tab.get("url", "")
        if "colonist" in tab_url.lower() or "colonist" in title.lower():
            ws_url = tab.get("webSocketDebuggerUrl")
            print(f"Found tab: {title!r}  url={tab_url}")
            break

    if not ws_url:
        print("No colonist.io tab found. Available tabs:")
        for t in tabs:
            print(f"  {t.get('title', '?')!r}  {t.get('url', '?')}")
        sys.exit(1)

    print(f"Connecting to {ws_url} ...")

    # 2. Connect via WebSocket
    ws = websocket.create_connection(
        ws_url,
        timeout=10,
        suppress_origin=True,
    )

    # 3. Send Runtime.evaluate
    msg = json.dumps(
        {
            "id": 1,
            "method": "Runtime.evaluate",
            "params": {
                "expression": JS_EXPR,
                "returnByValue": True,
            },
        }
    )
    ws.send(msg)

    # 4. Read response (skip any events)
    while True:
        raw = ws.recv()
        data = json.loads(raw)
        if data.get("id") == 1:
            break

    ws.close()

    # 5. Parse and display
    result = data.get("result", {}).get("result", {})
    value = result.get("value", "{}")

    exc = data.get("result", {}).get("exceptionDetails")
    if exc:
        print("JS exception:")
        print(json.dumps(exc, indent=2))
        sys.exit(1)

    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value

    # Debug info
    debug = parsed.get("debug", {})
    print(f"\nUsername: {debug.get('username')}")
    print(f"Local color: {debug.get('localColor')}")
    if debug.get("fallback"):
        print(f"Fallback: {debug['fallback']}")

    if not debug.get("foundDevCardState", True):
        print("\nNo mechanicDevelopmentCardsState found in React tree.")
        print("Is there an active game in the tab?")
        sys.exit(1)

    print(f"Dev card player keys: {debug.get('devCardPlayersKeys')}")
    print(f"Used key: {parsed.get('usedKey')}")

    # Keys
    keys = parsed.get("keys", [])
    print(f"\n=== Player object keys ({len(keys)}) ===")
    for key in keys:
        print(f"  - {key}")

    # Sample data
    print("\n=== Sample player data ===")
    print(json.dumps(parsed.get("sample", {}), indent=2))

    # All players overview
    all_players = parsed.get("allPlayers", {})
    if len(all_players) > 1:
        print(f"\n=== All players ({len(all_players)}) ===")
        for pk, pv in all_players.items():
            print(f"\n  Player key={pk}:")
            print(f"    Keys: {pv.get('keys')}")
            print(f"    Data: {json.dumps(pv.get('summary', {}), indent=6)}")


if __name__ == "__main__":
    main()
