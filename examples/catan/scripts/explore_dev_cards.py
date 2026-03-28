#!/usr/bin/env python3
"""Explore colonist.io dev card state via Chrome DevTools Protocol.

Usage: python explore_dev_cards.py

Requires a running colonist game with CDP on port 9223 (SSH tunnel).
Uses only stdlib — no pip dependencies.
"""

import json
import hashlib
import struct
import socket
import ssl
import urllib.request
import sys
import os

CDP_BASE = "http://localhost:9223"


def websocket_handshake(sock, host, path):
    """Minimal WebSocket handshake (RFC 6455)."""
    key = "dGhlIHNhbXBsZSBub25jZQ=="  # static key is fine for local CDP
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    )
    sock.sendall(req.encode())
    # Read response headers
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += sock.recv(4096)
    if b"101" not in resp.split(b"\r\n")[0]:
        raise RuntimeError(f"WebSocket handshake failed: {resp.decode()}")


def ws_send(sock, data: str):
    """Send a WebSocket text frame (masked, as required for client)."""
    payload = data.encode()
    frame = bytearray()
    frame.append(0x81)  # FIN + text
    length = len(payload)
    mask_key = os.urandom(4)
    if length < 126:
        frame.append(0x80 | length)  # MASK bit set
    elif length < 65536:
        frame.append(0x80 | 126)
        frame.extend(struct.pack(">H", length))
    else:
        frame.append(0x80 | 127)
        frame.extend(struct.pack(">Q", length))
    frame.extend(mask_key)
    masked = bytearray(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    frame.extend(masked)
    sock.sendall(frame)


def ws_recv(sock) -> str:
    """Receive a WebSocket text frame. Handles fragmentation for large payloads."""
    full_payload = bytearray()

    while True:
        # Read frame header
        header = b""
        while len(header) < 2:
            header += sock.recv(2 - len(header))

        fin = header[0] & 0x80
        opcode = header[0] & 0x0F
        masked = header[1] & 0x80
        length = header[1] & 0x7F

        if length == 126:
            raw = b""
            while len(raw) < 2:
                raw += sock.recv(2 - len(raw))
            length = struct.unpack(">H", raw)[0]
        elif length == 127:
            raw = b""
            while len(raw) < 8:
                raw += sock.recv(8 - len(raw))
            length = struct.unpack(">Q", raw)[0]

        if masked:
            mask_key = b""
            while len(mask_key) < 4:
                mask_key += sock.recv(4 - len(mask_key))

        payload = bytearray()
        while len(payload) < length:
            chunk = sock.recv(min(65536, length - len(payload)))
            if not chunk:
                raise RuntimeError("Connection closed")
            payload.extend(chunk)

        if masked:
            payload = bytearray(b ^ mask_key[i % 4] for i, b in enumerate(payload))

        full_payload.extend(payload)

        if fin:
            break

    return full_payload.decode()


def cdp_eval(ws_url: str, expression: str) -> dict:
    """Evaluate JS expression via CDP WebSocket."""
    # Parse ws://host:port/path
    url = ws_url.replace("ws://", "")
    host_port, path = url.split("/", 1)
    host, port = host_port.split(":")
    port = int(port)

    sock = socket.create_connection((host, port), timeout=10)
    try:
        websocket_handshake(sock, host_port, "/" + path)
        msg = json.dumps(
            {
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {"expression": expression, "returnByValue": True},
            }
        )
        ws_send(sock, msg)
        result = json.loads(ws_recv(sock))
        return result
    finally:
        sock.close()


# JS that dumps the full mechanicDevelopmentCardsState for all players
DUMP_DEV_STATE_JS = r"""(() => {
    // First find localColor
    let localColor = null;
    try {
        let me = JSON.parse(localStorage.getItem('userState'))?.username;
        if (me) {
            let seen = new Set();
            for (let el of document.querySelectorAll('*')) {
                let fk = Object.keys(el).find(k => k.startsWith('__reactFiber'));
                if (!fk) continue;
                let node = el[fk];
                for (let d = 0; d < 50 && node; d++) {
                    if (seen.has(node)) { node = node.return; continue; }
                    seen.add(node);
                    let p = node.memoizedProps;
                    if (p && p.gameValidator && p.gameValidator.userStates) {
                        let match = p.gameValidator.userStates.find(u => u.username === me);
                        if (match) localColor = match.selectedColor;
                        break;
                    }
                    node = node.return;
                }
                if (localColor !== null) break;
            }
        }
    } catch(e) {}

    // Now walk hooks for mechanicDevelopmentCardsState
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
                    let state = v.mechanicDevelopmentCardsState;
                    let dump = {
                        top_level_keys: Object.keys(state),
                        localColor: localColor,
                    };

                    // Dump each player's full dev card state
                    if (state.players) {
                        dump.players = {};
                        for (let [color, playerState] of Object.entries(state.players)) {
                            let ps = {
                                _keys: Object.keys(playerState),
                            };
                            for (let [k, val] of Object.entries(playerState)) {
                                try {
                                    ps[k] = JSON.parse(JSON.stringify(val));
                                } catch(e) {
                                    ps[k] = String(val);
                                }
                            }
                            if (playerState.developmentCards) {
                                ps._dc_keys = Object.keys(playerState.developmentCards);
                                let dc = playerState.developmentCards;
                                ps._dc_full = {};
                                for (let [k2, v2] of Object.entries(dc)) {
                                    try {
                                        ps._dc_full[k2] = JSON.parse(JSON.stringify(v2));
                                    } catch(e) {
                                        ps._dc_full[k2] = String(v2);
                                    }
                                }
                            }
                            dump.players[color] = ps;
                        }
                    }

                    // Top-level fields besides players
                    for (let [k, val] of Object.entries(state)) {
                        if (k !== 'players') {
                            try {
                                dump['top_' + k] = JSON.parse(JSON.stringify(val));
                            } catch(e) {
                                dump['top_' + k] = String(val);
                            }
                        }
                    }

                    return JSON.stringify(dump, null, 2);
                }
                ms = ms.next;
            }
            node = node.return;
        }
    }
    return JSON.stringify({error: "mechanicDevelopmentCardsState not found"});
})()"""


def main():
    print("Connecting to CDP...")
    try:
        resp = urllib.request.urlopen(f"{CDP_BASE}/json", timeout=3)
        tabs = json.loads(resp.read())
    except Exception as e:
        print(f"Cannot connect to CDP at {CDP_BASE}: {e}")
        print("Make sure SSH tunnel is running:")
        print("  ssh -N -L 9223:127.0.0.1:9222 cullback@192.168.64.1")
        sys.exit(1)

    colonist_tabs = [t for t in tabs if "colonist" in t.get("url", "").lower()]
    if not colonist_tabs:
        print("No colonist tab found. Available tabs:")
        for t in tabs:
            print(f"  - {t.get('title', '?')}: {t.get('url', '?')}")
        sys.exit(1)

    tab = colonist_tabs[0]
    ws_url = tab["webSocketDebuggerUrl"]
    print(f"Tab: {tab.get('title', '?')}")
    print(f"WS:  {ws_url}")
    print()

    result = cdp_eval(ws_url, DUMP_DEV_STATE_JS)

    if "result" in result and "result" in result["result"]:
        val = result["result"]["result"].get("value", "")
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                print(json.dumps(parsed, indent=2))
            except json.JSONDecodeError:
                print(f"Raw: {val}")
        else:
            print(f"Value: {val}")
    else:
        print(f"Response: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
