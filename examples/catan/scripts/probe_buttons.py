#!/usr/bin/env python3
"""Probe colonist.io for button elements via CDP."""

import json
import urllib.request
import socket
import struct
import base64
import re

CDP_PORT = 9223

with urllib.request.urlopen(f"http://localhost:{CDP_PORT}/json") as resp:
    tabs = json.loads(resp.read())

ws_url = None
for tab in tabs:
    if "colonist" in tab.get("url", "").lower():
        ws_url = tab.get("webSocketDebuggerUrl")
        break
if not ws_url:
    print("No colonist tab found")
    exit(1)


def ws_connect(url):
    m = re.match(r"wss?://([^:/]+):?(\d+)?(/.*)?", url)
    host, port, path = m.group(1), int(m.group(2) or 80), m.group(3) or "/"
    sock = socket.create_connection((host, port))
    key = base64.b64encode(b"probe-buttons-key!").decode()
    sock.sendall(
        f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode()
    )
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += sock.recv(4096)
    return sock


def ws_send(sock, text):
    data = text.encode("utf-8")
    frame = bytearray([0x81])
    mask_key = b"\x00\x00\x00\x00"
    if len(data) < 126:
        frame.append(0x80 | len(data))
    elif len(data) < 65536:
        frame.append(0x80 | 126)
        frame.extend(struct.pack(">H", len(data)))
    else:
        frame.append(0x80 | 127)
        frame.extend(struct.pack(">Q", len(data)))
    frame.extend(mask_key)
    frame.extend(data)
    sock.sendall(frame)


def ws_recv(sock):
    def read_exact(n):
        buf = b""
        while len(buf) < n:
            buf += sock.recv(n - len(buf))
        return buf

    header = read_exact(2)
    masked = header[1] & 0x80
    length = header[1] & 0x7F
    if length == 126:
        length = struct.unpack(">H", read_exact(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", read_exact(8))[0]
    mask = read_exact(4) if masked else None
    payload = read_exact(length)
    if mask:
        payload = bytes(payload[i] ^ mask[i % 4] for i in range(length))
    return payload.decode("utf-8")


def evaluate(sock, expr, msg_id=1):
    ws_send(
        sock,
        json.dumps(
            {
                "id": msg_id,
                "method": "Runtime.evaluate",
                "params": {"expression": expr, "returnByValue": True},
            }
        ),
    )
    while True:
        data = json.loads(ws_recv(sock))
        if data.get("id") == msg_id:
            result = data.get("result", {}).get("result", {})
            if result.get("type") == "undefined" or "exceptionDetails" in data.get(
                "result", {}
            ):
                print("JS error:", json.dumps(data.get("result"), indent=2))
                return None
            return result.get("value")


sock = ws_connect(ws_url)

# Focus: find the action buttons area and dump its structure
js = r"""(() => {
    try {
        let results = [];
        // Search for elements with action/dice/roll/endTurn in class
        let selectors = '[class*="action" i], [class*="dice" i], [class*="endTurn" i], [class*="roll" i], [class*="turn" i]';
        let els = document.querySelectorAll(selectors);
        for (let el of els) {
            let rect = el.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) continue;
            let cls = el.className;
            if (typeof cls !== 'string') cls = cls?.baseVal || '';
            results.push({
                tag: el.tagName,
                cls: cls.substring(0, 200),
                text: (el.textContent || '').trim().substring(0, 40),
                rect: [Math.round(rect.x), Math.round(rect.y), Math.round(rect.width), Math.round(rect.height)],
                childCount: el.children.length,
            });
        }
        return JSON.stringify(results);
    } catch(e) {
        return JSON.stringify({error: e.message});
    }
})()"""

value = evaluate(sock, js)
if value:
    items = json.loads(value)
    if isinstance(items, dict) and "error" in items:
        print("Error:", items["error"])
    else:
        for item in items:
            print(
                f"{item['tag']:6} {item['rect']}  children={item['childCount']}  text={item['text']!r}"
            )
            print(f"       {item['cls']}")
            print()
        print(f"{len(items)} elements")

# Also check turnState from React state
js2 = r"""(() => {
    try {
        let seen = new Set();
        for (let el of document.querySelectorAll('*')) {
            let fk = Object.keys(el).find(k => k.startsWith('__reactFiber'));
            if (!fk) continue;
            let node = el[fk];
            for (let d = 0; d < 50 && node; d++) {
                if (seen.has(node)) { node = node.return; continue; }
                seen.add(node);
                let ms = node.memoizedState;
                for (let i = 0; i < 30 && ms; i++) {
                    let v = ms.memoizedState;
                    if (v && typeof v === 'object' && v.currentState && v.diceState) {
                        return JSON.stringify({
                            turnPlayerColor: v.currentState.currentTurnPlayerColor,
                            turnState: v.currentState.turnState,
                            turnStateKeys: Object.keys(v.currentState),
                            diceThrown: v.diceState.diceThrown,
                            diceValues: v.diceState.diceValues,
                            diceStateKeys: Object.keys(v.diceState),
                            topLevelKeys: Object.keys(v),
                        });
                    }
                    ms = ms.next;
                }
                node = node.return;
            }
        }
        return JSON.stringify({found: false});
    } catch(e) {
        return JSON.stringify({error: e.message});
    }
})()"""

value2 = evaluate(sock, js2, msg_id=2)
if value2:
    print("\n=== React turnState ===")
    print(json.dumps(json.loads(value2), indent=2))

sock.close()
