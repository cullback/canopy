#!/usr/bin/env python3
"""Poll colonist.io game state every second and log changes."""

import json
import urllib.request
import socket
import struct
import base64
import re
import time

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


msg_id = 0


def evaluate(sock, expr):
    global msg_id
    msg_id += 1
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
            if "exceptionDetails" in data.get("result", {}):
                return None
            return result.get("value")


JS = r"""(() => {
    try {
        let seen = new Set();
        let liveGs = null;
        let localColor = null;

        try {
            let me = JSON.parse(localStorage.getItem('userState'))?.username;
        } catch {}

        for (let el of document.querySelectorAll('*')) {
            let fk = Object.keys(el).find(k => k.startsWith('__reactFiber'));
            if (!fk) continue;
            let node = el[fk];
            for (let d = 0; d < 50 && node; d++) {
                if (seen.has(node)) { node = node.return; continue; }
                seen.add(node);

                // Get local color from props
                if (!localColor) {
                    let p = node.memoizedProps;
                    if (p && p.gameValidator && p.gameValidator.userStates) {
                        try {
                            let me = JSON.parse(localStorage.getItem('userState'))?.username;
                            if (me) {
                                let match = p.gameValidator.userStates.find(u => u.username === me);
                                if (match) localColor = match.selectedColor;
                            }
                        } catch {}
                    }
                }

                if (!liveGs) {
                    let ms = node.memoizedState;
                    for (let i = 0; i < 30 && ms; i++) {
                        let v = ms.memoizedState;
                        if (v && typeof v === 'object' && v.currentState && v.diceState) {
                            liveGs = v;
                            break;
                        }
                        ms = ms.next;
                    }
                }
                if (liveGs && localColor) break;
                node = node.return;
            }
            if (liveGs && localColor) break;
        }

        // Check button states via CSS classes
        let dicePulsing = !!document.querySelector('[class*="pulsing"]');
        let turnBtn = document.querySelector('[class*="turnButton"]');
        let turnBtnDisabled = false;
        if (turnBtn) {
            let style = getComputedStyle(turnBtn);
            // Check if it has a disabled-looking class or opacity
            turnBtnDisabled = turnBtn.classList.toString().includes('disabled') ||
                              style.opacity < 0.5 ||
                              style.pointerEvents === 'none';
        }

        // Check action button container text for clues
        let actionContainer = document.querySelector('[class*="actionButtonContainer"]');
        let actionText = actionContainer?.textContent?.trim().substring(0, 80) || '';

        // Count log entries
        let vs = document.querySelector('[class*="virtualScroller"]');
        let logCount = vs ? vs.children.length : 0;

        let cs = liveGs?.currentState;
        let ds = liveGs?.diceState;

        return JSON.stringify({
            turnColor: cs?.currentTurnPlayerColor ?? null,
            turnState: cs?.turnState ?? null,
            actionState: cs?.actionState ?? null,
            localColor,
            diceThrown: ds?.diceThrown ?? null,
            dice: ds ? [ds.dice1, ds.dice2] : null,
            dicePulsing,
            turnBtnExists: !!turnBtn,
            turnBtnDisabled,
            actionText,
            logCount,
            robberTile: liveGs?.mechanicRobberState?.locationTileIndex ?? null,
        });
    } catch(e) {
        return JSON.stringify({error: e.message});
    }
})()"""

sock = ws_connect(ws_url)
print("Polling... play 2-3 turns then stop with Ctrl+C")
print()

prev = None
t0 = time.time()
try:
    while True:
        value = evaluate(sock, JS)
        if value:
            state = json.loads(value)
            # Only print when something changed
            if state != prev:
                elapsed = time.time() - t0
                is_local = state.get("turnColor") == state.get("localColor")
                who = "LOCAL" if is_local else "OPP"
                print(
                    f"[{elapsed:6.1f}s] {who} turnState={state.get('turnState')} "
                    f"actionState={state.get('actionState')} "
                    f"diceThrown={state.get('diceThrown')} "
                    f"dice={state.get('dice')} "
                    f"dicePulsing={state.get('dicePulsing')} "
                    f"turnBtn={state.get('turnBtnExists')}/{state.get('turnBtnDisabled')} "
                    f"logCount={state.get('logCount')} "
                    f"actionText={state.get('actionText')!r}"
                )
                prev = state
        time.sleep(1)
except KeyboardInterrupt:
    print("\nDone.")
finally:
    sock.close()
