#!/usr/bin/env python3
"""Inspect a colonist.io board via Chrome DevTools Protocol.

Dumps tile and port data, detects the D6 orientation, and determines which
port configuration (PORT_SPECS vs PORT_SPECS_ALT) the board uses.

Requires: Chrome on Mac with --remote-debugging-port=9222, SSH tunnel to
VM port 9223.  See memory/cdp-connection.md for setup.

No external dependencies — uses a minimal stdlib WebSocket client.
"""

import hashlib
import json
import os
import socket
import struct
import sys
import urllib.request
from base64 import b64encode

CDP_ENDPOINT = "http://localhost:9223/json"

# --- Minimal WebSocket client (RFC 6455) --------------------------------------


def ws_connect(url):
    """Open a WebSocket connection (no TLS). Returns raw socket."""
    assert url.startswith("ws://")
    rest = url[5:]
    host_port, path = rest.split("/", 1)
    path = "/" + path
    if ":" in host_port:
        host, port = host_port.split(":")
        port = int(port)
    else:
        host, port = host_port, 80

    sock = socket.create_connection((host, port))
    key = b64encode(os.urandom(16)).decode()
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host_port}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    )
    sock.sendall(req.encode())
    # Read HTTP response headers
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += sock.recv(4096)
    return sock


def ws_send(sock, text):
    """Send a text frame."""
    payload = text.encode()
    mask_key = os.urandom(4)
    # Build frame: FIN + text opcode
    header = b"\x81"
    length = len(payload)
    if length < 126:
        header += struct.pack("B", 0x80 | length)
    elif length < 65536:
        header += struct.pack("!BH", 0x80 | 126, length)
    else:
        header += struct.pack("!BQ", 0x80 | 127, length)
    header += mask_key
    # Mask payload
    masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    sock.sendall(header + masked)


def ws_recv(sock):
    """Receive one text/binary frame (handles continuation). Returns str."""

    def read_exact(n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("WebSocket closed")
            buf += chunk
        return buf

    result = b""
    while True:
        b0, b1 = struct.unpack("BB", read_exact(2))
        fin = b0 & 0x80
        opcode = b0 & 0x0F
        masked = b1 & 0x80
        length = b1 & 0x7F
        if length == 126:
            length = struct.unpack("!H", read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", read_exact(8))[0]
        if masked:
            mask_key = read_exact(4)
            data = bytes(b ^ mask_key[i % 4] for i, b in enumerate(read_exact(length)))
        else:
            data = read_exact(length)
        result += data
        if fin:
            break
    return result.decode()


# --- Hex math -----------------------------------------------------------------

DIRECTION_VECTORS = {
    "E": (1, 0),
    "SE": (0, 1),
    "SW": (-1, 1),
    "W": (-1, 0),
    "NW": (0, -1),
    "NE": (1, -1),
}


def hex_neighbor(q, r, direction):
    dq, dr = DIRECTION_VECTORS[direction]
    return (q + dq, r + dr)


# --- Board constants ----------------------------------------------------------

LAND_HEXES = [
    (0, -2),
    (1, -2),
    (2, -2),
    (2, -1),
    (2, 0),
    (1, 1),
    (0, 2),
    (-1, 2),
    (-2, 2),
    (-2, 1),
    (-2, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, 0),
]

TOKEN_SEQUENCE = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]

PORT_SPECS = [
    ((3, 0), "W"),
    ((1, 2), "NW"),
    ((-1, 3), "NW"),
    ((-3, 3), "NE"),
    ((-3, 1), "E"),
    ((-2, -1), "E"),
    ((0, -3), "SE"),
    ((2, -3), "SW"),
    ((3, -2), "SW"),
]

PORT_SPECS_ALT = [
    ((1, -3), "SE"),
    ((3, -3), "SW"),
    ((3, -1), "W"),
    ((2, 1), "W"),
    ((0, 3), "NW"),
    ((-2, 3), "NE"),
    ((-3, 2), "NE"),
    ((-3, 0), "E"),
    ((-1, -2), "SE"),
]


# --- D6 transforms -----------------------------------------------------------


def cube_rotate_cw(q, r):
    s = -q - r
    return (-r, -s)


def cube_reflect(q, r):
    s = -q - r
    return (q, s)


def apply_transform(q, r, rotation, reflect):
    if reflect:
        q, r = cube_reflect(q, r)
    for _ in range(rotation):
        q, r = cube_rotate_cw(q, r)
    return (q, r)


# --- CDP helpers --------------------------------------------------------------

EXTRACT_JS = r"""(() => {
    let gv = null;
    let seen = new Set();
    for (let el of document.querySelectorAll('*')) {
        let fk = Object.keys(el).find(k => k.startsWith('__reactFiber'));
        if (!fk) continue;
        let node = el[fk];
        for (let d = 0; d < 50 && node; d++) {
            if (seen.has(node)) { node = node.return; continue; }
            seen.add(node);
            let p = node.memoizedProps;
            if (p && p.gameValidator && !gv) gv = p.gameValidator;
            if (gv) break;
            node = node.return;
        }
        if (gv) break;
    }
    if (!gv) return '{}';
    let ts = gv.mapValidator?.tileState;
    let tiles = ts?._tiles?.map(t => t.state);
    let ports = gv.mapValidator?.portState?._portEdges?.map(p => p.state);
    return JSON.stringify({ tiles, ports });
})()"""


def cdp_eval(sock, expr, cmd_id=1):
    ws_send(
        sock,
        json.dumps(
            {
                "id": cmd_id,
                "method": "Runtime.evaluate",
                "params": {"expression": expr, "returnByValue": True},
            }
        ),
    )
    while True:
        resp = json.loads(ws_recv(sock))
        if resp.get("id") == cmd_id:
            return resp["result"]["result"].get("value")


# --- Main ---------------------------------------------------------------------


def detect_orientation(tiles):
    hex_to_land = {h: i for i, h in enumerate(LAND_HEXES)}
    for reflect in (False, True):
        for rotation in range(6):
            numbers_by_index = [None] * 19
            land_count = 0
            for t in tiles:
                mx, my = apply_transform(t["x"], t["y"], rotation, reflect)
                if (mx, my) in hex_to_land:
                    idx = hex_to_land[(mx, my)]
                    land_count += 1
                    dn = t.get("diceNumber", 0) or 0
                    if dn > 0:
                        numbers_by_index[idx] = dn
            if land_count != 19:
                continue
            numbers = [n for n in numbers_by_index if n is not None]
            if numbers == list(TOKEN_SEQUENCE):
                return rotation, reflect
    return None, None


def match_ports(ports, rotation, reflect, specs, label):
    water_set = {}
    land_set = {}
    for i, (whex, direction) in enumerate(specs):
        water_set[whex] = i
        land = hex_neighbor(*whex, direction)
        land_set[land] = i

    matched = 0
    unmatched = []
    for p in ports:
        transformed = apply_transform(p["x"], p["y"], rotation, reflect)
        idx = water_set.get(transformed)
        if idx is None:
            idx = land_set.get(transformed)
        if idx is not None:
            matched += 1
        else:
            unmatched.append((p["x"], p["y"]))

    return matched, unmatched


def main():
    print("Fetching CDP targets...")
    targets = json.loads(urllib.request.urlopen(CDP_ENDPOINT).read())
    page = next(
        (t for t in targets if t["type"] == "page" and "colonist" in t.get("url", "")),
        None,
    )
    if not page:
        page = next((t for t in targets if t["type"] == "page"), None)
    if not page:
        print("No page target found")
        sys.exit(1)

    ws_url = page["webSocketDebuggerUrl"]
    print(f"Connecting to {page.get('title', 'unknown')}...")
    sock = ws_connect(ws_url)

    raw = cdp_eval(sock, EXTRACT_JS)
    sock.close()

    if not raw or raw == "{}":
        print("No board data found (not in a game?)")
        sys.exit(1)

    data = json.loads(raw)
    tiles = data.get("tiles", [])
    ports = data.get("ports", [])

    # Print tiles
    print(f"\n--- {len(tiles)} tiles ---")
    terrain_names = {
        0: "Desert",
        1: "Lumber",
        2: "Brick",
        3: "Wool",
        4: "Grain",
        5: "Ore",
    }
    for t in sorted(tiles, key=lambda t: (t["y"], t["x"])):
        tn = terrain_names.get(t.get("type", -1), f"?{t.get('type')}")
        dn = t.get("diceNumber", 0) or 0
        print(f"  ({t['x']:2},{t['y']:2})  {tn:7s}  dice={dn}")

    # Print ports
    print(f"\n--- {len(ports)} ports ---")
    port_names = {1: "3:1", 2: "Lumber", 3: "Brick", 4: "Wool", 5: "Grain", 6: "Ore"}
    for p in ports:
        pn = port_names.get(p.get("type", -1), f"?{p.get('type')}")
        print(f"  ({p['x']:2},{p['y']:2},{p['z']})  {pn}")

    # Detect orientation
    rotation, reflect = detect_orientation(tiles)
    if rotation is None:
        print("\nCould not detect orientation!")
        sys.exit(1)

    label = f"R{rotation}" + (" reflected" if reflect else "")
    print(f"\nOrientation: {label}")

    # Transform port coords
    print(f"\nTransformed port coords:")
    for p in ports:
        raw = (p["x"], p["y"])
        tx, ty = apply_transform(p["x"], p["y"], rotation, reflect)
        pn = port_names.get(p.get("type", -1), f"?{p.get('type')}")
        print(f"  ({raw[0]:2},{raw[1]:2}) -> ({tx:2},{ty:2})  z={p['z']}  {pn}")

    # Match against both port spec sets
    m1, u1 = match_ports(ports, rotation, reflect, PORT_SPECS, "PORT_SPECS")
    m2, u2 = match_ports(ports, rotation, reflect, PORT_SPECS_ALT, "PORT_SPECS_ALT")

    print(f"\nPORT_SPECS:     {m1}/{len(ports)} matched")
    if u1:
        print(f"  unmatched: {u1}")
    print(f"PORT_SPECS_ALT: {m2}/{len(ports)} matched")
    if u2:
        print(f"  unmatched: {u2}")

    if m1 > m2:
        print("\n=> Board uses PRIMARY port configuration (PORT_SPECS)")
    elif m2 > m1:
        print("\n=> Board uses ALTERNATE port configuration (PORT_SPECS_ALT)")
    else:
        print("\n=> Ambiguous (same number of matches)")

    # Show which spec each port matched
    winner_specs = PORT_SPECS if m1 >= m2 else PORT_SPECS_ALT
    winner_name = "PORT_SPECS" if m1 >= m2 else "PORT_SPECS_ALT"
    water_set = {}
    land_set = {}
    for i, (whex, direction) in enumerate(winner_specs):
        water_set[whex] = i
        land = hex_neighbor(*whex, direction)
        land_set[land] = i

    print(f"\nPort detail ({winner_name}):")
    for p in ports:
        raw = (p["x"], p["y"])
        transformed = apply_transform(p["x"], p["y"], rotation, reflect)
        pn = port_names.get(p.get("type", -1), f"?{p.get('type')}")
        idx = water_set.get(transformed)
        if idx is None:
            idx = land_set.get(transformed)
        if idx is not None:
            whex, d = winner_specs[idx]
            lhex = hex_neighbor(*whex, d)
            print(
                f"  [{idx}] ({raw[0]:2},{raw[1]:2}) z={p['z']} -> water=({whex[0]:2},{whex[1]:2}) dir={d:2s} land=({lhex[0]:2},{lhex[1]:2})  {pn}"
            )
        else:
            print(f"  [?] ({raw[0]:2},{raw[1]:2}) z={p['z']}  {pn}  UNMATCHED")


if __name__ == "__main__":
    main()
