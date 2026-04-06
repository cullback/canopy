#!/usr/bin/env python3
"""Debug topology mapping: extract colonist board via CDP, compute our topology,
and check which tiles are adjacent to each settlement for starting resources."""

import json
import http.client
import websocket  # pip install websocket-client

CDP_PORT = 9223


def discover_tab():
    conn = http.client.HTTPConnection("127.0.0.1", CDP_PORT)
    conn.request("GET", "/json")
    resp = conn.getresponse()
    tabs = json.loads(resp.read())
    for tab in tabs:
        if "colonist.io" in tab.get("url", "") and tab.get("type") == "page":
            return tab["webSocketDebuggerUrl"]
    raise RuntimeError(f"no colonist.io tab among {len(tabs)} tabs")


def evaluate(ws, expr):
    import random

    msg_id = random.randint(1, 999999)
    ws.send(
        json.dumps(
            {
                "id": msg_id,
                "method": "Runtime.evaluate",
                "params": {"expression": expr, "returnByValue": True},
            }
        )
    )
    while True:
        resp = json.loads(ws.recv())
        if resp.get("id") == msg_id:
            if "error" in resp:
                raise RuntimeError(f"CDP error: {resp['error']}")
            return resp["result"]["result"]["value"]


# Canonical LAND_HEXES (same as topology.rs)
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

# Hex directions: NE, E, SE, SW, W, NW (axial offsets)
DIR_OFFSETS = {
    "NE": (1, -1),
    "E": (1, 0),
    "SE": (0, 1),
    "SW": (-1, 1),
    "W": (-1, 0),
    "NW": (0, -1),
}
DIRS = ["NE", "E", "SE", "SW", "W", "NW"]

# SHARED_CORNERS[corner] = [(direction, neighbor_corner), ...]
SHARED_CORNERS = [
    [("NE", 4), ("NW", 2)],  # corner 0 (N)
    [("NE", 3), ("E", 5)],  # corner 1 (NE)
    [("E", 4), ("SE", 0)],  # corner 2 (SE)
    [("SE", 5), ("SW", 1)],  # corner 3 (S)
    [("SW", 0), ("W", 2)],  # corner 4 (SW)
    [("W", 1), ("NW", 3)],  # corner 5 (NW)
]


def hex_neighbor(q, r, direction):
    dq, dr = DIR_OFFSETS[direction]
    return (q + dq, r + dr)


def map_hex(q, r, rotation, reflect):
    cq, cr, cs = q, r, -q - r
    if reflect:
        cr, cs = cs, cr
    for _ in range(rotation):
        cq, cr, cs = -cr, -cs, -cq
    return (cq, cr)


def rotate_corner(corner, rotation, reflect):
    c = corner
    if reflect:
        c = (8 - c) % 6
    return (c + rotation) % 6


def map_corner(x, y, z, rotation, reflect):
    if rotation == 0 and not reflect:
        return (x, y, z)
    corner = 0 if z == 0 else 3
    rc = rotate_corner(corner, rotation, reflect)
    hx, hy = map_hex(x, y, rotation, reflect)
    if rc == 0:
        return (hx, hy, 0)
    if rc == 3:
        return (hx, hy, 1)
    # Find neighbor where this corner is 0 or 3
    for direction, nc in SHARED_CORNERS[rc]:
        if nc == 0 or nc == 3:
            nq, nr = hex_neighbor(hx, hy, direction)
            nz = 0 if nc == 0 else 1
            return (nq, nr, nz)
    return (hx, hy, z)


def detect_orientation(tiles):
    hex_to_land = {h: i for i, h in enumerate(LAND_HEXES)}
    for reflect in [False, True]:
        for rotation in range(6):
            numbers_by_index = [None] * 19
            land_count = 0
            for t in tiles:
                mx, my = map_hex(t["x"], t["y"], rotation, reflect)
                idx = hex_to_land.get((mx, my))
                if idx is not None:
                    land_count += 1
                    if t["diceNumber"] > 0:
                        numbers_by_index[idx] = t["diceNumber"]
            if land_count != 19:
                continue
            nums = [n for n in numbers_by_index if n is not None]
            if nums == TOKEN_SEQUENCE:
                return rotation, reflect
    return 0, False


def build_corner_map(tiles_by_index, rotation, reflect):
    """Build (hx, hy, z) -> (tile_indices, node_corner_on_tile) map.
    Returns a dict mapping canonical corner coords to the set of adjacent tile indices."""
    hex_to_land = {h: i for i, h in enumerate(LAND_HEXES)}
    # corner_to_tiles: canonical (q,r,z) -> list of tile_index
    corner_to_tiles = {}
    for i, h in enumerate(LAND_HEXES):
        for corner in range(6):
            # Find canonical representation
            cz = {0: 0, 3: 1}.get(corner)
            if cz is not None:
                key = (h[0], h[1], cz)
                corner_to_tiles.setdefault(key, set()).add(i)
            for direction, nc in SHARED_CORNERS[corner]:
                nq, nr = hex_neighbor(h[0], h[1], direction)
                nc_z = {0: 0, 3: 1}.get(nc)
                if nc_z is not None:
                    key = (nq, nr, nc_z)
                    corner_to_tiles.setdefault(key, set()).add(i)
    return corner_to_tiles


def main():
    ws_url = discover_tab()
    print(f"Connected: {ws_url}")
    ws = websocket.create_connection(ws_url, suppress_origin=True)

    # Extract board
    board_js = r"""(() => {
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
        let ts = gv.mapValidator?.tileState?._tiles?.map(t => t.state);
        let corners = gv.mapValidator?.tileState?.tileCornerStates || {};
        // Also get building corners from gameManager
        let gm = null;
        seen = new Set();
        for (let el of document.querySelectorAll('*')) {
            let fk = Object.keys(el).find(k => k.startsWith('__reactFiber'));
            if (!fk) continue;
            let node = el[fk];
            for (let d = 0; d < 50 && node; d++) {
                if (seen.has(node)) { node = node.return; continue; }
                seen.add(node);
                let p = node.memoizedProps;
                if (p?.gameManager && !gm) gm = p.gameManager;
                if (gm) break;
                node = node.return;
            }
            if (gm) break;
        }
        let gmCs = gm?.mapController?.uiGameManager?.gameState?.mapState?.tileState?.tileCornerStates || {};
        return JSON.stringify({tiles: ts, buildingCorners: gmCs});
    })()"""

    result = evaluate(ws, board_js)
    data = json.loads(result)
    tiles = data.get("tiles", [])
    building_corners = data.get("buildingCorners", {})

    if not tiles:
        print("ERROR: no tiles extracted")
        return

    # Detect orientation
    rotation, reflect = detect_orientation(tiles)
    label = f"R{rotation}" + (" reflected" if reflect else "")
    print(f"Detected orientation: {label}")

    # Build tile terrain map
    hex_to_land = {h: i for i, h in enumerate(LAND_HEXES)}
    terrain_by_index = {}
    terrain_names = {
        0: "Desert",
        1: "Lumber",
        2: "Brick",
        3: "Wool",
        4: "Grain",
        5: "Ore",
    }
    for t in tiles:
        mx, my = map_hex(t["x"], t["y"], rotation, reflect)
        idx = hex_to_land.get((mx, my))
        if idx is not None:
            terrain_by_index[idx] = terrain_names.get(t["type"], f"?{t['type']}")
            print(
                f"  tile[{idx:2d}] ({mx:2d},{my:2d}) = {terrain_by_index[idx]:8s} dice={t['diceNumber']}"
                f"  (colonist: {t['x']},{t['y']} type={t['type']})"
            )

    # Build corner→tiles map
    corner_to_tiles = build_corner_map(tiles, rotation, reflect)

    # Extract settlements
    print(f"\n--- Settlements ---")
    for key, entry in building_corners.items():
        owner = entry.get("owner")
        bt = entry.get("buildingType")
        if owner is None or bt not in (1, 2):
            continue
        x = entry["x"]
        y = entry["y"]
        z = entry["z"]
        btype = "settlement" if bt == 1 else "city"
        mx, my, mz = map_corner(x, y, z, rotation, reflect)
        print(
            f"\n  {btype} owner={owner} colonist=({x},{y},{z}) -> canonical=({mx},{my},{mz})"
        )

        # Find adjacent tiles
        adj_key = (mx, my, mz)
        adj_tiles = corner_to_tiles.get(adj_key, set())
        if not adj_tiles:
            print(f"    WARNING: no adjacent tiles found for ({mx},{my},{mz})!")
            # Try to find close matches
            for k, v in corner_to_tiles.items():
                if abs(k[0] - mx) <= 1 and abs(k[1] - my) <= 1:
                    print(f"    nearby corner: {k} -> tiles {v}")
        else:
            print(f"    adjacent tiles ({len(adj_tiles)}):")
            for ti in sorted(adj_tiles):
                terrain = terrain_by_index.get(ti, "???")
                h = LAND_HEXES[ti]
                print(f"      tile[{ti:2d}] ({h[0]:2d},{h[1]:2d}) = {terrain}")

    ws.close()


if __name__ == "__main__":
    main()
