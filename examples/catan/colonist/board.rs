//! Parse the colonist.io board state extracted via Chrome DevTools Protocol.
//!
//! Tiles and ports come from `gameValidator.mapValidator`.  Building and road
//! positions come from `gameManager.mapController.uiGameManager.gameState`
//! which carries `owner`/`buildingType`/`type` on corner and edge states.

use std::collections::HashMap;

use crate::game::board::{EdgeId, NodeId, Terrain as CanopyTerrain};
use crate::game::hex::{Direction, Hex};
use crate::game::resource::Resource;
use crate::game::topology::{
    EDGE_NEIGHBOR_DIR, LAND_HEXES, PORT_SPECS, PORT_SPECS_ALT, SHARED_CORNERS, TOKEN_SEQUENCE,
    Topology,
};

// -- Coordinate orientation mapper --------------------------------------------

/// Maps colonist.io coordinates to our canonical LAND_HEXES orientation.
///
/// Colonist boards may apply the number token spiral starting from any of the
/// 6 corners and in either CW or CCW direction (12 D6 orientations). We detect
/// which orientation is used and apply the inverse transform so the model sees
/// the canonical layout it was trained on.
#[derive(Clone, Debug)]
pub struct CoordMapper {
    pub rotation: u8,
    pub reflect: bool,
}

impl CoordMapper {
    /// Identity mapper (no transformation).
    pub fn identity() -> Self {
        Self {
            rotation: 0,
            reflect: false,
        }
    }

    /// Detect the board orientation by brute-forcing all 12 D6 transforms.
    ///
    /// For each transform, applies it to the colonist tile coords, maps to
    /// LAND_HEXES indices, reads non-desert numbers in index order, and checks
    /// against TOKEN_SEQUENCE.
    pub fn detect(tiles: &[Tile]) -> Self {
        for reflect in [false, true] {
            for rotation in 0..6u8 {
                let mapper = Self { rotation, reflect };
                if mapper.matches_token_sequence(tiles) {
                    let label = if reflect {
                        format!("R{rotation} reflected")
                    } else {
                        format!("R{rotation}")
                    };
                    eprintln!("detected orientation: {label}");
                    return mapper;
                }
            }
        }

        // Debug: show what the identity mapper produces for diagnostics.
        let identity = Self::identity();
        identity.debug_token_order(tiles);

        eprintln!("warning: could not detect board orientation, using identity");
        identity
    }

    fn matches_token_sequence(&self, tiles: &[Tile]) -> bool {
        let hex_to_land: HashMap<(i32, i32), usize> = LAND_HEXES
            .iter()
            .enumerate()
            .map(|(i, h)| ((h.q as i32, h.r as i32), i))
            .collect();

        // Build array: for each LAND_HEXES index, what number token is there?
        let mut numbers_by_index: [Option<u8>; 19] = [None; 19];
        let mut land_count = 0u32;
        for tile in tiles {
            let (mx, my) = self.map_hex(tile.x, tile.y);
            if let Some(&idx) = hex_to_land.get(&(mx, my)) {
                land_count += 1;
                if tile.dice_number > 0 {
                    numbers_by_index[idx] = Some(tile.dice_number);
                }
                // else: desert, leave as None
            }
            // Skip water/port hexes that don't map to LAND_HEXES.
        }

        // All 19 land hexes must be accounted for.
        if land_count != 19 {
            return false;
        }

        // Read non-desert numbers in LAND_HEXES index order, compare to TOKEN_SEQUENCE.
        let numbers: Vec<u8> = numbers_by_index.iter().filter_map(|&n| n).collect();
        numbers.len() == 18 && numbers == TOKEN_SEQUENCE
    }

    fn debug_token_order(&self, tiles: &[Tile]) {
        const SPIRAL_ORDER: [usize; 19] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 16, 15, 14, 13, 18,
        ];

        let hex_to_land: HashMap<(i32, i32), usize> = LAND_HEXES
            .iter()
            .enumerate()
            .map(|(i, h)| ((h.q as i32, h.r as i32), i))
            .collect();

        let mut numbers_by_index: [Option<u8>; 19] = [None; 19];
        let mut land_count = 0u32;
        let mut unmapped = Vec::new();
        for tile in tiles {
            let (mx, my) = self.map_hex(tile.x, tile.y);
            if let Some(&idx) = hex_to_land.get(&(mx, my)) {
                land_count += 1;
                if tile.dice_number > 0 {
                    numbers_by_index[idx] = Some(tile.dice_number);
                }
            } else {
                unmapped.push((tile.x, tile.y, mx, my, tile.dice_number));
            }
        }

        eprintln!(
            "debug detect: land_count={land_count}, unmapped={}",
            unmapped.len()
        );
        for (ox, oy, mx, my, d) in &unmapped {
            eprintln!("  unmapped tile ({ox},{oy}) -> ({mx},{my}) dice={d}");
        }

        eprintln!("numbers_by_index (LAND_HEXES order):");
        for (i, n) in numbers_by_index.iter().enumerate() {
            let hex = LAND_HEXES[i];
            eprintln!("  [{i:2}] ({:2},{:2}) = {:?}", hex.q, hex.r, n);
        }

        let spiral_nums: Vec<String> = SPIRAL_ORDER
            .iter()
            .map(|&idx| match numbers_by_index[idx] {
                Some(n) => format!("{n}"),
                None => "--".to_string(),
            })
            .collect();
        eprintln!("spiral order:    [{}]", spiral_nums.join(", "));
        eprintln!("TOKEN_SEQUENCE:  {:?}", TOKEN_SEQUENCE);

        let index_nums: Vec<String> = numbers_by_index
            .iter()
            .map(|n| match n {
                Some(v) => format!("{v}"),
                None => "--".to_string(),
            })
            .collect();
        eprintln!("index order:     [{}]", index_nums.join(", "));
    }

    /// Apply the D6 transform to a hex coordinate.
    pub fn map_hex(&self, x: i32, y: i32) -> (i32, i32) {
        // Convert axial (q, r) to cube (q, r, s) where s = -q - r
        let (mut q, mut r, mut s) = (x, y, -x - y);

        // Optional mirror reflection: swap r and s (reflects through the q axis,
        // i.e. the N-S axis of the hex grid). This reverses CW↔CCW ring direction.
        if self.reflect {
            std::mem::swap(&mut r, &mut s);
        }

        // Apply rotation: each step is a 60° CW rotation in cube coords
        for _ in 0..self.rotation {
            let new_q = -r;
            let new_r = -s;
            let new_s = -q;
            q = new_q;
            r = new_r;
            s = new_s;
        }
        let _ = s; // axial only needs q, r

        (q, r)
    }

    /// Transform a colonist corner (x, y, z) to canonical coordinates.
    ///
    /// z ∈ {0, 1}: z=0 → corner 0 (N), z=1 → corner 3 (S).
    /// After rotating the hex, the corner index rotates too. We remap back
    /// to z=0/z=1 by finding which shared-corner alias gives a valid canonical
    /// representation.
    pub fn map_corner(&self, x: i32, y: i32, z: u8) -> (i32, i32, u8) {
        if self.rotation == 0 && !self.reflect {
            return (x, y, z);
        }
        let corner = if z == 0 { 0u8 } else { 3u8 };
        let rotated_corner = self.rotate_corner(corner);
        let (hx, hy) = self.map_hex(x, y);

        // If rotated corner is already 0 or 3, we can use it directly
        if rotated_corner == 0 {
            return (hx, hy, 0);
        }
        if rotated_corner == 3 {
            return (hx, hy, 1);
        }

        // Otherwise find an adjacent hex where this corner is 0 or 3
        let hex = Hex::new(hx as i8, hy as i8);
        for &(dir, neighbor_corner) in &SHARED_CORNERS[rotated_corner as usize] {
            if neighbor_corner == 0 || neighbor_corner == 3 {
                let nb = hex.neighbor(dir);
                let nz = if neighbor_corner == 0 { 0 } else { 1 };
                return (nb.q as i32, nb.r as i32, nz);
            }
        }

        // Fallback (shouldn't happen): return as-is with best guess
        (hx, hy, z)
    }

    /// Transform a colonist edge (x, y, z) to canonical coordinates.
    ///
    /// z ∈ {0, 1, 2}: z=0 → edge 5, z=1 → edge 4, z=2 → edge 3.
    /// After rotating the hex, the edge index rotates too. We remap back
    /// to z=0/1/2 by finding a canonical representation.
    pub fn map_edge(&self, x: i32, y: i32, z: u8) -> (i32, i32, u8) {
        if self.rotation == 0 && !self.reflect {
            return (x, y, z);
        }
        let edge = match z {
            0 => 5u8,
            1 => 4u8,
            2 => 3u8,
            _ => return (x, y, z),
        };
        let rotated_edge = self.rotate_edge(edge);
        let (hx, hy) = self.map_hex(x, y);

        // Canonical edges are 3, 4, 5 → z values 2, 1, 0
        if rotated_edge >= 3 {
            let nz = 5 - rotated_edge; // 3→2, 4→1, 5→0
            return (hx, hy, nz);
        }

        // Edge 0-2: use the neighbor hex where this is the opposite edge (3-5)
        let hex = Hex::new(hx as i8, hy as i8);
        let neighbor_dir = EDGE_NEIGHBOR_DIR[rotated_edge as usize];
        let nb = hex.neighbor(neighbor_dir);
        let opposite = (rotated_edge + 3) % 6;
        let nz = 5 - opposite;
        (nb.q as i32, nb.r as i32, nz)
    }

    /// LAND_HEXES index for a transformed hex coordinate, or None.
    pub fn tile_index(&self, x: i32, y: i32) -> Option<usize> {
        let (mx, my) = self.map_hex(x, y);
        LAND_HEXES
            .iter()
            .position(|h| h.q as i32 == mx && h.r as i32 == my)
    }

    fn rotate_corner(&self, corner: u8) -> u8 {
        let mut c = corner;
        if self.reflect {
            // Mirror (q,r,s)→(q,s,r) reflects through corners 1 and 4 (NE-SW axis).
            // Permutation: 0↔2, 3↔5, 1 and 4 fixed.
            c = (8 - c) % 6;
        }
        (c + self.rotation) % 6
    }

    fn rotate_edge(&self, edge: u8) -> u8 {
        let mut e = edge;
        if self.reflect {
            // Mirror (q,r,s)→(q,s,r) on edges: 0↔1, 2↔5, 3↔4.
            e = (7 - e) % 6;
        }
        (e + self.rotation) % 6
    }
}

// -- Extraction JS ------------------------------------------------------------

/// JS snippet to extract board data from the React component tree.
///
/// Searches for two React props:
///   - `gameValidator` — tiles, ports, log, mechanic states (robber)
///   - `gameManager`  — tileCornerStates/tileEdgeStates with owner + building data
pub const EXTRACT_JS: &str = r#"(() => {
    let gv = null, gm = null;
    let seen = new Set();
    for (let el of document.querySelectorAll('*')) {
        let fk = Object.keys(el).find(k => k.startsWith('__reactFiber'));
        if (!fk) continue;
        let node = el[fk];
        for (let d = 0; d < 50 && node; d++) {
            if (seen.has(node)) { node = node.return; continue; }
            seen.add(node);
            let p = node.memoizedProps;
            if (p) {
                if (p.gameValidator && !gv) gv = p.gameValidator;
                if (p.gameManager && !gm) gm = p.gameManager;
            }
            if (gv && gm) break;
            node = node.return;
        }
        if (gv && gm) break;
    }
    if (!gv) return '{}';

    // Tiles + ports from gameValidator
    let ts = gv.mapValidator?.tileState;
    let tiles = ts?._tiles?.map(t => t.state);
    let ports = gv.mapValidator?.portState?._portEdges?.map(p => p.state);

    // Corner/edge states with building data from gameManager
    let gmTs = gm?.mapController?.uiGameManager?.gameState?.mapState?.tileState;
    let corners = gmTs?.tileCornerStates || {};
    let edges = gmTs?.tileEdgeStates || {};

    // Robber: use mechanicRobberState.locationTileIndex to index tiles directly.
    let robber = null;
    let ri = gv.gameState?.mechanicRobberState?.locationTileIndex;
    if (ri != null && tiles && tiles[ri]) {
        robber = { x: tiles[ri].x, y: tiles[ri].y };
    }

    return JSON.stringify({ tiles, ports, corners, edges, robber });
})()"#;

/// JS snippet to extract the local player's dev card state from the React
/// component tree.
///
/// Returns JSON: `{cards: [11,...], bought_this_turn: [14,...]}`.
///
/// Card enums: 11=Knight, 12=VictoryPoint, 13=Monopoly, 14=RoadBuilding, 15=YearOfPlenty.
pub const EXTRACT_CARDS_JS: &str = r#"(() => {
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
    } catch {}

    // Walk hooks for live dev card state.
    // Fields (discovered via CDP dump of mechanicDevelopmentCardsState):
    //   me.developmentCards.cards        — array of card IDs held (11-15)
    //   me.developmentCardsBoughtThisTurn — array of card IDs bought this turn (local player only)
    //   me.developmentCardsUsed          — array of card IDs played this turn
    if (localColor !== null) {
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
                        let ps = v.mechanicDevelopmentCardsState.players;
                        let me = ps && ps[localColor];
                        if (me) {
                            let dc = me.developmentCards || {};
                            let cards = (dc.cards || []).filter(c => c >= 11 && c <= 15);
                            let bought = (me.developmentCardsBoughtThisTurn || []).filter(c => c >= 11 && c <= 15);
                            return JSON.stringify({
                                cards,
                                bought_this_turn: bought,
                            });
                        }
                    }
                    ms = ms.next;
                }
                node = node.return;
            }
        }
    }

    // Fallback: props cardState (may be stale — no bought/played info).
    let seen3 = new Set();
    for (let el of document.querySelectorAll('*')) {
        let fk = Object.keys(el).find(k => k.startsWith('__reactFiber'));
        if (!fk) continue;
        let node = el[fk];
        for (let d = 0; d < 30 && node; d++) {
            if (seen3.has(node)) { node = node.return; continue; }
            seen3.add(node);
            let p = node.memoizedProps;
            if (p && p.cardState && typeof p.cardState === 'object' && !Array.isArray(p.cardState)) {
                let cards = [];
                for (let [k, count] of Object.entries(p.cardState)) {
                    let e = parseInt(k);
                    if (e >= 11 && e <= 15) {
                        for (let i = 0; i < count; i++) cards.push(e);
                    }
                }
                if (cards.length > 0) return JSON.stringify({cards, bought_this_turn: []});
            }
            node = node.return;
        }
    }
    return '{}';
})()"#;

/// JS snippet to extract live game metadata from the React component tree.
///
/// Returns JSON: `{players: [{username, color}], localColor, currentTurnColor, robberHex, diceThrown, turnState}`.
/// Player info comes from memoizedProps (gameValidator.userStates).
/// Live game state (dice, turn) comes from React hooks (memoizedState) on a
/// separate fiber node, since the props snapshot is stale.
pub const EXTRACT_LIVE_JS: &str = r#"(() => {
    let seen = new Set();
    let players = null, localColor = null, robberHex = null, tileState = null;
    let liveGs = null;

    for (let el of document.querySelectorAll('*')) {
        let fk = Object.keys(el).find(k => k.startsWith('__reactFiber'));
        if (!fk) continue;
        let node = el[fk];
        for (let d = 0; d < 50 && node; d++) {
            if (seen.has(node)) { node = node.return; continue; }
            seen.add(node);

            // Player info + robber from props (stale gameState is fine for these).
            if (!players) {
                let p = node.memoizedProps;
                if (p && p.gameValidator && p.gameValidator.userStates) {
                    let users = p.gameValidator.userStates;
                    players = users.map(u => ({username: u.username, color: u.selectedColor}));
                    try {
                        let me = JSON.parse(localStorage.getItem('userState'))?.username;
                        if (me) {
                            let match = users.find(u => u.username === me);
                            if (match) localColor = match.selectedColor;
                        }
                    } catch {}
                    tileState = p.gameValidator.mapValidator?.tileState?._tiles;
                    let gs = p.gameValidator.gameState;
                    let ri = gs?.mechanicRobberState?.locationTileIndex;
                    if (ri != null && tileState && tileState[ri]) {
                        robberHex = { x: tileState[ri].state.x, y: tileState[ri].state.y };
                    }
                }
            }

            // Live game state from hooks (different fiber node).
            if (!liveGs) {
                let ms = node.memoizedState;
                for (let i = 0; i < 30 && ms; i++) {
                    let v = ms.memoizedState;
                    if (v && typeof v === 'object' && v.diceState && v.currentState) {
                        liveGs = v;
                        break;
                    }
                    ms = ms.next;
                }
            }

            if (players && liveGs) break;
            node = node.return;
        }
        if (players && liveGs) break;
    }

    let currentTurnColor = liveGs?.currentState?.currentTurnPlayerColor ?? null;
    let diceThrown = liveGs?.diceState?.diceThrown ?? null;
    let turnState = liveGs?.currentState?.turnState ?? null;

    // Update robber from live state if available.
    if (liveGs && tileState) {
        let ri = liveGs.mechanicRobberState?.locationTileIndex;
        if (ri != null && tileState[ri]) {
            robberHex = { x: tileState[ri].state.x, y: tileState[ri].state.y };
        }
    }

    return JSON.stringify({
        players: players || [],
        localColor,
        currentTurnColor,
        robberHex,
        diceThrown,
        turnState,
    });
})()"#;

/// Lightweight JS to extract only tileCornerStates/tileEdgeStates for building
/// detection during polling. Uses the same `gameManager` path as `EXTRACT_JS`.
pub const EXTRACT_BUILDINGS_JS: &str = r#"(() => {
    let seen = new Set();
    for (let el of document.querySelectorAll('*')) {
        let fk = Object.keys(el).find(k => k.startsWith('__reactFiber'));
        if (!fk) continue;
        let node = el[fk];
        for (let d = 0; d < 50 && node; d++) {
            if (seen.has(node)) { node = node.return; continue; }
            seen.add(node);
            let p = node.memoizedProps;
            if (p?.gameManager) {
                let ts = p.gameManager?.mapController?.uiGameManager?.gameState?.mapState?.tileState;
                let corners = ts?.tileCornerStates || {};
                let edges = ts?.tileEdgeStates || {};
                return JSON.stringify({corners, edges});
            }
            node = node.return;
        }
    }
    return '{}';
})()"#;

/// Parse the lightweight buildings extraction into a `BuildingData`.
pub fn parse_buildings_poll(json_str: &str) -> BuildingData {
    let v: serde_json::Value = serde_json::from_str(json_str).unwrap_or_default();
    extract_buildings_from(
        v.get("corners").unwrap_or(&serde_json::Value::Null),
        v.get("edges").unwrap_or(&serde_json::Value::Null),
    )
}

// -- Types --------------------------------------------------------------------

#[derive(Debug)]
pub struct BoardData {
    pub tiles: Vec<Tile>,
    pub ports: Vec<Port>,
    /// tileCornerStates — entries with `owner`+`buildingType` are buildings.
    pub corners: serde_json::Value,
    /// tileEdgeStates — entries with `owner`+`type` are roads.
    pub edges: serde_json::Value,
    /// Robber hex (x, y) from the last MoveRobber log entry.
    pub robber: Option<serde_json::Value>,
}

#[derive(Debug)]
pub struct Tile {
    pub x: i32,
    pub y: i32,
    pub terrain: Terrain,
    pub dice_number: u8,
}

#[derive(Debug, Clone, Copy)]
pub enum Terrain {
    Desert,
    Lumber,
    Brick,
    Grain,
    Wool,
    Ore,
    Unknown(()),
}

impl Terrain {
    fn from_type(t: u64) -> Self {
        match t {
            0 => Terrain::Desert,
            1 => Terrain::Lumber,
            2 => Terrain::Brick,
            3 => Terrain::Wool,
            4 => Terrain::Grain,
            5 => Terrain::Ore,
            _ => Terrain::Unknown(()),
        }
    }

    #[allow(dead_code)]
    fn resource(self) -> Option<Resource> {
        match self {
            Terrain::Lumber => Some(Resource::Lumber),
            Terrain::Brick => Some(Resource::Brick),
            Terrain::Grain => Some(Resource::Grain),
            Terrain::Wool => Some(Resource::Wool),
            Terrain::Ore => Some(Resource::Ore),
            _ => None,
        }
    }

    fn short(self) -> &'static str {
        match self {
            Terrain::Desert => "  --  ",
            Terrain::Lumber => " Lum  ",
            Terrain::Brick => " Brk  ",
            Terrain::Grain => " Grn  ",
            Terrain::Wool => " Wol  ",
            Terrain::Ore => " Ore  ",
            Terrain::Unknown(_) => "  ??  ",
        }
    }
}

#[derive(Debug)]
pub struct Port {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub port_type: PortType,
}

#[derive(Debug, Clone, Copy)]
pub enum PortType {
    Generic,
    Lumber,
    Brick,
    Grain,
    Wool,
    Ore,
    Unknown(u64),
}

impl PortType {
    fn from_type(t: u64) -> Self {
        match t {
            1 => PortType::Generic,
            2 => PortType::Lumber,
            3 => PortType::Brick,
            4 => PortType::Wool,
            5 => PortType::Grain,
            6 => PortType::Ore,
            _ => PortType::Unknown(t),
        }
    }
}

impl std::fmt::Display for PortType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PortType::Generic => write!(f, "3:1"),
            PortType::Lumber => write!(f, "2:1 lumber"),
            PortType::Brick => write!(f, "2:1 brick"),
            PortType::Grain => write!(f, "2:1 grain"),
            PortType::Wool => write!(f, "2:1 wool"),
            PortType::Ore => write!(f, "2:1 ore"),
            PortType::Unknown(t) => write!(f, "?{t}"),
        }
    }
}

// -- Parsing ------------------------------------------------------------------

pub fn parse(json_str: &str) -> Option<BoardData> {
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;

    let tiles = v["tiles"]
        .as_array()?
        .iter()
        .filter_map(|t| {
            Some(Tile {
                x: t["x"].as_i64()? as i32,
                y: t["y"].as_i64()? as i32,
                terrain: Terrain::from_type(t["type"].as_u64()?),
                dice_number: t["diceNumber"].as_u64().unwrap_or(0) as u8,
            })
        })
        .collect();

    let ports = v["ports"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    Some(Port {
                        x: p["x"].as_i64()? as i32,
                        y: p["y"].as_i64()? as i32,
                        z: p["z"].as_i64()? as i32,
                        port_type: PortType::from_type(p["type"].as_u64()?),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Some(BoardData {
        tiles,
        ports,
        corners: v["corners"].clone(),
        edges: v["edges"].clone(),
        robber: v.get("robber").cloned(),
    })
}

// -- Conversion to canopy topology layout -------------------------------------

impl Terrain {
    /// Convert colonist terrain to canopy terrain.
    pub fn to_canopy(self) -> CanopyTerrain {
        match self {
            Terrain::Desert => CanopyTerrain::Desert,
            Terrain::Lumber => CanopyTerrain::Forest,
            Terrain::Brick => CanopyTerrain::Hills,
            Terrain::Grain => CanopyTerrain::Fields,
            Terrain::Wool => CanopyTerrain::Pasture,
            Terrain::Ore => CanopyTerrain::Mountains,
            Terrain::Unknown(_) => CanopyTerrain::Desert,
        }
    }
}

impl PortType {
    fn to_resource(self) -> Option<Resource> {
        match self {
            PortType::Generic => None,
            PortType::Lumber => Some(Resource::Lumber),
            PortType::Brick => Some(Resource::Brick),
            PortType::Grain => Some(Resource::Grain),
            PortType::Wool => Some(Resource::Wool),
            PortType::Ore => Some(Resource::Ore),
            PortType::Unknown(_) => None,
        }
    }
}

/// Convert colonist board data to `Topology::from_layout_with_ports()` arguments.
///
/// Returns `(terrains, numbers, port_resources, port_specs)` where port_specs
/// is whichever of `PORT_SPECS` / `PORT_SPECS_ALT` best matches the board.
pub fn to_layout(
    board: &BoardData,
    mapper: &CoordMapper,
) -> (
    [CanopyTerrain; 19],
    [Option<u8>; 19],
    [Option<Resource>; 9],
    &'static [(Hex, Direction); 9],
) {
    // Map colonist tile (x,y) → LAND_HEXES index
    let hex_to_land: HashMap<(i32, i32), usize> = LAND_HEXES
        .iter()
        .enumerate()
        .map(|(i, h)| ((h.q as i32, h.r as i32), i))
        .collect();

    let mut terrains = [CanopyTerrain::Desert; 19];
    let mut numbers = [None; 19];

    for tile in &board.tiles {
        let (mx, my) = mapper.map_hex(tile.x, tile.y);
        if let Some(&idx) = hex_to_land.get(&(mx, my)) {
            terrains[idx] = tile.terrain.to_canopy();
            if tile.dice_number > 0 {
                numbers[idx] = Some(tile.dice_number);
            }
        } else {
            eprintln!(
                "warning: colonist tile ({},{}) mapped to ({},{}) not found in LAND_HEXES",
                tile.x, tile.y, mx, my
            );
        }
    }

    // Detect which port configuration this board uses by trying both sets.
    let (port_resources, port_specs) = match_ports(&board.ports, mapper);

    (terrains, numbers, port_resources, port_specs)
}

/// Try matching colonist ports against a port spec set using transformed coords.
/// Returns (port_resources, match_count).
fn try_match_port_specs(
    ports: &[Port],
    mapper: &CoordMapper,
    specs: &[(Hex, Direction); 9],
) -> ([Option<Resource>; 9], usize) {
    let water_to_port: HashMap<(i32, i32), usize> = specs
        .iter()
        .enumerate()
        .map(|(i, &(hex, _))| ((hex.q as i32, hex.r as i32), i))
        .collect();

    let land_to_port: HashMap<(i32, i32), usize> = specs
        .iter()
        .enumerate()
        .map(|(i, &(hex, dir))| {
            let land = hex.neighbor(dir);
            ((land.q as i32, land.r as i32), i)
        })
        .collect();

    let mut port_resources = [None; 9];
    let mut matched = 0;

    for port in ports {
        let transformed = mapper.map_hex(port.x, port.y);
        let idx = water_to_port
            .get(&transformed)
            .or_else(|| land_to_port.get(&transformed));
        if let Some(&idx) = idx {
            port_resources[idx] = port.port_type.to_resource();
            matched += 1;
        }
    }

    (port_resources, matched)
}

/// Match colonist ports against both PORT_SPECS and PORT_SPECS_ALT.
/// Returns port resources indexed by the winning spec set and a reference to it.
fn match_ports(
    ports: &[Port],
    mapper: &CoordMapper,
) -> ([Option<Resource>; 9], &'static [(Hex, Direction); 9]) {
    let (res_primary, n_primary) = try_match_port_specs(ports, mapper, &PORT_SPECS);
    let (res_alt, n_alt) = try_match_port_specs(ports, mapper, &PORT_SPECS_ALT);

    if n_alt > n_primary {
        eprintln!(
            "detected alternate port configuration ({n_alt}/{} matched)",
            ports.len()
        );
        (res_alt, &PORT_SPECS_ALT)
    } else {
        if n_primary < ports.len() && !ports.is_empty() {
            eprintln!(
                "warning: only {n_primary}/{} ports matched PORT_SPECS (alt matched {n_alt})",
                ports.len()
            );
        }
        (res_primary, &PORT_SPECS)
    }
}

// -- Coordinate mapping -------------------------------------------------------

/// Map from colonist corner (x, y, z) to our NodeId.
///
/// Colonist uses z ∈ {0, 1} for corners:
///   z=0 → corner 0 (N), z=1 → corner 3 (S)
///
/// Covers water hex references by registering through sharing neighbors.
pub fn build_corner_map(topology: &Topology) -> HashMap<(i32, i32, u8), NodeId> {
    const Z_TO_CORNER: [u8; 2] = [0, 3];

    // Reverse: corner index → colonist z (None for non-canonical corners)
    let mut corner_to_z = [None; 6];
    for (z, &c) in Z_TO_CORNER.iter().enumerate() {
        corner_to_z[c as usize] = Some(z as u8);
    }

    let mut map = HashMap::new();
    for (i, &hex) in LAND_HEXES.iter().enumerate() {
        for corner in 0..6u8 {
            let nid = topology.tiles[i].nodes[corner as usize];
            if let Some(cz) = corner_to_z[corner as usize] {
                map.entry((hex.q as i32, hex.r as i32, cz)).or_insert(nid);
            }
            for &(dir, nc) in &SHARED_CORNERS[corner as usize] {
                if let Some(cz) = corner_to_z[nc as usize] {
                    let nb = hex.neighbor(dir);
                    map.entry((nb.q as i32, nb.r as i32, cz)).or_insert(nid);
                }
            }
        }
    }
    map
}

/// Map from colonist edge (x, y, z) to our EdgeId.
///
/// Colonist uses z ∈ {0, 1, 2} for edges:
///   z=0 → edge 5 (NW-N), z=1 → edge 4 (SW-NW), z=2 → edge 3 (S-SW)
///
/// Covers water hex references by registering through sharing neighbors.
pub fn build_edge_map(topology: &Topology) -> HashMap<(i32, i32, u8), EdgeId> {
    const Z_TO_EDGE: [u8; 3] = [5, 4, 3];

    let mut edge_to_z = [None; 6];
    for (z, &e) in Z_TO_EDGE.iter().enumerate() {
        edge_to_z[e as usize] = Some(z as u8);
    }

    let mut map = HashMap::new();
    for (i, &hex) in LAND_HEXES.iter().enumerate() {
        for edge in 0..6u8 {
            let eid = topology.tiles[i].edges[edge as usize];
            if let Some(ez) = edge_to_z[edge as usize] {
                map.entry((hex.q as i32, hex.r as i32, ez)).or_insert(eid);
            }
            let opposite = (edge + 3) % 6;
            if let Some(ez) = edge_to_z[opposite as usize] {
                let nb = hex.neighbor(EDGE_NEIGHBOR_DIR[edge as usize]);
                map.entry((nb.q as i32, nb.r as i32, ez)).or_insert(eid);
            }
        }
    }
    map
}

/// Extracted building and road positions from tileCornerStates / tileEdgeStates.
#[derive(Default)]
pub struct BuildingData {
    /// (player_color, x, y, z) for each settlement
    pub settlements: Vec<(u8, i32, i32, u8)>,
    /// (player_color, x, y, z) for each city
    pub cities: Vec<(u8, i32, i32, u8)>,
    /// (player_color, x, y, z) for each road
    pub roads: Vec<(u8, i32, i32, u8)>,
    /// Robber position as LAND_HEXES index (converted from colonist tile index)
    pub robber_tile_index: Option<u8>,
}

/// Extract buildings and roads from the board data, including robber position.
pub fn extract_buildings(board: &BoardData, mapper: &CoordMapper) -> BuildingData {
    let mut data = extract_buildings_from(&board.corners, &board.edges);

    if let Some(robber) = &board.robber {
        let x = robber["x"].as_i64().unwrap_or(0) as i32;
        let y = robber["y"].as_i64().unwrap_or(0) as i32;
        let (mx, my) = mapper.map_hex(x, y);
        data.robber_tile_index = LAND_HEXES
            .iter()
            .position(|h| h.q as i32 == mx && h.r as i32 == my)
            .map(|i| i as u8);
    }

    data
}

/// Extract buildings and roads from raw corner/edge JSON values.
///
/// Corners with `owner` + `buildingType` (1=settlement, 2=city) are buildings.
/// Edges with `owner` + `type` (1=road) are roads.
fn extract_buildings_from(corners: &serde_json::Value, edges: &serde_json::Value) -> BuildingData {
    let mut data = BuildingData {
        settlements: Vec::new(),
        cities: Vec::new(),
        roads: Vec::new(),
        robber_tile_index: None,
    };

    if let Some(obj) = corners.as_object() {
        for entry in obj.values() {
            let owner = match entry["owner"].as_u64() {
                Some(o) => o as u8,
                None => continue,
            };
            let x = entry["x"].as_i64().unwrap_or(0) as i32;
            let y = entry["y"].as_i64().unwrap_or(0) as i32;
            let z = entry["z"].as_u64().unwrap_or(0) as u8;
            match entry["buildingType"].as_u64() {
                Some(1) => data.settlements.push((owner, x, y, z)),
                Some(2) => data.cities.push((owner, x, y, z)),
                _ => {}
            }
        }
    }

    if let Some(obj) = edges.as_object() {
        for entry in obj.values() {
            let owner = match entry["owner"].as_u64() {
                Some(o) => o as u8,
                None => continue,
            };
            let x = entry["x"].as_i64().unwrap_or(0) as i32;
            let y = entry["y"].as_i64().unwrap_or(0) as i32;
            let z = entry["z"].as_u64().unwrap_or(0) as u8;
            if entry["type"].as_u64() == Some(1) {
                data.roads.push((owner, x, y, z));
            }
        }
    }

    data
}

// -- Display ------------------------------------------------------------------

pub fn print(board: &BoardData) {
    println!("\n--- Board ---");

    // Print tiles as hex grid
    // Colonist uses offset axial coords. Group by y, offset by y for hex layout.
    let mut rows: std::collections::BTreeMap<i32, Vec<&Tile>> = std::collections::BTreeMap::new();
    for tile in &board.tiles {
        rows.entry(tile.y).or_default().push(tile);
    }
    for (_, tiles) in &mut rows {
        tiles.sort_by_key(|t| t.x);
    }

    for (&y, tiles) in &rows {
        // Indent based on row for hex layout
        let indent = ((y + 2) * 3) as usize; // adjust offset so top row is least indented
        let pad = " ".repeat(indent);
        let mut terrain_line = pad.clone();
        let mut dice_line = pad;

        for tile in tiles {
            terrain_line.push_str(tile.terrain.short());
            if tile.dice_number > 0 {
                dice_line.push_str(&format!("  {:2}  ", tile.dice_number));
            } else {
                dice_line.push_str("      ");
            }
        }
        println!("{terrain_line}");
        println!("{dice_line}");
    }

    // Ports
    println!("\nPorts:");
    for port in &board.ports {
        println!("  ({},{},{}) {}", port.x, port.y, port.z, port.port_type);
    }

    // Tile counts for verification
    let mut terrain_counts = [0u32; 6];
    for tile in &board.tiles {
        match tile.terrain {
            Terrain::Desert => terrain_counts[0] += 1,
            Terrain::Lumber => terrain_counts[1] += 1,
            Terrain::Brick => terrain_counts[2] += 1,
            Terrain::Grain => terrain_counts[3] += 1,
            Terrain::Wool => terrain_counts[4] += 1,
            Terrain::Ore => terrain_counts[5] += 1,
            _ => {}
        }
    }
    println!(
        "\nTiles: {} desert, {} lumber, {} brick, {} grain, {} wool, {} ore",
        terrain_counts[0],
        terrain_counts[1],
        terrain_counts[2],
        terrain_counts[3],
        terrain_counts[4],
        terrain_counts[5]
    );

    // Buildings and roads
    let buildings = extract_buildings(board, &CoordMapper::identity());
    println!(
        "\nBuildings: {} settlements, {} cities, {} roads",
        buildings.settlements.len(),
        buildings.cities.len(),
        buildings.roads.len()
    );
    for (color, x, y, z) in &buildings.settlements {
        println!("  settlement: color={color} ({x},{y},{z})");
    }
    for (color, x, y, z) in &buildings.cities {
        println!("  city: color={color} ({x},{y},{z})");
    }
    for (color, x, y, z) in &buildings.roads {
        println!("  road: color={color} ({x},{y},{z})");
    }
    if let Some(idx) = buildings.robber_tile_index {
        println!("  robber: tile index {idx}");
    }
}
