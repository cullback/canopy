//! Parse the colonist.io board state extracted via Chrome DevTools Protocol.
//!
//! Tiles and ports come from `gameValidator.mapValidator`.  Building and road
//! positions come from `gameManager.mapController.uiGameManager.gameState`
//! which carries `owner`/`buildingType`/`type` on corner and edge states.

use std::collections::HashMap;

use crate::game::board::{EdgeId, NodeId, Terrain as CanopyTerrain};
use crate::game::resource::Resource;
use crate::game::topology::{EDGE_NEIGHBOR_DIR, LAND_HEXES, PORT_SPECS, SHARED_CORNERS, Topology};

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

    // Robber: find the last MoveRobber (type 11) log entry and match its
    // tileInfo (tileType + diceNumber) against the tile array to get coords.
    let robber = null;
    let vs = document.querySelector('[class*="virtualScroller"]');
    if (vs) {
        let fk2 = Object.keys(vs).find(k => k.startsWith('__reactFiber'));
        let logNode = fk2 && vs[fk2].return?.return;
        let children = logNode?.memoizedProps?.children;
        if (Array.isArray(children)) {
            for (let i = children.length - 1; i >= 0; i--) {
                let e = children[i]?.props?.gameLogData?.text;
                if (e?.type === 11 && e.tileInfo && tiles) {
                    let ti = e.tileInfo;
                    let match = tiles.find(t => t.type === ti.tileType && t.diceNumber === ti.diceNumber);
                    if (match) robber = { x: match.x, y: match.y };
                    break;
                }
            }
        }
    }

    return JSON.stringify({ tiles, ports, corners, edges, robber });
})()"#;

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
    Unknown(u64),
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
            _ => Terrain::Unknown(t),
        }
    }

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
                dice_number: t["diceNumber"].as_u64()? as u8,
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

/// Convert colonist board data to `Topology::from_layout()` arguments.
///
/// Returns `(terrains, numbers, port_resources)` indexed by LAND_HEXES and
/// PORT_SPECS position respectively.
pub fn to_layout(
    board: &BoardData,
) -> ([CanopyTerrain; 19], [Option<u8>; 19], [Option<Resource>; 9]) {
    // Map colonist tile (x,y) → LAND_HEXES index
    let hex_to_land: HashMap<(i32, i32), usize> = LAND_HEXES
        .iter()
        .enumerate()
        .map(|(i, h)| ((h.q as i32, h.r as i32), i))
        .collect();

    let mut terrains = [CanopyTerrain::Desert; 19];
    let mut numbers = [None; 19];

    for tile in &board.tiles {
        if let Some(&idx) = hex_to_land.get(&(tile.x, tile.y)) {
            terrains[idx] = tile.terrain.to_canopy();
            if tile.dice_number > 0 {
                numbers[idx] = Some(tile.dice_number);
            }
        } else {
            eprintln!(
                "warning: colonist tile ({},{}) not found in LAND_HEXES",
                tile.x, tile.y
            );
        }
    }

    // Map colonist ports to PORT_SPECS positions.
    // Try matching (x,y) as water hex first, then as land hex.
    let water_to_port: HashMap<(i32, i32), usize> = PORT_SPECS
        .iter()
        .enumerate()
        .map(|(i, &(hex, _))| ((hex.q as i32, hex.r as i32), i))
        .collect();

    let land_to_port: HashMap<(i32, i32), usize> = PORT_SPECS
        .iter()
        .enumerate()
        .map(|(i, &(hex, dir))| {
            let land = hex.neighbor(dir);
            ((land.q as i32, land.r as i32), i)
        })
        .collect();

    let mut port_resources = [None; 9];

    for port in &board.ports {
        let idx = water_to_port
            .get(&(port.x, port.y))
            .or_else(|| land_to_port.get(&(port.x, port.y)));
        if let Some(&idx) = idx {
            port_resources[idx] = port.port_type.to_resource();
        } else {
            eprintln!(
                "warning: colonist port ({},{},{}) not matched to PORT_SPECS",
                port.x, port.y, port.z
            );
        }
    }

    (terrains, numbers, port_resources)
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

/// Extract buildings and roads from the board data.
///
/// Corners with `owner` + `buildingType` (1=settlement, 2=city) are buildings.
/// Edges with `owner` + `type` (1=road) are roads.
pub fn extract_buildings(board: &BoardData) -> BuildingData {
    let mut data = BuildingData {
        settlements: Vec::new(),
        cities: Vec::new(),
        roads: Vec::new(),
        robber_tile_index: None,
    };

    // Parse corners (settlements/cities)
    if let Some(obj) = board.corners.as_object() {
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

    // Parse edges (roads)
    if let Some(obj) = board.edges.as_object() {
        for entry in obj.values() {
            let owner = match entry["owner"].as_u64() {
                Some(o) => o as u8,
                None => continue,
            };
            let x = entry["x"].as_i64().unwrap_or(0) as i32;
            let y = entry["y"].as_i64().unwrap_or(0) as i32;
            let z = entry["z"].as_u64().unwrap_or(0) as u8;
            // type 1 = road
            if entry["type"].as_u64() == Some(1) {
                data.roads.push((owner, x, y, z));
            }
        }
    }

    // Robber: (x, y) from the last MoveRobber log entry
    if let Some(robber) = &board.robber {
        let x = robber["x"].as_i64().unwrap_or(0) as i32;
        let y = robber["y"].as_i64().unwrap_or(0) as i32;
        data.robber_tile_index = LAND_HEXES
            .iter()
            .position(|h| h.q as i32 == x && h.r as i32 == y)
            .map(|i| i as u8);
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
    let buildings = extract_buildings(board);
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
