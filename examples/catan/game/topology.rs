use std::collections::HashMap;

use super::board::{AdjacencyBitboards, Edge, EdgeId, Node, NodeId, Port, Terrain, Tile, TileId};
use super::hex::{Direction, Hex};
use super::resource::Resource;

pub struct Topology {
    pub tiles: Vec<Tile>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub dice_to_tiles: [Vec<TileId>; 13],
    pub robber_start: TileId,
    pub adj: AdjacencyBitboards,
}

/// For a hex, corner i is shared with specific corners of adjacent hexes.
/// This maps (hex_direction, corner_of_neighbor) pairs that share each corner.
pub const SHARED_CORNERS: [[(Direction, u8); 2]; 6] = [
    // Corner 0 (North)
    [(Direction::Northeast, 4), (Direction::Northwest, 2)],
    // Corner 1 (NE)
    [(Direction::Northeast, 3), (Direction::East, 5)],
    // Corner 2 (SE)
    [(Direction::East, 4), (Direction::Southeast, 0)],
    // Corner 3 (South)
    [(Direction::Southeast, 5), (Direction::Southwest, 1)],
    // Corner 4 (SW)
    [(Direction::Southwest, 0), (Direction::West, 2)],
    // Corner 5 (NW)
    [(Direction::West, 1), (Direction::Northwest, 3)],
];

/// Each edge of a hex is shared with one neighbor.
/// Edge i connects corner i and corner (i+1)%6.
/// Edge i is shared with the neighbor in direction EDGE_NEIGHBOR_DIR[i],
/// specifically their edge (i+3)%6 (the opposite edge).
pub const EDGE_NEIGHBOR_DIR: [Direction; 6] = [
    Direction::Northeast, // edge 0 (N-NE) shared with NE neighbor
    Direction::East,      // edge 1 (NE-SE) shared with E neighbor
    Direction::Southeast, // edge 2 (SE-S) shared with SE neighbor
    Direction::Southwest, // edge 3 (S-SW) shared with SW neighbor
    Direction::West,      // edge 4 (SW-NW) shared with W neighbor
    Direction::Northwest, // edge 5 (NW-N) shared with NW neighbor
];

/// The 19 hex positions of the standard Catan board in axial coordinates.
/// Canonical clockwise spiral from top-left: ring 2 → ring 1 → center.
pub const LAND_HEXES: [Hex; 19] = [
    // Ring 2 (clockwise from top-left corner)
    Hex::new(0, -2),
    Hex::new(1, -2),
    Hex::new(2, -2),
    Hex::new(2, -1),
    Hex::new(2, 0),
    Hex::new(1, 1),
    Hex::new(0, 2),
    Hex::new(-1, 2),
    Hex::new(-2, 2),
    Hex::new(-2, 1),
    Hex::new(-2, 0),
    Hex::new(-1, -1),
    // Ring 1 (clockwise from top)
    Hex::new(0, -1),
    Hex::new(1, -1),
    Hex::new(1, 0),
    Hex::new(0, 1),
    Hex::new(-1, 1),
    Hex::new(-1, 0),
    // Center
    Hex::new(0, 0),
];

/// Standard terrain distribution: 4 wood, 3 brick, 4 sheep, 4 wheat, 3 ore, 1 desert.
const TERRAIN_POOL: [Terrain; 19] = [
    Terrain::Forest,
    Terrain::Forest,
    Terrain::Forest,
    Terrain::Forest,
    Terrain::Hills,
    Terrain::Hills,
    Terrain::Hills,
    Terrain::Pasture,
    Terrain::Pasture,
    Terrain::Pasture,
    Terrain::Pasture,
    Terrain::Fields,
    Terrain::Fields,
    Terrain::Fields,
    Terrain::Fields,
    Terrain::Mountains,
    Terrain::Mountains,
    Terrain::Mountains,
    Terrain::Desert,
];

/// Standard number tokens (placed on 18 non-desert tiles).
#[cfg(test)]
const NUMBER_TOKENS: [u8; 18] = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12];

/// Official Catan number token sequence (letters A through R).
pub const TOKEN_SEQUENCE: [u8; 18] = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11];

/// Port positions: (water hex, direction from water toward adjacent land hex).
/// Derived from catanatron reference cube coords converted to axial (q=x, r=z).
///
/// The 18 water hexes around the land ring alternate between two sets of 9.
/// `PORT_SPECS` covers the "primary" set; `PORT_SPECS_ALT` covers the other.
pub const PORT_SPECS: [(Hex, Direction); 9] = [
    (Hex::new(3, 0), Direction::West),       // land: (2,0)
    (Hex::new(1, 2), Direction::Northwest),  // land: (1,1)
    (Hex::new(-1, 3), Direction::Northwest), // land: (-1,2)
    (Hex::new(-3, 3), Direction::Northeast), // land: (-2,2)
    (Hex::new(-3, 1), Direction::East),      // land: (-2,1)
    (Hex::new(-2, -1), Direction::East),     // land: (-1,-1)
    (Hex::new(0, -3), Direction::Southeast), // land: (0,-2)
    (Hex::new(2, -3), Direction::Southwest), // land: (1,-2)
    (Hex::new(3, -2), Direction::Southwest), // land: (2,-1)
];

/// Alternate port positions — the other 9 water hexes around the land ring.
pub const PORT_SPECS_ALT: [(Hex, Direction); 9] = [
    (Hex::new(1, -3), Direction::Southeast),  // land: (1,-2)
    (Hex::new(3, -3), Direction::Southwest),  // land: (2,-2)
    (Hex::new(3, -1), Direction::West),       // land: (2,-1)
    (Hex::new(2, 1), Direction::West),        // land: (1,1)
    (Hex::new(0, 3), Direction::Northwest),   // land: (0,2)
    (Hex::new(-2, 3), Direction::Northeast),  // land: (-1,2)
    (Hex::new(-3, 2), Direction::Northeast),  // land: (-2,1)
    (Hex::new(-3, 0), Direction::East),       // land: (-2,0)
    (Hex::new(-1, -2), Direction::Southeast), // land: (-1,-1)
];

/// Port resource pool: 5 specific + 4 generic.
const PORT_POOL: [Option<Resource>; 9] = [
    Some(Resource::Lumber),
    Some(Resource::Brick),
    Some(Resource::Wool),
    Some(Resource::Grain),
    Some(Resource::Ore),
    None,
    None,
    None,
    None,
];

/// Given a direction from a water hex toward land, return the two corner
/// indices of the adjacent land hex that get port access.
fn port_direction_to_corners(dir: Direction) -> (u8, u8) {
    match dir {
        Direction::East => (1, 2),
        Direction::Southeast => (2, 3),
        Direction::Southwest => (3, 4),
        Direction::West => (4, 5),
        Direction::Northwest => (5, 0),
        Direction::Northeast => (0, 1),
    }
}

// --- Gödel numbering helpers ---

/// Compute the multinomial coefficient n! / (c0! · c1! · ... · ck!)
/// where n = sum of counts.
fn multinomial(counts: &[u8]) -> u64 {
    let n: u8 = counts.iter().sum();
    let mut result: u64 = 1;
    let mut numerator = n as u64;
    // We compute n! / (c0! · c1! · ...) by iterating:
    // for each count ci, divide by ci! as we go.
    // Equivalent to: result = C(n, c0) * C(n-c0, c1) * ...
    for &c in counts {
        // Multiply result by C(numerator, c)
        // C(numerator, c) = numerator! / (c! * (numerator-c)!)
        // but we do it incrementally to avoid overflow
        let mut binom: u64 = 1;
        for j in 0..c as u64 {
            binom = binom * (numerator - j) / (j + 1);
        }
        result *= binom;
        numerator -= c as u64;
    }
    result
}

/// Rank a multiset permutation using the lehmer code algorithm.
/// `perm` contains symbol indices (0..num_symbols).
/// `num_symbols` is the number of distinct symbol types.
fn rank_multiset_perm(perm: &[u8], num_symbols: usize) -> u64 {
    let len = perm.len();
    // Count remaining occurrences of each symbol
    let mut counts = vec![0u8; num_symbols];
    for &s in perm {
        counts[s as usize] += 1;
    }

    let mut rank: u64 = 0;
    for i in 0..len {
        let symbol = perm[i] as usize;
        // For each symbol smaller than perm[i], count permutations if that
        // symbol were placed here instead.
        for s in 0..symbol {
            if counts[s] > 0 {
                counts[s] -= 1;
                rank += multinomial(&counts);
                counts[s] += 1;
            }
        }
        // "Place" perm[i] and move on
        counts[symbol] -= 1;
    }
    rank
}

/// Unrank a multiset permutation. Given a rank and initial counts of each
/// symbol, reconstruct the permutation of length `len`.
fn unrank_multiset_perm(mut rank: u64, counts: &[u8], num_symbols: usize, len: usize) -> Vec<u8> {
    let mut counts = counts.to_vec();
    let mut result = Vec::with_capacity(len);

    for _ in 0..len {
        // Try each symbol in order; subtract permutation counts until we
        // find which symbol belongs at this position.
        for s in 0..num_symbols {
            if counts[s] == 0 {
                continue;
            }
            counts[s] -= 1;
            let perms = multinomial(&counts);
            if rank < perms {
                result.push(s as u8);
                break; // counts[s] already decremented
            }
            rank -= perms;
            counts[s] += 1;
        }
    }
    result
}

// Terrain canonical indices for Gödel encoding
const TERRAIN_SYMBOLS: usize = 6; // Desert=0, Forest=1, Hills=2, Pasture=3, Fields=4, Mountains=5
const TERRAIN_COUNTS: [u8; 6] = [1, 4, 3, 4, 4, 3]; // Desert, Forest, Hills, Pasture, Fields, Mountains

// Port type canonical indices for Gödel encoding
const PORT_SYMBOLS: usize = 6; // Generic=0, Lumber=1, Brick=2, Wool=3, Grain=4, Ore=5
const PORT_COUNTS: [u8; 6] = [4, 1, 1, 1, 1, 1];

fn terrain_to_symbol(t: Terrain) -> u8 {
    match t {
        Terrain::Desert => 0,
        Terrain::Forest => 1,
        Terrain::Hills => 2,
        Terrain::Pasture => 3,
        Terrain::Fields => 4,
        Terrain::Mountains => 5,
    }
}

fn symbol_to_terrain(s: u8) -> Terrain {
    match s {
        0 => Terrain::Desert,
        1 => Terrain::Forest,
        2 => Terrain::Hills,
        3 => Terrain::Pasture,
        4 => Terrain::Fields,
        5 => Terrain::Mountains,
        _ => unreachable!(),
    }
}

fn symbol_to_port_resource(s: u8) -> Option<Resource> {
    match s {
        0 => None,
        1 => Some(Resource::Lumber),
        2 => Some(Resource::Brick),
        3 => Some(Resource::Wool),
        4 => Some(Resource::Grain),
        5 => Some(Resource::Ore),
        _ => unreachable!(),
    }
}

impl Topology {
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        Self::build(&mut rng)
    }

    fn build(rng: &mut fastrand::Rng) -> Self {
        let mut terrains = TERRAIN_POOL;
        rng.shuffle(&mut terrains);

        let mut numbers = [None; 19];
        let mut token_iter = TOKEN_SEQUENCE.iter().copied();
        for (i, &t) in terrains.iter().enumerate() {
            if t != Terrain::Desert {
                numbers[i] = Some(token_iter.next().unwrap());
            }
        }

        let mut port_resources = PORT_POOL;
        rng.shuffle(&mut port_resources);

        let port_specs = if rng.bool() {
            &PORT_SPECS
        } else {
            &PORT_SPECS_ALT
        };
        Self::from_layout_with_ports(terrains, numbers, port_resources, port_specs)
    }

    /// Encode this topology's board layout as a compact u64 Gödel number.
    ///
    /// Layout (53 bits total):
    /// - bits  0..37: terrain permutation index (lehmer code for multiset)
    /// - bit  38:     port spec set (0 = PORT_SPECS, 1 = PORT_SPECS_ALT)
    /// - bits 39..52: port type permutation index (lehmer code for multiset)
    pub fn board_code(&self) -> u64 {
        // --- Terrain permutation rank ---
        let terrain_perm: Vec<u8> = self
            .tiles
            .iter()
            .map(|t| terrain_to_symbol(t.terrain))
            .collect();
        let terrain_rank = rank_multiset_perm(&terrain_perm, TERRAIN_SYMBOLS);

        // --- Port spec set bit ---
        // Determine which port spec set is in use by checking which set's
        // first water hex has a port on its expected land node.
        // We match by checking the actual port node positions.
        let port_spec_bit: u64 = if self.matches_port_specs(&PORT_SPECS) {
            0
        } else {
            1
        };

        // --- Port type permutation rank ---
        // We need to recover the port_resources array: for each port spec
        // position, find what port type was assigned.
        let port_specs = if port_spec_bit == 0 {
            &PORT_SPECS
        } else {
            &PORT_SPECS_ALT
        };
        let hex_set: HashMap<Hex, usize> = LAND_HEXES
            .iter()
            .enumerate()
            .map(|(i, &h)| (h, i))
            .collect();

        let port_perm: Vec<u8> = port_specs
            .iter()
            .map(|&(water_hex, dir)| {
                let land_hex = water_hex.neighbor(dir);
                let land_idx = hex_set[&land_hex];
                let (c0, _) = port_direction_to_corners(dir.opposite());
                let node_id = self.tiles[land_idx].nodes[c0 as usize];
                match self.nodes[node_id.0 as usize].port {
                    Some(Port::Generic) => 0,
                    Some(Port::Specific(r)) => (r as u8) + 1,
                    None => unreachable!("port spec position has no port"),
                }
            })
            .collect();
        let port_rank = rank_multiset_perm(&port_perm, PORT_SYMBOLS);

        terrain_rank | (port_spec_bit << 38) | (port_rank << 39)
    }

    /// Decode a u64 Gödel number back into a `Topology`.
    pub fn from_board_code(code: u64) -> Self {
        let terrain_rank = code & ((1u64 << 38) - 1);
        let port_spec_bit = (code >> 38) & 1;
        let port_rank = (code >> 39) & ((1u64 << 14) - 1);

        // --- Unrank terrain ---
        let terrain_symbols =
            unrank_multiset_perm(terrain_rank, &TERRAIN_COUNTS, TERRAIN_SYMBOLS, 19);
        let mut terrains = [Terrain::Desert; 19];
        for (i, &s) in terrain_symbols.iter().enumerate() {
            terrains[i] = symbol_to_terrain(s);
        }

        // --- Assign number tokens ---
        let mut numbers = [None; 19];
        let mut token_iter = TOKEN_SEQUENCE.iter().copied();
        for (i, &t) in terrains.iter().enumerate() {
            if t != Terrain::Desert {
                numbers[i] = Some(token_iter.next().unwrap());
            }
        }

        // --- Unrank port types ---
        let port_symbols = unrank_multiset_perm(port_rank, &PORT_COUNTS, PORT_SYMBOLS, 9);
        let mut port_resources = [None; 9];
        for (i, &s) in port_symbols.iter().enumerate() {
            port_resources[i] = symbol_to_port_resource(s);
        }

        // --- Select port spec set ---
        let port_specs = if port_spec_bit == 0 {
            &PORT_SPECS
        } else {
            &PORT_SPECS_ALT
        };

        Self::from_layout_with_ports(terrains, numbers, port_resources, port_specs)
    }

    /// Check whether this topology's port positions match the given port specs.
    fn matches_port_specs(&self, specs: &[(Hex, Direction); 9]) -> bool {
        let hex_set: HashMap<Hex, usize> = LAND_HEXES
            .iter()
            .enumerate()
            .map(|(i, &h)| (h, i))
            .collect();

        for &(water_hex, dir) in specs {
            let land_hex = water_hex.neighbor(dir);
            if let Some(&land_idx) = hex_set.get(&land_hex) {
                let (c0, _) = port_direction_to_corners(dir.opposite());
                let node_id = self.tiles[land_idx].nodes[c0 as usize];
                if self.nodes[node_id.0 as usize].port.is_none() {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    /// Build a topology from an explicit board layout.
    ///
    /// - `terrains[i]`: terrain at `LAND_HEXES[i]`
    /// - `numbers[i]`: dice number at `LAND_HEXES[i]`, `None` for desert
    /// - `port_resources[i]`: resource for port at `port_specs[i]`, `None` = generic 3:1
    /// - `port_specs`: which 9 water hex positions to use (e.g. `PORT_SPECS` or `PORT_SPECS_ALT`)
    pub fn from_layout_with_ports(
        terrains: [Terrain; 19],
        numbers: [Option<u8>; 19],
        port_resources: [Option<Resource>; 9],
        port_specs: &[(Hex, Direction); 9],
    ) -> Self {
        let mut tile_data = [(Hex::new(0, 0), Terrain::Desert, None); 19];
        for i in 0..19 {
            tile_data[i] = (LAND_HEXES[i], terrains[i], numbers[i]);
        }

        // --- Build node and edge mappings ---
        let hex_set: HashMap<Hex, usize> = LAND_HEXES
            .iter()
            .enumerate()
            .map(|(i, &h)| (h, i))
            .collect();

        // Canonical node: for each (hex, corner), pick the representative
        let mut node_map: HashMap<(usize, u8), NodeId> = HashMap::new();
        let mut node_count: u8 = 0;

        let mut get_or_create_node =
            |hex_idx: usize, corner: u8, node_map: &mut HashMap<(usize, u8), NodeId>| -> NodeId {
                let hex = LAND_HEXES[hex_idx];
                let mut canonical = (hex_idx, corner);

                for &(dir, neighbor_corner) in &SHARED_CORNERS[corner as usize] {
                    let neighbor_hex = hex.neighbor(dir);
                    if let Some(&ni) = hex_set.get(&neighbor_hex) {
                        let key = (ni, neighbor_corner);
                        if key < canonical {
                            canonical = key;
                        }
                    }
                }

                if let Some(&id) = node_map.get(&canonical) {
                    node_map.insert((hex_idx, corner), id);
                    id
                } else {
                    let id = NodeId(node_count);
                    node_count += 1;
                    node_map.insert(canonical, id);
                    node_map.insert((hex_idx, corner), id);
                    id
                }
            };

        // Create all nodes for all tiles
        let mut tile_nodes: Vec<[NodeId; 6]> = Vec::with_capacity(19);
        for hex_idx in 0..19 {
            let mut nodes = [NodeId(0); 6];
            for corner in 0..6u8 {
                nodes[corner as usize] = get_or_create_node(hex_idx, corner, &mut node_map);
            }
            tile_nodes.push(nodes);
        }

        // Similarly for edges
        let mut edge_map: HashMap<(usize, u8), EdgeId> = HashMap::new();
        let mut edge_count: u8 = 0;

        let mut get_or_create_edge =
            |hex_idx: usize, edge_idx: u8, edge_map: &mut HashMap<(usize, u8), EdgeId>| -> EdgeId {
                let hex = LAND_HEXES[hex_idx];
                let opposite_edge = (edge_idx + 3) % 6;
                let neighbor_dir = EDGE_NEIGHBOR_DIR[edge_idx as usize];
                let neighbor_hex = hex.neighbor(neighbor_dir);

                let mut canonical = (hex_idx, edge_idx);

                if let Some(&ni) = hex_set.get(&neighbor_hex) {
                    let key = (ni, opposite_edge);
                    if key < canonical {
                        canonical = key;
                    }
                }

                if let Some(&id) = edge_map.get(&canonical) {
                    edge_map.insert((hex_idx, edge_idx), id);
                    id
                } else {
                    let id = EdgeId(edge_count);
                    edge_count += 1;
                    edge_map.insert(canonical, id);
                    edge_map.insert((hex_idx, edge_idx), id);
                    id
                }
            };

        let mut tile_edges: Vec<[EdgeId; 6]> = Vec::with_capacity(19);
        for hex_idx in 0..19 {
            let mut edges = [EdgeId(0); 6];
            for ei in 0..6u8 {
                edges[ei as usize] = get_or_create_edge(hex_idx, ei, &mut edge_map);
            }
            tile_edges.push(edges);
        }

        assert_eq!(node_count, 54, "Expected 54 nodes, got {}", node_count);
        assert_eq!(edge_count, 72, "Expected 72 edges, got {}", edge_count);

        // --- Build tiles ---
        let mut robber_start = TileId(0);
        let mut tiles: Vec<Tile> = Vec::with_capacity(19);
        let mut dice_to_tiles: [Vec<TileId>; 13] = Default::default();

        for (i, &(_, terrain, dice_number)) in tile_data.iter().enumerate() {
            let tid = TileId(i as u8);
            if terrain == Terrain::Desert {
                robber_start = tid;
            }
            if let Some(n) = dice_number {
                dice_to_tiles[n as usize].push(tid);
            }
            tiles.push(Tile {
                id: tid,
                terrain,
                nodes: tile_nodes[i],
                edges: tile_edges[i],
            });
        }

        // --- Build nodes ---
        let mut nodes: Vec<Node> = (0..node_count)
            .map(|i| Node {
                id: NodeId(i),
                adjacent_nodes: Vec::new(),
                adjacent_edges: Vec::new(),
                adjacent_tiles: Vec::new(),
                port: None,
            })
            .collect();

        // adjacent_tiles
        for tile in &tiles {
            for &nid in &tile.nodes {
                let node = &mut nodes[nid.0 as usize];
                if !node.adjacent_tiles.contains(&tile.id) {
                    node.adjacent_tiles.push(tile.id);
                }
            }
        }

        // Build edge data
        let mut edges: Vec<Edge> = (0..edge_count)
            .map(|i| Edge {
                id: EdgeId(i),
                nodes: [NodeId(0), NodeId(0)],
                adjacent_edges: Vec::new(),
            })
            .collect();

        for tile in &tiles {
            for ei in 0..6usize {
                let eid = tile.edges[ei];
                let n0 = tile.nodes[ei];
                let n1 = tile.nodes[(ei + 1) % 6];
                edges[eid.0 as usize].nodes = [n0, n1];
            }
        }

        // adjacent_nodes
        for edge in &edges {
            let [n0, n1] = edge.nodes;
            let node0 = &mut nodes[n0.0 as usize];
            if !node0.adjacent_nodes.contains(&n1) {
                node0.adjacent_nodes.push(n1);
            }
            let node1 = &mut nodes[n1.0 as usize];
            if !node1.adjacent_nodes.contains(&n0) {
                node1.adjacent_nodes.push(n0);
            }
        }

        // adjacent_edges (node -> edges)
        for edge in &edges {
            let [n0, n1] = edge.nodes;
            let node0 = &mut nodes[n0.0 as usize];
            if !node0.adjacent_edges.contains(&edge.id) {
                node0.adjacent_edges.push(edge.id);
            }
            let node1 = &mut nodes[n1.0 as usize];
            if !node1.adjacent_edges.contains(&edge.id) {
                node1.adjacent_edges.push(edge.id);
            }
        }

        // edge adjacent_edges: edges sharing a node
        for i in 0..edge_count {
            let eid = EdgeId(i);
            let [n0, n1] = edges[i as usize].nodes;
            let mut adj = Vec::new();
            for &other_eid in nodes[n0.0 as usize].adjacent_edges.iter() {
                if other_eid != eid {
                    adj.push(other_eid);
                }
            }
            for &other_eid in nodes[n1.0 as usize].adjacent_edges.iter() {
                if other_eid != eid && !adj.contains(&other_eid) {
                    adj.push(other_eid);
                }
            }
            edges[i as usize].adjacent_edges = adj;
        }

        // --- Assign ports ---
        for (spec_idx, &(water_hex, dir)) in port_specs.iter().enumerate() {
            let land_hex = water_hex.neighbor(dir);
            if let Some(&land_idx) = hex_set.get(&land_hex) {
                let (c0, c1) = port_direction_to_corners(dir.opposite());
                let node_a = tile_nodes[land_idx][c0 as usize];
                let node_b = tile_nodes[land_idx][c1 as usize];
                let port = match port_resources[spec_idx] {
                    Some(r) => Port::Specific(r),
                    None => Port::Generic,
                };
                nodes[node_a.0 as usize].port = Some(port);
                nodes[node_b.0 as usize].port = Some(port);
            }
        }

        // --- Build adjacency bitboards ---
        let mut adj = AdjacencyBitboards {
            node_adj_nodes: [0u64; 54],
            node_adj_edges: [0u128; 54],
            edge_endpoints: [0u64; 72],
            edge_adj_edges: [0u128; 72],
            tile_nodes: [0u64; 19],
            port_specific: [0u64; 5],
            port_generic: 0u64,
        };

        for node in &nodes {
            let i = node.id.0 as usize;
            for &an in &node.adjacent_nodes {
                adj.node_adj_nodes[i] |= 1u64 << an.0;
            }
            for &ae in &node.adjacent_edges {
                adj.node_adj_edges[i] |= 1u128 << ae.0;
            }
            match node.port {
                Some(Port::Specific(r)) => adj.port_specific[r as usize] |= 1u64 << node.id.0,
                Some(Port::Generic) => adj.port_generic |= 1u64 << node.id.0,
                None => {}
            }
        }

        for edge in &edges {
            let i = edge.id.0 as usize;
            adj.edge_endpoints[i] = (1u64 << edge.nodes[0].0) | (1u64 << edge.nodes[1].0);
            for &ae in &edge.adjacent_edges {
                adj.edge_adj_edges[i] |= 1u128 << ae.0;
            }
        }

        for tile in &tiles {
            let i = tile.id.0 as usize;
            for &nid in &tile.nodes {
                adj.tile_nodes[i] |= 1u64 << nid.0;
            }
        }

        Topology {
            tiles,
            nodes,
            edges,
            dice_to_tiles,
            robber_start,
            adj,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_topo() -> Topology {
        Topology::from_seed(42)
    }

    /// A standard Catan board has exactly 19 hex tiles, 54 vertex nodes, and 72 edges.
    #[test]
    fn topology_correct_counts() {
        let topo = make_topo();
        assert_eq!(topo.tiles.len(), 19);
        assert_eq!(topo.nodes.len(), 54);
        assert_eq!(topo.edges.len(), 72);
    }

    /// Every node on a Catan board has exactly 2 or 3 adjacent nodes/edges
    /// (2 for perimeter corners, 3 for interior vertices).
    #[test]
    fn node_adjacency_counts() {
        let topo = make_topo();
        for node in &topo.nodes {
            let n = node.adjacent_nodes.len();
            assert!(
                n == 2 || n == 3,
                "node {:?} has {} adjacent nodes",
                node.id,
                n
            );
            let e = node.adjacent_edges.len();
            assert!(
                e == 2 || e == 3,
                "node {:?} has {} adjacent edges",
                node.id,
                e
            );
        }
    }

    /// Every edge connects exactly two distinct nodes.
    #[test]
    fn edge_endpoint_counts() {
        let topo = make_topo();
        for edge in &topo.edges {
            let [n0, n1] = edge.nodes;
            assert_ne!(n0, n1, "edge {:?} has duplicate endpoints", edge.id);
            assert!((n0.0 as usize) < topo.nodes.len());
            assert!((n1.0 as usize) < topo.nodes.len());
        }
    }

    /// Every edge is adjacent to 2-4 other edges (depending on whether its
    /// endpoints are degree-2 perimeter or degree-3 interior nodes).
    #[test]
    fn edge_adjacency_counts() {
        let topo = make_topo();
        for edge in &topo.edges {
            let n = edge.adjacent_edges.len();
            assert!(
                (2..=4).contains(&n),
                "edge {:?} has {} adjacent edges",
                edge.id,
                n
            );
        }
    }

    /// Standard Catan has 9 ports: 5 resource-specific (2:1) and 4 generic (3:1).
    /// Each port spans 2 nodes, giving 10 specific + 8 generic = 18 port nodes.
    #[test]
    fn port_counts() {
        let topo = make_topo();
        let mut specific_count = 0;
        let mut generic_count = 0;
        for node in &topo.nodes {
            match node.port {
                Some(Port::Specific(_)) => specific_count += 1,
                Some(Port::Generic) => generic_count += 1,
                None => {}
            }
        }
        // 5 specific ports * 2 nodes each = 10, 4 generic ports * 2 nodes each = 8
        assert_eq!(specific_count, 10, "expected 10 specific port nodes");
        assert_eq!(generic_count, 8, "expected 8 generic port nodes");
        assert_eq!(
            specific_count + generic_count,
            18,
            "expected 18 total port nodes"
        );
    }

    /// Building a topology from the same seed must produce identical layouts
    /// (terrain, ports, adjacency) for deterministic replay.
    #[test]
    fn deterministic_with_seed() {
        let t1 = Topology::from_seed(123);
        let t2 = Topology::from_seed(123);
        assert_eq!(t1.tiles.len(), t2.tiles.len());
        for (a, b) in t1.tiles.iter().zip(t2.tiles.iter()) {
            assert_eq!(a.terrain, b.terrain);
            assert_eq!(a.nodes, b.nodes);
            assert_eq!(a.edges, b.edges);
        }
        for (a, b) in t1.nodes.iter().zip(t2.nodes.iter()) {
            assert_eq!(a.port, b.port);
            assert_eq!(a.adjacent_nodes, b.adjacent_nodes);
        }
    }

    /// Across many random seeds, verify no two edge-adjacent tiles both have
    /// numbers in {6, 8}.
    #[test]
    fn spiral_no_adjacent_6_8() {
        // Build a hex adjacency map for the 19 land hexes
        let hex_set: HashMap<Hex, usize> = LAND_HEXES
            .iter()
            .enumerate()
            .map(|(i, &h)| (h, i))
            .collect();

        let mut hex_neighbors: Vec<Vec<usize>> = vec![Vec::new(); 19];
        for (i, &hex) in LAND_HEXES.iter().enumerate() {
            for dir in [
                Direction::East,
                Direction::West,
                Direction::Northeast,
                Direction::Northwest,
                Direction::Southeast,
                Direction::Southwest,
            ] {
                let nb = hex.neighbor(dir);
                if let Some(&j) = hex_set.get(&nb) {
                    hex_neighbors[i].push(j);
                }
            }
        }

        for seed in 0..1000u64 {
            let topo = Topology::from_seed(seed);
            // Collect which hex indices have 6 or 8
            let hot: Vec<bool> = topo
                .tiles
                .iter()
                .map(|t| {
                    topo.dice_to_tiles[6].contains(&t.id) || topo.dice_to_tiles[8].contains(&t.id)
                })
                .collect();

            for (i, &is_hot) in hot.iter().enumerate() {
                if !is_hot {
                    continue;
                }
                for &j in &hex_neighbors[i] {
                    assert!(
                        !hot[j],
                        "seed {}: adjacent tiles {} and {} both have 6 or 8",
                        seed, i, j
                    );
                }
            }
        }
    }

    /// TOKEN_SEQUENCE contains the same multiset as NUMBER_TOKENS.
    #[test]
    fn token_sequence_has_correct_counts() {
        let mut seq_sorted = TOKEN_SEQUENCE;
        seq_sorted.sort();
        let mut tok_sorted = NUMBER_TOKENS;
        tok_sorted.sort();
        assert_eq!(seq_sorted, tok_sorted);
    }

    /// from_layout produces the same topology as build when given the same
    /// terrain, number, and port assignments.
    #[test]
    fn from_layout_matches_build() {
        for seed in [0, 42, 123, 999] {
            let mut rng = fastrand::Rng::with_seed(seed);
            let t_build = Topology::build(&mut rng);

            // Reconstruct the same inputs build() would pass to from_layout
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut terrains = TERRAIN_POOL;
            rng.shuffle(&mut terrains);

            let mut numbers = [None; 19];
            let mut token_iter = TOKEN_SEQUENCE.iter().copied();
            for (i, &t) in terrains.iter().enumerate() {
                if t != Terrain::Desert {
                    numbers[i] = Some(token_iter.next().unwrap());
                }
            }

            let mut port_resources = PORT_POOL;
            rng.shuffle(&mut port_resources);

            let port_specs = if rng.bool() {
                &PORT_SPECS
            } else {
                &PORT_SPECS_ALT
            };
            let t_layout =
                Topology::from_layout_with_ports(terrains, numbers, port_resources, port_specs);

            // Compare terrains
            for (a, b) in t_build.tiles.iter().zip(t_layout.tiles.iter()) {
                assert_eq!(a.terrain, b.terrain, "seed {seed}: terrain mismatch");
                assert_eq!(a.nodes, b.nodes, "seed {seed}: tile nodes mismatch");
                assert_eq!(a.edges, b.edges, "seed {seed}: tile edges mismatch");
            }
            // Compare ports
            for (a, b) in t_build.nodes.iter().zip(t_layout.nodes.iter()) {
                assert_eq!(
                    a.port, b.port,
                    "seed {seed}: port mismatch at node {:?}",
                    a.id
                );
            }
            assert_eq!(
                t_build.robber_start, t_layout.robber_start,
                "seed {seed}: robber mismatch"
            );
        }
    }

    #[test]
    fn multinomial_basic() {
        // 3!/(1!·1!·1!) = 6
        assert_eq!(multinomial(&[1, 1, 1]), 6);
        // 4!/(2!·2!) = 6
        assert_eq!(multinomial(&[2, 2]), 6);
        // 19!/(4!·3!·4!·4!·3!·1!) = 244_432_188_000
        assert_eq!(multinomial(&TERRAIN_COUNTS), 244_432_188_000);
        // 9!/(4!·1!·1!·1!·1!·1!) = 15120
        assert_eq!(multinomial(&PORT_COUNTS), 15120);
    }

    /// Encoding then decoding must produce identical terrain and port layouts.
    #[test]
    fn board_code_round_trip() {
        for seed in [0, 1, 42, 123, 456, 789, 999, 12345, 99999] {
            let topo = Topology::from_seed(seed);
            let code = topo.board_code();
            let decoded = Topology::from_board_code(code);

            // Verify terrains match
            for (i, (a, b)) in topo.tiles.iter().zip(decoded.tiles.iter()).enumerate() {
                assert_eq!(
                    a.terrain, b.terrain,
                    "seed {seed}, tile {i}: terrain mismatch"
                );
            }
            // Verify ports match
            for (i, (a, b)) in topo.nodes.iter().zip(decoded.nodes.iter()).enumerate() {
                assert_eq!(a.port, b.port, "seed {seed}, node {i}: port mismatch");
            }
            // Verify robber start matches
            assert_eq!(
                topo.robber_start, decoded.robber_start,
                "seed {seed}: robber start mismatch"
            );
            // Verify the code fits in 53 bits
            assert!(
                code < (1u64 << 53),
                "seed {seed}: code {code} exceeds 53 bits"
            );
        }
    }

    /// Same seed must produce the same board code.
    #[test]
    fn board_code_deterministic() {
        for seed in [0, 42, 123, 999] {
            let code1 = Topology::from_seed(seed).board_code();
            let code2 = Topology::from_seed(seed).board_code();
            assert_eq!(code1, code2, "seed {seed}: non-deterministic board code");
        }
    }

    /// Different seeds should (usually) produce different board codes.
    #[test]
    fn board_code_different_boards() {
        let codes: Vec<u64> = (0..100)
            .map(|s| Topology::from_seed(s).board_code())
            .collect();
        let unique: std::collections::HashSet<u64> = codes.iter().copied().collect();
        // With 100 seeds over a space of ~3.7 trillion, collisions are vanishingly unlikely
        assert!(
            unique.len() > 90,
            "too many collisions: {} unique out of 100",
            unique.len()
        );
    }
}
