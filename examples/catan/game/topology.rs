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
const SHARED_CORNERS: [[(Direction, u8); 2]; 6] = [
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
const EDGE_NEIGHBOR_DIR: [Direction; 6] = [
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
const TOKEN_SEQUENCE: [u8; 18] = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11];

/// Port positions: (water hex, direction from water toward adjacent land hex).
/// Derived from catanatron reference cube coords converted to axial (q=x, r=z).
const PORT_SPECS: [(Hex, Direction); 9] = [
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

impl Topology {
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        Self::build(&mut rng)
    }

    fn build(rng: &mut fastrand::Rng) -> Self {
        // Shuffle terrains
        let mut terrains = TERRAIN_POOL;
        rng.shuffle(&mut terrains);

        // Shuffle ports
        let mut port_resources = PORT_POOL;
        rng.shuffle(&mut port_resources);

        // Assign terrains to hex positions, then place tokens in LAND_HEXES order
        let mut tile_data = [(Hex::new(0, 0), Terrain::Desert, None); 19];
        for i in 0..19 {
            tile_data[i].0 = LAND_HEXES[i];
            tile_data[i].1 = terrains[i];
        }
        let mut token_iter = TOKEN_SEQUENCE.iter().copied();
        for hex_idx in 0..19 {
            if terrains[hex_idx] != Terrain::Desert {
                tile_data[hex_idx].2 = Some(token_iter.next().unwrap());
            }
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
        for (spec_idx, &(water_hex, dir)) in PORT_SPECS.iter().enumerate() {
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
}
