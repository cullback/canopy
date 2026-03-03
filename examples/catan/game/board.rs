use super::resource::Resource;

// --- Newtypes for type safety ---

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub u8);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct EdgeId(pub u8);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TileId(pub u8);

// --- Terrain & Port ---

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Terrain {
    Forest,
    Hills,
    Pasture,
    Fields,
    Mountains,
    Desert,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Port {
    Specific(Resource),
    Generic,
}

impl Terrain {
    pub fn resource(self) -> Option<Resource> {
        match self {
            Terrain::Forest => Some(Resource::Lumber),
            Terrain::Hills => Some(Resource::Brick),
            Terrain::Pasture => Some(Resource::Wool),
            Terrain::Fields => Some(Resource::Grain),
            Terrain::Mountains => Some(Resource::Ore),
            Terrain::Desert => None,
        }
    }
}

// --- Board data ---

pub struct Tile {
    pub id: TileId,
    pub terrain: Terrain,
    pub nodes: [NodeId; 6],
    pub edges: [EdgeId; 6],
}

pub struct Node {
    pub id: NodeId,
    pub adjacent_nodes: Vec<NodeId>,
    pub adjacent_edges: Vec<EdgeId>,
    pub adjacent_tiles: Vec<TileId>,
    pub port: Option<Port>,
}

pub struct Edge {
    pub id: EdgeId,
    pub nodes: [NodeId; 2],
    pub adjacent_edges: Vec<EdgeId>,
}

pub struct AdjacencyBitboards {
    pub node_adj_nodes: [u64; 54],  // per node: bitmask of adjacent nodes
    pub node_adj_edges: [u128; 54], // per node: bitmask of incident edges
    pub edge_endpoints: [u64; 72],  // per edge: bitmask of 2 endpoint nodes
    pub edge_adj_edges: [u128; 72], // per edge: bitmask of adjacent edges
    pub tile_nodes: [u64; 19],      // per tile: bitmask of 6 corner nodes
    pub port_specific: [u64; 5],    // per resource: nodes with 2:1 port
    pub port_generic: u64,          // nodes with 3:1 port
}
