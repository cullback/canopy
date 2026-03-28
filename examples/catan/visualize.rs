use serde::Serialize;

use canopy::player::Player;

use crate::game::action::{
    ActionId, BUY_DEV_CARD, CITY_END, CITY_START, DISCARD_END, DISCARD_START, END_TURN,
    MARITIME_END, MARITIME_START, MONOPOLY_END, MONOPOLY_START, PLAY_KNIGHT, PLAY_ROAD_BUILDING,
    ROAD_END, ROAD_START, ROBBER_END, ROBBER_START, ROLL, SETTLEMENT_END, SETTLEMENT_START,
    YOP_END, YOP_START,
};
use crate::game::board::{Port, TileId};
use crate::game::hex;
use crate::game::state::{GameState, Phase};
use crate::game::topology::LAND_HEXES;

// ─── Replay Data ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ReplayBoard {
    tiles: Vec<ReplayTile>,
    nodes: Vec<[f64; 2]>,
    edges: Vec<[u8; 2]>,
    ports: Vec<ReplayPort>,
}

#[derive(Serialize)]
pub struct ReplayTile {
    terrain: &'static str,
    number: Option<u8>,
    cx: f64,
    cy: f64,
    nodes: [u8; 6],
}

#[derive(Serialize)]
pub struct ReplayPort {
    nodes: [u8; 2],
    kind: String,
}

#[derive(Serialize)]
pub struct ReplayFrame {
    action: String,
    player: u8,
    phase: String,
    turn: u16,
    robber: u8,
    last_roll: Option<u8>,
    players: [PlayerFrame; 2],
    buildings: [BuildingsFrame; 2],
    longest_road: Option<[u8; 2]>,
    largest_army: Option<[u8; 2]>,
    dev_pool: [u8; 5],
}

#[derive(Serialize)]
pub struct PlayerFrame {
    hand: [u8; 5],
    vp: u8,
    dev_cards: [u8; 5],
    dev_cards_bought_this_turn: [u8; 5],
    hidden_dev_cards: u8,
    knights: u8,
    trade_ratios: [u8; 5],
}

#[derive(Serialize)]
pub struct BuildingsFrame {
    settlements: Vec<u8>,
    cities: Vec<u8>,
    roads: Vec<u8>,
}

// ─── Coordinate math ─────────────────────────────────────────────────────────

const HEX_SIZE: f64 = 50.0;

fn compute_node_positions(state: &GameState) -> Vec<[f64; 2]> {
    let mut positions = vec![[0.0, 0.0]; 54];
    let mut assigned = [false; 54];

    for (i, tile) in state.topology.tiles.iter().enumerate() {
        let (cx, cy) = LAND_HEXES[i].pixel_center(HEX_SIZE);
        for corner in 0..6 {
            let nid = tile.nodes[corner].0 as usize;
            if !assigned[nid] {
                let (dx, dy) = hex::corner_pixel_offset(corner, HEX_SIZE);
                positions[nid] = [cx + dx, cy + dy];
                assigned[nid] = true;
            }
        }
    }

    // Round to 1 decimal place to keep JSON compact
    for pos in &mut positions {
        pos[0] = (pos[0] * 10.0).round() / 10.0;
        pos[1] = (pos[1] * 10.0).round() / 10.0;
    }

    positions
}

// ─── Board construction ──────────────────────────────────────────────────────

fn tile_number(state: &GameState, tid: TileId) -> Option<u8> {
    for roll in 2..=12u8 {
        if state.topology.dice_to_tiles[roll as usize].contains(&tid) {
            return Some(roll);
        }
    }
    None
}

fn terrain_name(terrain: crate::game::board::Terrain) -> &'static str {
    use crate::game::board::Terrain;
    match terrain {
        Terrain::Forest => "forest",
        Terrain::Hills => "hills",
        Terrain::Pasture => "pasture",
        Terrain::Fields => "fields",
        Terrain::Mountains => "mountains",
        Terrain::Desert => "desert",
    }
}

pub fn build_board(state: &GameState) -> ReplayBoard {
    let node_positions = compute_node_positions(state);
    let topo = &state.topology;

    let tiles: Vec<ReplayTile> = topo
        .tiles
        .iter()
        .enumerate()
        .map(|(i, tile)| {
            let (cx, cy) = LAND_HEXES[i].pixel_center(HEX_SIZE);
            ReplayTile {
                terrain: terrain_name(tile.terrain),
                number: tile_number(state, tile.id),
                cx: (cx * 10.0).round() / 10.0,
                cy: (cy * 10.0).round() / 10.0,
                nodes: tile.nodes.map(|n| n.0),
            }
        })
        .collect();

    let edges: Vec<[u8; 2]> = topo
        .edges
        .iter()
        .map(|e| [e.nodes[0].0, e.nodes[1].0])
        .collect();

    // Find port pairs: an edge where both endpoints have the same port type
    let mut ports = Vec::new();
    for edge in &topo.edges {
        let [n0, n1] = edge.nodes;
        let p0 = &topo.nodes[n0.0 as usize].port;
        let p1 = &topo.nodes[n1.0 as usize].port;
        if let (Some(p0), Some(p1)) = (p0, p1) {
            if p0 == p1 {
                let kind = match p0 {
                    Port::Generic => "generic".to_string(),
                    Port::Specific(r) => r.to_string(),
                };
                ports.push(ReplayPort {
                    nodes: [n0.0, n1.0],
                    kind,
                });
            }
        }
    }

    ReplayBoard {
        tiles,
        nodes: node_positions,
        edges,
        ports,
    }
}

// ─── Frame capture ───────────────────────────────────────────────────────────

fn bits_to_vec(mut bits: u64) -> Vec<u8> {
    let mut v = Vec::new();
    while bits != 0 {
        v.push(bits.trailing_zeros() as u8);
        bits &= bits - 1;
    }
    v
}

fn bits128_to_vec(mut bits: u128) -> Vec<u8> {
    let mut v = Vec::new();
    while bits != 0 {
        v.push(bits.trailing_zeros() as u8);
        bits &= bits - 1;
    }
    v
}

fn player_frame(state: &GameState, pid: Player) -> PlayerFrame {
    let ps = &state.players[pid];
    PlayerFrame {
        hand: ps.hand.0,
        vp: state.total_vps(pid),
        dev_cards: ps.dev_cards.0,
        dev_cards_bought_this_turn: ps.dev_cards_bought_this_turn.0,
        hidden_dev_cards: ps.hidden_dev_cards,
        knights: ps.knights_played,
        trade_ratios: ps.trade_ratios,
    }
}

fn buildings_frame(state: &GameState, pid: Player) -> BuildingsFrame {
    BuildingsFrame {
        settlements: bits_to_vec(state.boards[pid].settlements),
        cities: bits_to_vec(state.boards[pid].cities),
        roads: bits128_to_vec(state.boards[pid].road_network.roads),
    }
}

pub fn capture_frame(
    state: &GameState,
    action: &str,
    player: u8,
    last_roll: Option<u8>,
) -> ReplayFrame {
    ReplayFrame {
        action: action.to_string(),
        player,
        phase: format_phase(&state.phase),
        turn: state.turn_number,
        robber: state.robber.0,
        last_roll,
        players: [
            player_frame(state, Player::One),
            player_frame(state, Player::Two),
        ],
        buildings: [
            buildings_frame(state, Player::One),
            buildings_frame(state, Player::Two),
        ],
        longest_road: state.longest_road.map(|(p, len)| [p as u8, len]),
        largest_army: state.largest_army.map(|(p, cnt)| [p as u8, cnt]),
        dev_pool: state.unknown_dev_pool(),
    }
}

// ─── Action / phase formatting ───────────────────────────────────────────────

/// Compact action description without player prefix (for web UI / tree explorer).
#[allow(non_contiguous_range_endpoints)]
pub fn format_action_desc(action: ActionId, _state: &GameState) -> String {
    match action.0 {
        SETTLEMENT_START..SETTLEMENT_END => {
            format!("Settle N{}", action.settlement_node().0)
        }
        ROAD_START..ROAD_END => {
            format!("Road E{}", action.road_edge().0)
        }
        CITY_START..CITY_END => format!("City N{}", action.city_node().0),
        ROLL => "Roll".to_string(),
        END_TURN => "End".to_string(),
        BUY_DEV_CARD => "Buy dev".to_string(),
        PLAY_KNIGHT => "Knight".to_string(),
        PLAY_ROAD_BUILDING => "Road Build".to_string(),
        YOP_START..YOP_END => {
            let (r1, r2) = action.year_of_plenty_resources();
            format!("YoP: {r1}+{r2}")
        }
        MONOPOLY_START..MONOPOLY_END => {
            format!("Mono: {}", action.monopoly_resource())
        }
        ROBBER_START..ROBBER_END => format!("Robber T{}", action.robber_tile().0),
        DISCARD_START..DISCARD_END => format!("Drop {}", action.discard_resource()),
        MARITIME_START..MARITIME_END => {
            let (give, recv) = action.maritime_trade();
            format!("{give}→{recv}")
        }
        _ => format!("#{}", action.0),
    }
}

pub fn format_phase(phase: &Phase) -> String {
    match phase {
        Phase::PlaceSettlement => "Place Settlement".to_string(),
        Phase::PlaceRoad => "Place Road".to_string(),
        Phase::PreRoll => "Pre-Roll".to_string(),
        Phase::Roll => "Roll".to_string(),
        Phase::Discard {
            player, remaining, ..
        } => format!("{player} discards ({remaining} left)"),
        Phase::MoveRobber => "Move Robber".to_string(),
        Phase::StealResolve => "Steal Resolve".to_string(),
        Phase::Main => "Main".to_string(),
        Phase::DevCardDraw => "Dev Card Draw".to_string(),
        Phase::RoadBuilding { roads_left } => format!("Road Building ({roads_left} left)"),
        Phase::GameOver(p) => format!("Game Over ({p} wins)"),
    }
}
