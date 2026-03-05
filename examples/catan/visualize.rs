use serde::Serialize;
use std::path::Path;

use canopy2::game::Game;
pub use canopy2::game_log::GameLog;
use canopy2::player::Player;

use crate::game;
use crate::game::action::{
    ActionId, BUY_DEV_CARD, CITY_END, CITY_START, DISCARD_END, DISCARD_START, END_TURN,
    MARITIME_END, MARITIME_START, MONOPOLY_END, MONOPOLY_START, PLAY_KNIGHT, PLAY_ROAD_BUILDING,
    ROAD_END, ROAD_START, ROBBER_END, ROBBER_START, SETTLEMENT_END, SETTLEMENT_START, YOP_END,
    YOP_START,
};
use crate::game::board::{Port, TileId};
use crate::game::dice::Dice;
use crate::game::hex;
use crate::game::state::{GameState, Phase};
use crate::game::topology::LAND_HEXES;

// ─── Replay Data ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ReplayData {
    board: ReplayBoard,
    frames: Vec<ReplayFrame>,
    result: String,
    bot1: String,
    bot2: String,
}

#[derive(Serialize)]
struct ReplayBoard {
    tiles: Vec<ReplayTile>,
    nodes: Vec<[f64; 2]>,
    edges: Vec<[u8; 2]>,
    ports: Vec<ReplayPort>,
}

#[derive(Serialize)]
struct ReplayTile {
    terrain: &'static str,
    number: Option<u8>,
    cx: f64,
    cy: f64,
    nodes: [u8; 6],
}

#[derive(Serialize)]
struct ReplayPort {
    nodes: [u8; 2],
    kind: String,
}

#[derive(Serialize)]
struct ReplayFrame {
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
}

#[derive(Serialize)]
struct PlayerFrame {
    hand: [u8; 5],
    vp: u8,
    dev_cards: [u8; 5],
    knights: u8,
    trade_ratios: [u8; 5],
}

#[derive(Serialize)]
struct BuildingsFrame {
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

fn build_board(state: &GameState) -> ReplayBoard {
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

fn capture_frame(
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
    }
}

// ─── Action / phase formatting ───────────────────────────────────────────────

#[allow(non_contiguous_range_endpoints)]
fn format_action(action: ActionId, state: &GameState) -> String {
    let player = match state.current_player {
        Player::One => "P1",
        Player::Two => "P2",
    };

    let desc = match action.0 {
        SETTLEMENT_START..SETTLEMENT_END => {
            let nid = action.settlement_node().0;
            if matches!(state.phase, Phase::PlaceSettlement) {
                format!("Place settlement at node {nid}")
            } else {
                format!("Build settlement at node {nid}")
            }
        }
        ROAD_START..ROAD_END => {
            let eid = action.road_edge().0;
            if matches!(state.phase, Phase::PlaceRoad) {
                format!("Place road on edge {eid}")
            } else {
                format!("Build road on edge {eid}")
            }
        }
        CITY_START..CITY_END => format!("Build city at node {}", action.city_node().0),
        END_TURN => "End turn".to_string(),
        BUY_DEV_CARD => "Buy development card".to_string(),
        PLAY_KNIGHT => "Play Knight".to_string(),
        PLAY_ROAD_BUILDING => "Play Road Building".to_string(),
        YOP_START..YOP_END => {
            let (r1, r2) = action.year_of_plenty_resources();
            format!("Year of Plenty: {r1} + {r2}")
        }
        MONOPOLY_START..MONOPOLY_END => format!("Monopoly: {}", action.monopoly_resource()),
        ROBBER_START..ROBBER_END => format!("Move robber to tile {}", action.robber_tile().0),
        DISCARD_START..DISCARD_END => format!("Discard {}", action.discard_resource()),
        MARITIME_START..MARITIME_END => {
            let (give, recv) = action.maritime_trade();
            format!("Trade {give} for {recv}")
        }
        _ => format!("Action {}", action.0),
    };

    format!("{player}: {desc}")
}

fn format_phase(phase: &Phase) -> String {
    match phase {
        Phase::PlaceSettlement => "Place Settlement".to_string(),
        Phase::PlaceRoad => "Place Road".to_string(),
        Phase::PreRoll => "Pre-Roll".to_string(),
        Phase::Roll => "Roll".to_string(),
        Phase::Discard { player, remaining } => format!("{player} discards ({remaining} left)"),
        Phase::MoveRobber => "Move Robber".to_string(),
        Phase::StealResolve => "Steal Resolve".to_string(),
        Phase::Main => "Main".to_string(),
        Phase::RoadBuilding { roads_left } => format!("Road Building ({roads_left} left)"),
        Phase::GameOver(p) => format!("Game Over ({p} wins)"),
    }
}

// ─── Render ──────────────────────────────────────────────────────────────────

const RESOURCE_NAMES: [&str; 5] = ["lumber", "brick", "wool", "grain", "ore"];

/// After a dice roll, compute per-player resource deltas and robber-blocked info.
fn format_dice_production(
    state: &GameState,
    hands_before: &[[u8; 5]; 2],
    roll: u8,
    lines: &mut Vec<String>,
) {
    if roll == 7 {
        return;
    }

    let mut any_produced = false;
    for (p, p_name) in [(Player::One, "P1"), (Player::Two, "P2")] {
        let before = &hands_before[p as usize];
        let after = &state.players[p].hand.0;
        let mut gained = Vec::new();
        for r in 0..5 {
            let delta = after[r].saturating_sub(before[r]);
            for _ in 0..delta {
                gained.push(RESOURCE_NAMES[r]);
            }
        }
        if !gained.is_empty() {
            any_produced = true;
            lines.push(format!("{p_name} got {}", gained.join(", ")));
        }
    }

    let mut any_blocked = false;
    for &tid in &state.topology.dice_to_tiles[roll as usize] {
        if tid == state.robber {
            let terrain = state.topology.tiles[tid.0 as usize].terrain;
            if let Some(resource) = terrain.resource() {
                lines.push(format!("Tile {} ({resource}) blocked by robber", tid.0));
                any_blocked = true;
            }
        }
    }

    if !any_produced && !any_blocked {
        lines.push("No resources produced".to_string());
    }
}

/// After a steal resolution, find what was stolen by diffing hands.
fn format_steal_result(state: &GameState, hands_before: &[[u8; 5]; 2], lines: &mut Vec<String>) {
    for (p, p_name) in [(Player::One, "P1"), (Player::Two, "P2")] {
        let before = &hands_before[p as usize];
        let after = &state.players[p].hand.0;
        for r in 0..5 {
            if after[r] > before[r] {
                let victim_name = if p == Player::One { "P2" } else { "P1" };
                lines.push(format!(
                    "{p_name} stole {} from {victim_name}",
                    RESOURCE_NAMES[r]
                ));
                return;
            }
        }
    }
}

/// Render a game log into a self-contained HTML replay file.
///
/// All actions in the log (both chance outcomes and player decisions) are
/// replayed in order. The phase determines how each action is interpreted.
pub fn render(log: &GameLog, output: &Path) {
    let mut state = game::new_game(log.seed, Dice::Random);

    let board = build_board(&state);

    // Frame 0: initial state before any action
    let mut frames = vec![capture_frame(&state, "Game start", 0, None)];
    for &action in &log.actions {
        let player = state.current_player as u8;

        if matches!(state.phase, Phase::Roll) {
            // Dice roll chance outcome
            let roll = (action + 2) as u8;
            let hands_before = [
                state.players[Player::One].hand.0,
                state.players[Player::Two].hand.0,
            ];
            state.apply_action(action);

            let player_name = if player == 0 { "P1" } else { "P2" };
            let mut desc = format!("{player_name}: Roll dice \u{2192} {roll}");
            let mut lines = Vec::new();
            format_dice_production(&state, &hands_before, roll, &mut lines);
            for line in &lines {
                desc.push('\n');
                desc.push_str(line);
            }
            frames.push(capture_frame(&state, &desc, player, Some(roll)));
        } else if matches!(state.phase, Phase::StealResolve) {
            // Steal resolution chance — fold result into previous frame
            let hands_before = [
                state.players[Player::One].hand.0,
                state.players[Player::Two].hand.0,
            ];
            state.apply_action(action);

            let mut lines = Vec::new();
            format_steal_result(&state, &hands_before, &mut lines);
            if let Some(prev) = frames.last_mut() {
                let mut desc = prev.action.clone();
                for line in &lines {
                    desc.push('\n');
                    desc.push_str(line);
                }
                *prev = capture_frame(&state, &desc, prev.player, prev.last_roll);
            }
        } else {
            // Player action
            let action_id = ActionId(action as u8);
            let desc = format_action(action_id, &state);
            state.apply_action(action);
            frames.push(capture_frame(&state, &desc, player, None));
        }
    }

    let result = match &state.phase {
        Phase::GameOver(Player::One) => "P1 wins!".to_string(),
        Phase::GameOver(Player::Two) => "P2 wins!".to_string(),
        _ => "Game in progress".to_string(),
    };

    let data = ReplayData {
        board,
        frames,
        result,
        bot1: "P1".to_string(),
        bot2: "P2".to_string(),
    };

    let json = serde_json::to_string(&data).expect("failed to serialize replay data");
    let html = include_str!("visualize_template.html").replace("\"__REPLAY_DATA__\"", &json);
    std::fs::write(output, html).expect("failed to write HTML file");
}
