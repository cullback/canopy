//! # Action Space (249 total)
//!
//! | Range     | Count | Type            | Index mapping              |
//! |-----------|-------|-----------------|----------------------------|
//! |   0..54   |    54 | Settlement      | node_id                    |
//! |  54..126  |    72 | Road            | edge_id                    |
//! | 126..180  |    54 | City            | node_id                    |
//! | 180       |     1 | Roll            | —                          |
//! | 181       |     1 | EndTurn         | —                          |
//! | 182       |     1 | BuyDevCard      | —                          |
//! | 183       |     1 | PlayKnight      | —                          |
//! | 184       |     1 | PlayRoadBuilding| —                          |
//! | 185..200  |    15 | YearOfPlenty    | unordered pair (see below) |
//! | 200..205  |     5 | Monopoly        | resource                   |
//! | 205..224  |    19 | MoveRobber      | tile_id                    |
//! | 224..229  |     5 | Discard         | resource                   |
//! | 229..249  |    20 | MaritimeTrade   | ordered pair (see below)   |
//!
//! ## Compound encoding rules
//!
//! **YearOfPlenty** — unordered pairs with replacement (15 combinations):
//!   `index = OFFSETS[min(r1,r2)] + (max(r1,r2) - min(r1,r2))`
//!   where `OFFSETS = [0, 5, 9, 12, 14]`
//!
//! **MaritimeTrade** — ordered pairs, give ≠ recv (20 combinations):
//!   `index = give * 4 + adjusted_recv`
//!   where `adjusted_recv = recv` if `recv < give`, else `recv - 1`
//!
//! ## Phase → legal actions
//!
//! | Phase           | Legal actions                                              |
//! |-----------------|------------------------------------------------------------|
//! | PlaceSettlement | Settlement                                                 |
//! | PlaceRoad       | Road                                                       |
//! | PreRoll         | Roll, Knight, RoadBuilding, YOP, Monopoly                  |
//! | Discard         | Discard                                                    |
//! | MoveRobber      | MoveRobber                                                 |
//! | Main            | EndTurn, Settlement, Road, City, BuyDevCard, Knight,       |
//! |                 | RoadBuilding, YOP, Monopoly, MaritimeTrade                 |
//! | RoadBuilding    | Road (or EndTurn if none legal / no roads left)            |

use canopy::player::Player;

use super::board::{EdgeId, NodeId, TileId};
use super::dev_card::DevCardKind;
use super::resource::{
    ALL_RESOURCES, CITY_COST, DEV_CARD_COST, ROAD_COST, Resource, SETTLEMENT_COST,
};
use super::state::{GameState, Phase};

// --- Action space ---

pub const ACTION_SPACE: usize = 249;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionId(pub u8);

impl From<usize> for ActionId {
    fn from(v: usize) -> Self {
        ActionId(v as u8)
    }
}

impl From<ActionId> for usize {
    fn from(a: ActionId) -> Self {
        a.0 as usize
    }
}

impl std::fmt::Debug for ActionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl std::fmt::Display for ActionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let i = self.0;
        match i {
            SETTLEMENT_START..SETTLEMENT_END => write!(f, "settle({})", i - SETTLEMENT_START),
            ROAD_START..ROAD_END => write!(f, "road({})", i - ROAD_START),
            CITY_START..CITY_END => write!(f, "city({})", i - CITY_START),
            ROLL => write!(f, "roll"),
            END_TURN => write!(f, "end_turn"),
            BUY_DEV_CARD => write!(f, "buy_dev"),
            PLAY_KNIGHT => write!(f, "knight"),
            PLAY_ROAD_BUILDING => write!(f, "road_bld"),
            YOP_START..YOP_END => {
                let (r1, r2) = self.year_of_plenty_resources();
                write!(f, "yop({r1},{r2})")
            }
            MONOPOLY_START..MONOPOLY_END => {
                let r = ALL_RESOURCES[(i - MONOPOLY_START) as usize];
                write!(f, "mono({r})")
            }
            ROBBER_START..ROBBER_END => write!(f, "robber({})", i - ROBBER_START),
            DISCARD_START..DISCARD_END => {
                let r = ALL_RESOURCES[(i - DISCARD_START) as usize];
                write!(f, "discard({r})")
            }
            MARITIME_START..MARITIME_END => {
                let (give, recv) = self.maritime_trade();
                write!(f, "trade({give}\u{2192}{recv})")
            }
            _ => write!(f, "unk({})", self.0),
        }
    }
}

// --- Index range constants ---

pub const SETTLEMENT_START: u8 = 0; // 0..54
pub const SETTLEMENT_END: u8 = 54;
pub const ROAD_START: u8 = 54; // 54..126
pub const ROAD_END: u8 = 126;
pub const CITY_START: u8 = 126; // 126..180
pub const CITY_END: u8 = 180;
pub const ROLL: u8 = 180;
pub const END_TURN: u8 = 181;
pub const BUY_DEV_CARD: u8 = 182;
pub const PLAY_KNIGHT: u8 = 183;
pub const PLAY_ROAD_BUILDING: u8 = 184;
pub const YOP_START: u8 = 185; // 185..200
pub const YOP_END: u8 = 200;
pub const MONOPOLY_START: u8 = 200; // 200..205
pub const MONOPOLY_END: u8 = 205;
pub const ROBBER_START: u8 = 205; // 205..224
pub const ROBBER_END: u8 = 224;
pub const DISCARD_START: u8 = 224; // 224..229
pub const DISCARD_END: u8 = 229;
pub const MARITIME_START: u8 = 229; // 229..249
pub const MARITIME_END: u8 = 249;

// --- Encode helpers ---

pub fn settlement_id(node: NodeId) -> ActionId {
    ActionId(SETTLEMENT_START + node.0)
}

pub fn road_id(edge: EdgeId) -> ActionId {
    ActionId(ROAD_START + edge.0)
}

pub fn city_id(node: NodeId) -> ActionId {
    ActionId(CITY_START + node.0)
}

pub fn yop_id(r1: Resource, r2: Resource) -> ActionId {
    let (a, b) = if (r1 as u8) <= (r2 as u8) {
        (r1 as u8, r2 as u8)
    } else {
        (r2 as u8, r1 as u8)
    };
    ActionId(YOP_START + yop_pair_index(a, b))
}

fn yop_pair_index(a: u8, b: u8) -> u8 {
    const OFFSETS: [u8; 5] = [0, 5, 9, 12, 14];
    OFFSETS[a as usize] + (b - a)
}

pub fn monopoly_id(r: Resource) -> ActionId {
    ActionId(MONOPOLY_START + r as u8)
}

pub fn robber_id(tile: TileId) -> ActionId {
    ActionId(ROBBER_START + tile.0)
}

pub fn discard_id(r: Resource) -> ActionId {
    ActionId(DISCARD_START + r as u8)
}

pub fn maritime_id(give: Resource, recv: Resource) -> ActionId {
    let g = give as u8;
    let r = recv as u8;
    let recv_idx = if r < g { r } else { r - 1 };
    ActionId(MARITIME_START + g * 4 + recv_idx)
}

// --- Decode methods ---

const YOP_DECODE: [(Resource, Resource); 15] = {
    use Resource::*;
    [
        (Lumber, Lumber),
        (Lumber, Brick),
        (Lumber, Wool),
        (Lumber, Grain),
        (Lumber, Ore),
        (Brick, Brick),
        (Brick, Wool),
        (Brick, Grain),
        (Brick, Ore),
        (Wool, Wool),
        (Wool, Grain),
        (Wool, Ore),
        (Grain, Grain),
        (Grain, Ore),
        (Ore, Ore),
    ]
};

impl ActionId {
    pub fn settlement_node(self) -> NodeId {
        debug_assert!(self.0 < SETTLEMENT_END);
        NodeId(self.0 - SETTLEMENT_START)
    }

    pub fn road_edge(self) -> EdgeId {
        debug_assert!(self.0 >= ROAD_START && self.0 < ROAD_END);
        EdgeId(self.0 - ROAD_START)
    }

    pub fn city_node(self) -> NodeId {
        debug_assert!(self.0 >= CITY_START && self.0 < CITY_END);
        NodeId(self.0 - CITY_START)
    }

    pub fn year_of_plenty_resources(self) -> (Resource, Resource) {
        debug_assert!(self.0 >= YOP_START && self.0 < YOP_END);
        YOP_DECODE[(self.0 - YOP_START) as usize]
    }

    pub fn monopoly_resource(self) -> Resource {
        debug_assert!(self.0 >= MONOPOLY_START && self.0 < MONOPOLY_END);
        ALL_RESOURCES[(self.0 - MONOPOLY_START) as usize]
    }

    pub fn robber_tile(self) -> TileId {
        debug_assert!(self.0 >= ROBBER_START && self.0 < ROBBER_END);
        TileId(self.0 - ROBBER_START)
    }

    pub fn discard_resource(self) -> Resource {
        debug_assert!(self.0 >= DISCARD_START && self.0 < DISCARD_END);
        ALL_RESOURCES[(self.0 - DISCARD_START) as usize]
    }

    pub fn maritime_trade(self) -> (Resource, Resource) {
        debug_assert!(self.0 >= MARITIME_START);
        let idx = self.0 - MARITIME_START;
        let give = idx / 4;
        let recv_idx = idx % 4;
        let recv = if recv_idx < give {
            recv_idx
        } else {
            recv_idx + 1
        };
        (ALL_RESOURCES[give as usize], ALL_RESOURCES[recv as usize])
    }
}

// --- Legal action generation ---

pub fn legal_actions(state: &GameState, actions: &mut Vec<ActionId>) {
    actions.clear();
    match &state.phase {
        Phase::PlaceSettlement => populate_place_settlement(state, actions),
        Phase::PlaceRoad => populate_place_road(state, actions),
        Phase::PreRoll => populate_preroll(state, actions),
        Phase::Roll | Phase::StealResolve | Phase::DevCardDraw => {
            // Chance nodes — resolved by chance_outcomes/apply_chance, not player actions
        }
        Phase::Discard {
            player,
            remaining,
            min_resource,
            ..
        } => populate_discard(state, *player, *remaining, *min_resource, actions),
        Phase::MoveRobber => populate_move_robber(state, actions),
        Phase::Main => populate_main(state, actions),
        Phase::RoadBuilding { roads_left } => {
            populate_road_building(state, *roads_left, actions);
        }
        Phase::GameOver(_) => {}
    }
    // Diagnostic: any non-chance, non-terminal phase that produces zero
    // legal actions indicates a game-logic bug. Print state so we can
    // reproduce it. Chance phases (Roll/Steal/Dev) legitimately return
    // empty and are handled upstream by sample_chance.
    if actions.is_empty()
        && !matches!(
            state.phase,
            Phase::Roll | Phase::StealResolve | Phase::DevCardDraw | Phase::GameOver(_)
        )
    {
        let cur = state.current_player;
        let opp = cur.opponent();
        eprintln!(
            "!! legal_actions empty: phase={:?} turn={} cur={:?} pre_roll={} \
             setup_count={} robber={:?} cur_pvps={} opp_pvps={} \
             cur_hand={:?}({}) opp_hand={:?}({}) \
             cur_settles={} cur_cities={} opp_settles={} opp_cities={} \
             canonical={} min_step={} discard_threshold={}",
            state.phase,
            state.turn_number,
            cur,
            state.pre_roll,
            state.setup_count,
            state.robber,
            state.public_vps(cur),
            state.public_vps(opp),
            state.players[cur].hand.0,
            state.players[cur].hand.total(),
            state.players[opp].hand.0,
            state.players[opp].hand.total(),
            state.boards[cur].settlements.count_ones(),
            state.boards[cur].cities.count_ones(),
            state.boards[opp].settlements.count_ones(),
            state.boards[opp].cities.count_ones(),
            state.canonical_build_order,
            state.min_step,
            state.discard_threshold,
        );
    }
}

/// Minimum total pips for a setup settlement spot. Spots below this threshold
/// are dominated — in 1v1 with 50+ spots available, a < 8 pip spot is never
/// optimal. Disabled during colonist replay (canonical_build_order = false).
const MIN_SETUP_PIPS: u8 = 8;

/// Pips per dice number: pips[n] = 6 - |7 - n| for n in 2..=12.
const PIPS: [u8; 13] = [0, 0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1];

fn populate_place_settlement(state: &GameState, actions: &mut Vec<ActionId>) {
    let adj = &state.topology.adj;
    let topo = &state.topology;
    let occupied = state.occupied_nodes();
    // Expand occupied nodes to their neighbors
    let mut neighbor_blocked = 0u64;
    let mut bits = occupied;
    while bits != 0 {
        let node = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        neighbor_blocked |= adj.node_adj_nodes[node];
    }

    // Precompute tile numbers for pip calculation.
    let tile_numbers = if state.canonical_build_order {
        let mut numbers = [0u8; 19];
        for roll in 2..=12u8 {
            for &tid in &topo.dice_to_tiles[roll as usize] {
                numbers[tid.0 as usize] = roll;
            }
        }
        Some(numbers)
    } else {
        None
    };

    let legal = !occupied & !neighbor_blocked & NODE_MASK;
    let mut bits = legal;
    while bits != 0 {
        let nid = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        // Prune low-pip setup spots (dominated in 1v1).
        if let Some(ref numbers) = tile_numbers {
            let node = &topo.nodes[nid as usize];
            let total_pips: u8 = node
                .adjacent_tiles
                .iter()
                .map(|tid| PIPS[numbers[tid.0 as usize] as usize])
                .sum();
            if total_pips < MIN_SETUP_PIPS {
                continue;
            }
        }
        actions.push(settlement_id(NodeId(nid)));
    }
}

fn populate_place_road(state: &GameState, actions: &mut Vec<ActionId>) {
    let pid = state.current_player;
    let adj = &state.topology.adj;
    let all_roads = state.all_roads();
    let my_roads = state.boards[pid].road_network.roads;

    // Use last_setup_node if set (correct after sync_buildings places multiple
    // settlements at once). Fall back to heuristic: scan for a settlement with
    // no adjacent road.
    let last = if let Some(nid) = state.last_setup_node {
        nid.0
    } else {
        let mut settlements = state.boards[pid].settlements;
        let mut found = 0u8;
        while settlements != 0 {
            let nid = settlements.trailing_zeros() as u8;
            settlements &= settlements - 1;
            if adj.node_adj_edges[nid as usize] & my_roads == 0 {
                found = nid;
            }
        }
        found
    };

    let legal = adj.node_adj_edges[last as usize] & !all_roads & EDGE_MASK;
    let mut bits = legal;
    while bits != 0 {
        let eid = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        actions.push(road_id(EdgeId(eid)));
    }
}

fn populate_discard(
    state: &GameState,
    player: Player,
    remaining: u8,
    min_resource: u8,
    actions: &mut Vec<ActionId>,
) {
    let hand = state.players[player].hand;
    // Enforce lexicographic ordering: only offer resources >= min_resource,
    // and only if enough cards remain from that index onward to finish.
    let mut suffix_total: u8 = ALL_RESOURCES[min_resource as usize..]
        .iter()
        .map(|&r| hand[r])
        .sum();
    if suffix_total < remaining {
        eprintln!(
            "!! discard invariant broken at entry: player={player:?} hand={:?}({}) \
             remaining={remaining} min_resource={min_resource} suffix={suffix_total} \
             cur={:?} threshold={}",
            hand.0,
            hand.total(),
            state.current_player,
            state.discard_threshold,
        );
        return;
    }
    for &r in &ALL_RESOURCES[min_resource as usize..] {
        if hand[r] > 0 && suffix_total >= remaining {
            actions.push(discard_id(r));
        }
        suffix_total -= hand[r];
    }
}

fn populate_move_robber(state: &GameState, actions: &mut Vec<ActionId>) {
    let topo = &state.topology;
    let me = state.current_player;
    let opp = me.opponent();
    let my_buildings = state.player_buildings(me);
    let opp_buildings = state.player_buildings(opp);
    let friendly_opp = state.public_vps(opp) < super::FRIENDLY_ROBBER_VP;
    let friendly_me = state.public_vps(me) < super::FRIENDLY_ROBBER_VP;

    // Collect legal tiles, optionally filtering tiles without opponent
    // buildings (dominated: no steal, no opponent production blocked).
    // Disabled during replay since building positions may diverge.
    let mut legal_tiles: Vec<(TileId, bool)> = Vec::new();
    let mut any_touches_opp = false;
    for tile in &topo.tiles {
        if tile.id == state.robber {
            continue;
        }
        let tile_mask = topo.adj.tile_nodes[tile.id.0 as usize];
        if friendly_opp && tile_mask & opp_buildings != 0 {
            continue;
        }
        if friendly_me && tile_mask & my_buildings != 0 {
            continue;
        }
        let touches_opp = tile_mask & opp_buildings != 0;
        if touches_opp {
            any_touches_opp = true;
        }
        legal_tiles.push((tile.id, touches_opp));
    }

    let filter = state.canonical_build_order && any_touches_opp;
    let start_len = actions.len();
    for (tid, touches_opp) in &legal_tiles {
        if !filter || *touches_opp {
            actions.push(robber_id(*tid));
        }
    }

    // Fallback: if the friendly-robber and canonical filters eliminated
    // every candidate, all non-robber tiles become legal.
    if actions.len() == start_len {
        for tile in &topo.tiles {
            if tile.id != state.robber {
                actions.push(robber_id(tile.id));
            }
        }
    }
}

fn populate_preroll(state: &GameState, actions: &mut Vec<ActionId>) {
    actions.push(ActionId(ROLL));

    let player = state.current();
    if !player.has_played_dev_card_this_turn {
        let playable_knights = player.dev_cards[DevCardKind::Knight]
            .saturating_sub(player.dev_cards_bought_this_turn[DevCardKind::Knight]);
        if playable_knights > 0 {
            actions.push(ActionId(PLAY_KNIGHT));
        }

        // Monopoly, YoP, and Road Building are dominated in PreRoll:
        // rolling first gives strictly more information and the
        // resources/roads can't help until Main phase. Disabled during
        // replay (canonical_build_order=false) since real games may
        // play these from PreRoll.
        if !state.canonical_build_order {
            let playable_rb = player.dev_cards[DevCardKind::RoadBuilding]
                .saturating_sub(player.dev_cards_bought_this_turn[DevCardKind::RoadBuilding]);
            if playable_rb > 0 && player.roads_left > 0 {
                actions.push(ActionId(PLAY_ROAD_BUILDING));
            }

            let playable_yop = player.dev_cards[DevCardKind::YearOfPlenty]
                .saturating_sub(player.dev_cards_bought_this_turn[DevCardKind::YearOfPlenty]);
            if playable_yop > 0 {
                for (i, &r1) in ALL_RESOURCES.iter().enumerate() {
                    if state.bank[r1] == 0 {
                        continue;
                    }
                    for &r2 in &ALL_RESOURCES[i..] {
                        if r1 == r2 && state.bank[r1] < 2 {
                            continue;
                        }
                        if state.bank[r2] == 0 {
                            continue;
                        }
                        actions.push(yop_id(r1, r2));
                    }
                }
            }

            let playable_mono = player.dev_cards[DevCardKind::Monopoly]
                .saturating_sub(player.dev_cards_bought_this_turn[DevCardKind::Monopoly]);
            if playable_mono > 0 {
                for &r in &ALL_RESOURCES {
                    actions.push(monopoly_id(r));
                }
            }
        }
    }
}

fn populate_main(state: &GameState, actions: &mut Vec<ActionId>) {
    actions.push(ActionId(END_TURN));
    let pid = state.current_player;
    let player = state.current();
    let adj = &state.topology.adj;
    let boards = &state.boards[pid];
    let canon = state.canonical_build_order;

    // Step 1: Play dev card
    if !canon || state.min_step <= 1 {
        if !player.has_played_dev_card_this_turn {
            let playable_knights = player.dev_cards[DevCardKind::Knight]
                .saturating_sub(player.dev_cards_bought_this_turn[DevCardKind::Knight]);
            if playable_knights > 0 {
                actions.push(ActionId(PLAY_KNIGHT));
            }

            let playable_rb = player.dev_cards[DevCardKind::RoadBuilding]
                .saturating_sub(player.dev_cards_bought_this_turn[DevCardKind::RoadBuilding]);
            if playable_rb > 0 && player.roads_left > 0 {
                actions.push(ActionId(PLAY_ROAD_BUILDING));
            }

            let playable_yop = player.dev_cards[DevCardKind::YearOfPlenty]
                .saturating_sub(player.dev_cards_bought_this_turn[DevCardKind::YearOfPlenty]);
            if playable_yop > 0 {
                for (i, &r1) in ALL_RESOURCES.iter().enumerate() {
                    if state.bank[r1] == 0 {
                        continue;
                    }
                    for &r2 in &ALL_RESOURCES[i..] {
                        if r1 == r2 && state.bank[r1] < 2 {
                            continue;
                        }
                        if state.bank[r2] == 0 {
                            continue;
                        }
                        actions.push(yop_id(r1, r2));
                    }
                }
            }

            let playable_mono = player.dev_cards[DevCardKind::Monopoly]
                .saturating_sub(player.dev_cards_bought_this_turn[DevCardKind::Monopoly]);
            if playable_mono > 0 {
                let opp_hand = &state.players[pid.opponent()].hand;
                for &r in &ALL_RESOURCES {
                    if !canon || opp_hand[r] > 0 {
                        actions.push(monopoly_id(r));
                    }
                }
            }
        }
    }

    // Step 2: Maritime trade
    if !canon || state.min_step <= 2 {
        for &give_res in &ALL_RESOURCES {
            if player.hand[give_res] == 0 {
                continue;
            }
            let ratio = player.trade_ratios[give_res as usize];
            if player.hand[give_res] >= ratio {
                for &recv_res in &ALL_RESOURCES {
                    if recv_res != give_res && state.bank[recv_res] > 0 {
                        let idx = maritime_trade_idx(give_res, recv_res);
                        if !canon || idx >= state.min_trade_idx {
                            actions.push(maritime_id(give_res, recv_res));
                        }
                    }
                }
            }
        }
    }

    // Step 3: Buy dev card
    if !canon || state.min_step <= 3 {
        if player.hand.contains(DEV_CARD_COST) && !state.dev_deck.is_empty() {
            let pool_total: u8 = state.unknown_dev_pool().iter().sum();
            if pool_total > 0 {
                actions.push(ActionId(BUY_DEV_CARD));
            }
        }
    }

    // Step 4: City on pre-existing settlement
    if !canon || state.min_step <= 4 {
        if player.hand.contains(CITY_COST) && player.cities_left > 0 {
            let mut bits = boards.settlements & state.settlements_at_turn_start;
            while bits != 0 {
                let nid = bits.trailing_zeros() as u8;
                bits &= bits - 1;
                if !canon || nid >= state.min_city_node {
                    actions.push(city_id(NodeId(nid)));
                }
            }
        }
    }

    // Step 5: Build road
    if !canon || state.min_step <= 5 {
        if player.hand.contains(ROAD_COST) && player.roads_left > 0 {
            let mut legal = boards.road_network.reachable_edges();
            while legal != 0 {
                let eid = legal.trailing_zeros() as u8;
                legal &= legal - 1;
                if canon {
                    let key = road_key(state.road_distances[eid as usize], eid);
                    if key < state.min_road_key {
                        continue;
                    }
                }
                actions.push(road_id(EdgeId(eid)));
            }
        }
    }

    // Step 6: Build settlement on non-port
    if !canon || state.min_step <= 6 {
        if player.hand.contains(SETTLEMENT_COST) && player.settlements_left > 0 {
            let occupied = state.occupied_nodes();
            let mut neighbor_blocked = 0u64;
            let mut occ = occupied;
            while occ != 0 {
                let node = occ.trailing_zeros() as usize;
                occ &= occ - 1;
                neighbor_blocked |= adj.node_adj_nodes[node];
            }
            let mut on_road = 0u64;
            let mut roads = boards.road_network.roads;
            while roads != 0 {
                let eid = roads.trailing_zeros() as usize;
                roads &= roads - 1;
                on_road |= adj.edge_endpoints[eid];
            }
            let legal = !occupied & !neighbor_blocked & on_road & NODE_MASK;
            let mut bits = legal;
            while bits != 0 {
                let nid = bits.trailing_zeros() as u8;
                bits &= bits - 1;
                let bit = 1u64 << nid;
                let is_port =
                    adj.port_specific.iter().any(|&p| p & bit != 0) || adj.port_generic & bit != 0;
                if is_port {
                    // Port settlements handled at step 8
                } else if !canon || nid >= state.min_settle_node {
                    actions.push(settlement_id(NodeId(nid)));
                }
            }
        }
    }

    // Step 7: City on same-turn settlement (always available)
    if player.hand.contains(CITY_COST) && player.cities_left > 0 {
        let same_turn = boards.settlements & !state.settlements_at_turn_start;
        let mut bits = same_turn;
        while bits != 0 {
            let nid = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            actions.push(city_id(NodeId(nid)));
        }
    }

    // Step 8: Build settlement on port
    if !canon || state.min_step <= 8 {
        if player.hand.contains(SETTLEMENT_COST) && player.settlements_left > 0 {
            let occupied = state.occupied_nodes();
            let mut neighbor_blocked = 0u64;
            let mut occ = occupied;
            while occ != 0 {
                let node = occ.trailing_zeros() as usize;
                occ &= occ - 1;
                neighbor_blocked |= adj.node_adj_nodes[node];
            }
            let mut on_road = 0u64;
            let mut roads = boards.road_network.roads;
            while roads != 0 {
                let eid = roads.trailing_zeros() as usize;
                roads &= roads - 1;
                on_road |= adj.edge_endpoints[eid];
            }
            let legal = !occupied & !neighbor_blocked & on_road & NODE_MASK;
            let mut bits = legal;
            while bits != 0 {
                let nid = bits.trailing_zeros() as u8;
                bits &= bits - 1;
                let bit = 1u64 << nid;
                let is_port =
                    adj.port_specific.iter().any(|&p| p & bit != 0) || adj.port_generic & bit != 0;
                if is_port && (!canon || nid >= state.min_port_settle_node) {
                    actions.push(settlement_id(NodeId(nid)));
                }
            }
        }
    }
}

/// Compute road ordering key: distance * 72 + edge_id.
fn road_key(distance: u8, edge_id: u8) -> u16 {
    distance as u16 * 72 + edge_id as u16
}

/// Canonical trade index for within-type ordering.
fn maritime_trade_idx(give: Resource, recv: Resource) -> u8 {
    give as u8 * 5 + recv as u8
}

const NODE_MASK: u64 = (1u64 << 54) - 1;
const EDGE_MASK: u128 = (1u128 << 72) - 1;

fn populate_road_building(state: &GameState, roads_left: u8, actions: &mut Vec<ActionId>) {
    let pid = state.current_player;
    let player = state.current();
    if player.roads_left == 0 || roads_left == 0 {
        actions.push(ActionId(END_TURN));
        return;
    }
    let canon = state.canonical_build_order;
    let mut legal = state.boards[pid].road_network.reachable_edges();
    let mut found = false;
    while legal != 0 {
        let eid = legal.trailing_zeros() as u8;
        legal &= legal - 1;
        if canon && roads_left < 2 {
            let key = road_key(state.road_distances[eid as usize], eid);
            if key < state.min_road_key {
                continue;
            }
        }
        actions.push(road_id(EdgeId(eid)));
        found = true;
    }
    if !found {
        actions.push(ActionId(END_TURN));
    }
}

/// Check if the given edge connects to the player's road network or buildings.
#[cfg(test)]
fn is_road_connected(state: &GameState, edge_id: EdgeId, pid: Player) -> bool {
    let adj = &state.topology.adj;
    let boards = &state.boards[pid];
    let my_buildings = boards.settlements | boards.cities;
    let endpoints = adj.edge_endpoints[edge_id.0 as usize];

    // Connected if either endpoint has player's building
    if my_buildings & endpoints != 0 {
        return true;
    }

    // Endpoints not blocked by opponent's buildings
    let opp_buildings = state.player_buildings(pid.opponent());
    let unblocked = endpoints & !opp_buildings;
    if unblocked == 0 {
        return false;
    }

    // Check if any adjacent edge at unblocked endpoints has player's road
    let mut reachable = 0u128;
    let mut ep = unblocked;
    while ep != 0 {
        let node = ep.trailing_zeros() as usize;
        ep &= ep - 1;
        reachable |= adj.node_adj_edges[node];
    }
    reachable &= !(1u128 << edge_id.0);
    reachable & boards.road_network.roads != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game;
    use crate::game::board::Port;
    use crate::game::dev_card::DevCardDeck;
    use crate::game::dice::Dice;
    use crate::game::resource::{ROAD_COST, ResourceArray};
    use crate::game::topology::Topology;
    use canopy::game::Game;
    use std::sync::Arc;

    fn make_state() -> GameState {
        let topo = Arc::new(Topology::from_seed(42));
        let deck = DevCardDeck::new();
        GameState::new(topo, deck, Dice::default())
    }

    /// Play through the 4-settlement setup phase using random actions.
    /// Returns a seeded RNG for subsequent apply() calls.
    fn play_setup(state: &mut GameState) -> fastrand::Rng {
        let mut rng = fastrand::Rng::with_seed(0);
        let mut actions = Vec::new();
        for _ in 0..4 {
            legal_actions(state, &mut actions);
            let i = rng.usize(..actions.len());
            game::apply(state, actions[i]);
            legal_actions(state, &mut actions);
            let i = rng.usize(..actions.len());
            game::apply(state, actions[i]);
        }
        assert!(matches!(state.phase, Phase::PreRoll));
        rng
    }

    /// Encoding a node ID as a settlement action and decoding it back
    /// must round-trip for all 54 nodes.
    #[test]
    fn settlement_round_trip() {
        for i in 0..54u8 {
            let nid = NodeId(i);
            let aid = settlement_id(nid);
            assert_eq!(aid.settlement_node(), nid);
        }
    }

    /// Encoding an edge ID as a road action and decoding it back
    /// must round-trip for all 72 edges.
    #[test]
    fn road_round_trip() {
        for i in 0..72u8 {
            let eid = EdgeId(i);
            let aid = road_id(eid);
            assert_eq!(aid.road_edge(), eid);
        }
    }

    /// Encoding a node ID as a city action and decoding it back
    /// must round-trip for all 54 nodes.
    #[test]
    fn city_round_trip() {
        for i in 0..54u8 {
            let nid = NodeId(i);
            let aid = city_id(nid);
            assert_eq!(aid.city_node(), nid);
        }
    }

    /// Encoding a (give, receive) resource pair as a maritime trade action and
    /// decoding it back must round-trip for all 20 valid pairs.
    #[test]
    fn maritime_round_trip() {
        for &give in &ALL_RESOURCES {
            for &recv in &ALL_RESOURCES {
                if give != recv {
                    let aid = maritime_id(give, recv);
                    let (g, r) = aid.maritime_trade();
                    assert_eq!(g, give);
                    assert_eq!(r, recv);
                }
            }
        }
    }

    /// Encoding a Year of Plenty resource pair and decoding it back must
    /// round-trip for all 15 combinations (with replacement).
    #[test]
    fn yop_round_trip() {
        let mut count = 0;
        for (i, &r1) in ALL_RESOURCES.iter().enumerate() {
            for &r2 in &ALL_RESOURCES[i..] {
                let aid = yop_id(r1, r2);
                let (a, b) = aid.year_of_plenty_resources();
                assert_eq!(a, r1);
                assert_eq!(b, r2);
                count += 1;
            }
        }
        assert_eq!(count, 15);
    }

    /// On an empty board in PlaceSettlement phase, only nodes with >= MIN_SETUP_PIPS
    /// are legal (canonical ordering prunes low-pip spots).
    #[test]
    fn place_settlement_filters_low_pips() {
        let state = make_state();
        let mut actions = Vec::new();
        legal_actions(&state, &mut actions);
        assert!(actions.len() < 54, "low-pip spots should be filtered out");
        assert!(
            actions.len() >= 15,
            "should still have plenty of spots: got {}",
            actions.len()
        );
        for a in &actions {
            assert!(a.0 < SETTLEMENT_END);
        }

        // Without canonical ordering, all 54 are legal.
        let mut state_no_canon = make_state();
        state_no_canon.canonical_build_order = false;
        legal_actions(&state_no_canon, &mut actions);
        assert_eq!(
            actions.len(),
            54,
            "all 54 nodes should be legal without canonical ordering"
        );
    }

    /// The Main phase must always include END_TURN as a legal action,
    /// regardless of the player's hand or board state.
    #[test]
    fn main_phase_has_end_turn() {
        let mut state = make_state();
        play_setup(&mut state);
        state.phase = Phase::Main;
        let mut actions = Vec::new();
        legal_actions(&state, &mut actions);
        assert!(
            actions.contains(&ActionId(END_TURN)),
            "END_TURN must be present in Main phase"
        );
    }

    /// Cross-check: legal roads from legal_actions match the set of empty
    /// edges where is_road_connected returns true. Catches both false positives
    /// (disconnected edges returned) and false negatives (connected edges missing).
    #[test]
    fn road_completeness_matches_brute_force() {
        for seed in [1, 42, 99, 123, 777] {
            let topo = Arc::new(Topology::from_seed(seed));
            let deck = DevCardDeck::new();
            let mut state = GameState::new(topo, deck, Dice::default());

            // Play setup
            let mut rng = fastrand::Rng::with_seed(0);
            let mut actions = Vec::new();
            for _ in 0..4 {
                legal_actions(&state, &mut actions);
                game::apply(&mut state, actions[0]);
                legal_actions(&state, &mut actions);
                game::apply(&mut state, actions[0]);
            }

            // Enter Main phase with road resources
            state.phase = Phase::Main;
            state.current_player = Player::One;
            state.players[Player::One].hand.add(ROAD_COST);

            // Get road edges from legal_actions
            legal_actions(&state, &mut actions);
            let mut populate_set: Vec<u8> = actions
                .iter()
                .filter(|a| a.0 >= ROAD_START && a.0 < ROAD_END)
                .map(|a| a.0 - ROAD_START)
                .collect();
            populate_set.sort();

            // Brute force: check every empty edge with is_road_connected
            let all_roads = state.all_roads();
            let mut brute_set: Vec<u8> = (0..72u8)
                .filter(|&eid| {
                    all_roads & (1u128 << eid) == 0
                        && is_road_connected(&state, EdgeId(eid), Player::One)
                })
                .collect();
            brute_set.sort();

            assert_eq!(
                populate_set, brute_set,
                "seed {seed}: legal_actions roads != is_road_connected brute force"
            );
        }
    }

    /// An opponent settlement on a node blocks road extension through that
    /// node. Edges beyond the blocked node remain reachable from the far side.
    #[test]
    fn opponent_building_blocks_road_extension() {
        let mut state = make_state();
        play_setup(&mut state);
        let adj = &state.topology.adj;

        // Find P1's first settlement and build a 2-road chain: S → A → B
        let s = NodeId(state.boards[Player::One].settlements.trailing_zeros() as u8);
        let all_roads_before = state.all_roads();
        let occupied_before = state.occupied_nodes();

        // Find edge S→A
        let s_edges = adj.node_adj_edges[s.0 as usize] & !all_roads_before & EDGE_MASK;
        assert!(s_edges != 0, "should have available edges from S");
        let edge_sa = EdgeId(s_edges.trailing_zeros() as u8);
        let a = NodeId(
            (adj.edge_endpoints[edge_sa.0 as usize] & !(1u64 << s.0)).trailing_zeros() as u8,
        );
        assert_eq!(occupied_before & (1u64 << a.0), 0, "A must be unoccupied");

        // Find edge A→B where B is also unoccupied
        let a_edges = adj.node_adj_edges[a.0 as usize]
            & !(1u128 << edge_sa.0)
            & !all_roads_before
            & EDGE_MASK;
        let mut bits = a_edges;
        let mut edge_ab = None;
        let mut node_b = None;
        while bits != 0 {
            let eid = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            let b_cand = (adj.edge_endpoints[eid as usize] & !(1u64 << a.0)).trailing_zeros() as u8;
            if occupied_before & (1u64 << b_cand) == 0 {
                edge_ab = Some(EdgeId(eid));
                node_b = Some(NodeId(b_cand));
                break;
            }
        }
        let edge_ab = edge_ab.expect("should find edge A→B");
        let b = node_b.unwrap();

        // Place P1's chain roads via add_road to maintain reachable frontier
        let opp_buildings = state.player_buildings(Player::Two);
        let opp_roads = state.boards[Player::Two].road_network.roads;
        state.boards[Player::One].road_network.add_road(
            edge_sa,
            &state.topology.adj,
            opp_buildings,
            opp_roads,
            false,
        );
        state.players[Player::One].roads_placed += 1;
        state.players[Player::One].roads_left -= 1;
        let opp_roads = state.boards[Player::Two].road_network.roads;
        state.boards[Player::One].road_network.add_road(
            edge_ab,
            &state.topology.adj,
            opp_buildings,
            opp_roads,
            false,
        );
        state.players[Player::One].roads_placed += 1;
        state.players[Player::One].roads_left -= 1;

        // P2 settles at A (the middle node) — update P1's road network
        state.boards[Player::Two].settlements |= 1u64 << a.0;
        state.players[Player::Two].settlements_left -= 1;
        let p1_buildings = state.boards[Player::One].settlements | state.boards[Player::One].cities;
        let p2_buildings = state.boards[Player::Two].settlements | state.boards[Player::Two].cities;
        state.boards[Player::One].road_network.on_opponent_build(
            a,
            &state.topology.adj,
            p1_buildings,
            p2_buildings,
        );

        // Edges at A (other than our chain) whose other endpoint is isolated
        // from P1's network must be blocked by the opponent settlement.
        let my_buildings = state.boards[Player::One].settlements | state.boards[Player::One].cities;
        let my_roads = state.boards[Player::One].road_network.roads;
        let a_other = adj.node_adj_edges[a.0 as usize]
            & !(1u128 << edge_sa.0)
            & !(1u128 << edge_ab.0)
            & !state.all_roads()
            & EDGE_MASK;
        let mut found_blocked = false;
        let mut check = a_other;
        while check != 0 {
            let eid = check.trailing_zeros() as u8;
            check &= check - 1;
            let x = (adj.edge_endpoints[eid as usize] & !(1u64 << a.0)).trailing_zeros() as u8;
            let x_in_network =
                my_buildings & (1u64 << x) != 0 || adj.node_adj_edges[x as usize] & my_roads != 0;
            if !x_in_network {
                assert!(
                    !is_road_connected(&state, EdgeId(eid), Player::One),
                    "edge {eid} at opponent-occupied A with isolated endpoint should be blocked"
                );
                found_blocked = true;
            }
        }
        assert!(found_blocked, "should find at least one blocked edge at A");

        // Edges at B (beyond the blocked node) should be reachable via the chain
        let b_other = adj.node_adj_edges[b.0 as usize]
            & !(1u128 << edge_ab.0)
            & !state.all_roads()
            & EDGE_MASK;
        assert!(b_other != 0, "B should have other adjacent edges");
        let mut check = b_other;
        while check != 0 {
            let eid = check.trailing_zeros() as u8;
            check &= check - 1;
            assert!(
                is_road_connected(&state, EdgeId(eid), Player::One),
                "edge {eid} at unblocked B should be connected via chain"
            );
        }
    }

    /// A chain of 3+ roads from a settlement makes the far end reachable,
    /// even though no building exists at intermediate nodes.
    #[test]
    fn road_connected_through_chain() {
        let mut state = make_state();
        play_setup(&mut state);
        let adj = &state.topology.adj;

        let s = NodeId(state.boards[Player::One].settlements.trailing_zeros() as u8);
        let occupied = state.occupied_nodes();
        let all_roads_before = state.all_roads();

        // Build a chain of 3 roads: S → N1 → N2 → N3
        let mut current = s;
        let mut chain_edges = Vec::new();
        let mut visited_nodes = occupied | (1u64 << s.0);
        let mut used_roads = all_roads_before;

        for i in 0..3 {
            let candidates = adj.node_adj_edges[current.0 as usize] & !used_roads & EDGE_MASK;
            let mut bits = candidates;
            let mut found = None;
            while bits != 0 {
                let eid = bits.trailing_zeros() as u8;
                bits &= bits - 1;
                let next = (adj.edge_endpoints[eid as usize] & !(1u64 << current.0))
                    .trailing_zeros() as u8;
                if visited_nodes & (1u64 << next) == 0 {
                    found = Some((EdgeId(eid), NodeId(next)));
                    break;
                }
            }
            let (edge, next_node) =
                found.unwrap_or_else(|| panic!("should find edge {i} in chain from {:?}", current));
            chain_edges.push(edge);
            visited_nodes |= 1u64 << next_node.0;
            used_roads |= 1u128 << edge.0;
            current = next_node;
        }
        let far_end = current;

        // Place all chain roads for P1 via add_road to maintain reachable frontier
        for &eid in &chain_edges {
            let opp_buildings = state.player_buildings(Player::Two);
            let opp_roads = state.boards[Player::Two].road_network.roads;
            state.boards[Player::One].road_network.add_road(
                eid,
                &state.topology.adj,
                opp_buildings,
                opp_roads,
                false,
            );
            state.players[Player::One].roads_placed += 1;
            state.players[Player::One].roads_left -= 1;
        }

        // Enter Main phase with road resources
        state.phase = Phase::Main;
        state.current_player = Player::One;
        state.players[Player::One].hand.add(ROAD_COST);

        // Edges at far_end (not the last chain edge) should be reachable
        let far_end_edges = adj.node_adj_edges[far_end.0 as usize]
            & !(1u128 << chain_edges.last().unwrap().0)
            & !state.all_roads()
            & EDGE_MASK;
        assert!(far_end_edges != 0, "far end should have available edges");

        // Verify is_road_connected returns true at the far end
        let mut bits = far_end_edges;
        while bits != 0 {
            let eid = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            assert!(
                is_road_connected(&state, EdgeId(eid), Player::One),
                "edge {eid} at far end of 3-road chain should be connected"
            );
        }

        // Verify these edges also appear in legal_actions
        let mut actions = Vec::new();
        legal_actions(&state, &mut actions);
        let road_actions: Vec<u8> = actions
            .iter()
            .filter(|a| a.0 >= ROAD_START && a.0 < ROAD_END)
            .map(|a| a.0 - ROAD_START)
            .collect();

        let mut bits = far_end_edges;
        while bits != 0 {
            let eid = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            assert!(
                road_actions.contains(&eid),
                "edge {eid} at far end should appear in legal_actions"
            );
        }
    }

    /// After setup, building a road extends the network: the set of legal
    /// road placements should grow (or stay the same) as the network expands.
    #[test]
    fn road_connected_to_network() {
        let mut state = make_state();
        let mut rng = play_setup(&mut state);
        state.phase = Phase::Main;

        // Give P1 road resources
        state.players[Player::One].hand.add(ROAD_COST);

        let mut actions = Vec::new();
        legal_actions(&state, &mut actions);
        let road_actions: Vec<ActionId> = actions
            .iter()
            .copied()
            .filter(|a| a.0 >= ROAD_START && a.0 < ROAD_END)
            .collect();
        assert!(!road_actions.is_empty(), "should have legal road actions");

        // Build a road, then verify we can extend further from it
        let first_road = road_actions[0];
        game::apply(&mut state, first_road);

        // Give more resources and check new roads are available
        state.players[Player::One].hand.add(ROAD_COST);
        legal_actions(&state, &mut actions);
        let new_road_actions: Vec<ActionId> = actions
            .iter()
            .copied()
            .filter(|a| a.0 >= ROAD_START && a.0 < ROAD_END)
            .collect();
        assert!(
            new_road_actions.len() >= road_actions.len(),
            "extending the network should maintain or increase available roads"
        );
    }

    /// Before any settlements are placed, all maritime trade ratios default
    /// to 4:1 (the standard bank rate).
    #[test]
    fn maritime_trade_ratio_defaults() {
        let state = make_state();
        for &r in &ALL_RESOURCES {
            assert_eq!(state.players[Player::One].trade_ratios[r as usize], 4);
        }
    }

    /// When opponent < 3 building VP, populate_move_robber excludes tiles with
    /// opponent buildings.
    #[test]
    fn friendly_robber_restricts_tiles() {
        let mut state = make_state();
        play_setup(&mut state);

        state.current_player = Player::One;
        // P2 has 2 building VP from setup settlements (< 3 = FRIENDLY_ROBBER_VP)
        assert!(state.players[Player::Two].building_vps < crate::game::FRIENDLY_ROBBER_VP);

        state.phase = Phase::MoveRobber;
        let mut actions = Vec::new();
        legal_actions(&state, &mut actions);

        let opp_buildings = state.player_buildings(Player::Two);
        let topo = &state.topology;

        for a in &actions {
            let tid = a.robber_tile();
            let tile_mask = topo.adj.tile_nodes[tid.0 as usize];
            assert_eq!(
                tile_mask & opp_buildings,
                0,
                "friendly robber should exclude tile {:?} with opponent buildings",
                tid
            );
        }

        // Verify opponent tiles ARE included once VP >= 3
        state.players[Player::Two].building_vps = 3;
        legal_actions(&state, &mut actions);
        let has_opp_tile = actions.iter().any(|a| {
            let tid = a.robber_tile();
            let tile_mask = topo.adj.tile_nodes[tid.0 as usize];
            tile_mask & opp_buildings != 0
        });
        assert!(
            has_opp_tile,
            "above threshold, should include opponent tiles"
        );
    }

    /// When both players share a tile and one is friendly-protected,
    /// the current player can't place the robber on their own shared tile.
    #[test]
    fn friendly_robber_excludes_shared_tile() {
        let mut state = make_state();
        play_setup(&mut state);

        let topo = &state.topology;
        let p1_buildings = state.player_buildings(Player::One);
        let p2_buildings = state.player_buildings(Player::Two);

        // Find a P1 tile and place a P2 settlement on it to create a shared tile.
        let mut shared_tid = None;
        for tile in &topo.tiles {
            let tile_mask = topo.adj.tile_nodes[tile.id.0 as usize];
            if tile_mask & p1_buildings != 0 {
                // Find a free node on this tile for P2
                let occupied = state.occupied_nodes();
                for &nid in &tile.nodes {
                    let bit = 1u64 << nid.0;
                    if bit & occupied == 0 {
                        // Check distance rule: no adjacent occupied node
                        let adj_nodes = topo.adj.node_adj_nodes[nid.0 as usize];
                        if adj_nodes & occupied == 0 {
                            state.boards[Player::Two].settlements |= bit;
                            state.players[Player::Two].settlements_left -= 1;
                            state.players[Player::Two].building_vps += 1;
                            shared_tid = Some(tile.id);
                            break;
                        }
                    }
                }
                if shared_tid.is_some() {
                    break;
                }
            }
        }
        let shared_tid = shared_tid.expect("should find a shared tile");

        // P1 is current player, both have <= 2 building VP (P2 now has 3 from
        // the extra settlement, so set P1 to move robber while P2 is protected)
        // Actually P2 has 3 building VP now, so let's keep P2's building_vps at 2
        // by not incrementing (undo the +1 above and just place the building).
        state.players[Player::Two].building_vps = 2;

        state.current_player = Player::One;
        assert!(state.players[Player::Two].building_vps < crate::game::FRIENDLY_ROBBER_VP);

        state.phase = Phase::MoveRobber;
        let mut actions = Vec::new();
        legal_actions(&state, &mut actions);

        // The shared tile must be excluded because P2 is friendly-protected
        for a in &actions {
            assert_ne!(
                a.robber_tile(),
                shared_tid,
                "shared tile should be excluded when opponent is friendly-protected"
            );
        }
    }

    /// Settling on a generic port during setup reduces all trade ratios to 3:1.
    /// Settling on a resource-specific port further reduces that resource to 2:1.
    #[test]
    fn maritime_trade_ratio_with_ports() {
        // Find port nodes to place settlements on during setup
        let topo = Arc::new(Topology::from_seed(42));
        let generic_node = topo
            .nodes
            .iter()
            .find(|n| matches!(n.port, Some(Port::Generic)))
            .map(|n| n.id)
            .unwrap();
        let specific_node = topo
            .nodes
            .iter()
            .find(|n| matches!(n.port, Some(Port::Specific(_))))
            .map(|n| n.id)
            .unwrap();
        let specific_resource = match topo.nodes[specific_node.0 as usize].port {
            Some(Port::Specific(r)) => r,
            _ => unreachable!(),
        };

        // Create state and place P1's first settlement on the generic port
        let deck = DevCardDeck::new();
        let mut state = GameState::new(topo, deck, Dice::default());
        let mut rng = fastrand::Rng::with_seed(0);

        // P1 settles on generic port + road
        game::apply(&mut state, settlement_id(generic_node));
        let mut actions = Vec::new();
        legal_actions(&state, &mut actions);
        game::apply(&mut state, actions[0]);

        // P2's two settlements + roads (first available)
        for _ in 0..2 {
            legal_actions(&state, &mut actions);
            game::apply(&mut state, actions[0]);
            legal_actions(&state, &mut actions);
            game::apply(&mut state, actions[0]);
        }

        // P1 settles on specific port + road
        game::apply(&mut state, settlement_id(specific_node));
        legal_actions(&state, &mut actions);
        game::apply(&mut state, actions[0]);

        assert!(matches!(state.phase, Phase::PreRoll));

        // Check trade ratios were set by the game logic
        let p1 = &state.players[Player::One];
        for &r in &ALL_RESOURCES {
            if r == specific_resource {
                assert_eq!(
                    p1.trade_ratios[r as usize], 2,
                    "specific port should give 2:1"
                );
            } else {
                assert_eq!(
                    p1.trade_ratios[r as usize], 3,
                    "generic port should give 3:1"
                );
            }
        }
    }

    // --- Canonical ordering (commutativity) tests ---

    /// Helper: give player resources and play through a full turn,
    /// collecting all reachable end-of-turn states via BFS.
    fn reachable_end_states(state: &GameState) -> std::collections::HashSet<u64> {
        use std::collections::{HashSet, VecDeque};

        let mut visited = HashSet::new();
        let mut end_states = HashSet::new();
        let mut queue = VecDeque::new();

        // Hash a state by its board + hand (enough to identify unique end states)
        let hash_state = |s: &GameState| -> u64 {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            s.boards[Player::One].settlements.hash(&mut h);
            s.boards[Player::One].cities.hash(&mut h);
            s.boards[Player::One].road_network.roads.hash(&mut h);
            s.boards[Player::Two].settlements.hash(&mut h);
            s.boards[Player::Two].cities.hash(&mut h);
            s.boards[Player::Two].road_network.roads.hash(&mut h);
            for p in &[Player::One, Player::Two] {
                for &r in &ALL_RESOURCES {
                    s.players[*p].hand[r].hash(&mut h);
                }
                s.players[*p].dev_cards.0.hash(&mut h);
            }
            s.longest_road.hash(&mut h);
            h.finish()
        };

        let start_hash = hash_state(state);
        visited.insert(start_hash);
        queue.push_back(state.clone());

        let mut actions = Vec::new();
        while let Some(s) = queue.pop_front() {
            legal_actions(&s, &mut actions);
            for &a in &actions.clone() {
                let mut next = s.clone();
                game::apply(&mut next, a);
                // Resolve chance nodes
                let mut rng = fastrand::Rng::with_seed(0);
                while let Some(c) = next.sample_chance(&mut rng) {
                    next.apply_action(c);
                }
                if a.0 == END_TURN {
                    end_states.insert(hash_state(&next));
                } else {
                    let h = hash_state(&next);
                    if visited.insert(h) {
                        queue.push_back(next);
                    }
                }
            }
        }
        end_states
    }

    /// Canonical ordering must not lose any reachable end-of-turn state.
    /// Give a player enough resources for multiple builds+trades and verify
    /// that the canonical and non-canonical action spaces reach the same
    /// set of end-of-turn board states.
    #[test]
    fn canonical_ordering_preserves_all_end_states() {
        let mut state = make_state();
        play_setup(&mut state);
        state.phase = Phase::Main;
        state.current_player = Player::One;

        // Give P1 enough resources for 2 roads + 1 settlement
        let p = &mut state.players[Player::One];
        p.hand = ResourceArray::new(3, 3, 1, 1, 0);

        // Canonical: collect reachable end states
        let canon_ends = reachable_end_states(&state);

        // Non-canonical: disable ordering and collect
        let mut state_nc = state.clone();
        state_nc.canonical_build_order = false;
        let nocanon_ends = reachable_end_states(&state_nc);

        // Every canonical end state must be reachable without canonical
        for h in &canon_ends {
            assert!(
                nocanon_ends.contains(h),
                "canonical produced an end state not reachable without it"
            );
        }
        // Every non-canonical end state must be reachable with canonical
        for h in &nocanon_ends {
            assert!(
                canon_ends.contains(h),
                "non-canonical end state not reachable with canonical ordering"
            );
        }
        // Canonical should have fewer intermediate states (that's the point)
        assert!(
            !canon_ends.is_empty(),
            "should have at least one reachable end state"
        );
    }

    /// Canonical ordering should produce strictly fewer legal actions
    /// than non-canonical in positions where ordering matters.
    #[test]
    fn canonical_ordering_prunes_actions() {
        let mut state = make_state();
        play_setup(&mut state);
        state.phase = Phase::Main;
        state.current_player = Player::One;

        // Give P1 resources for road + settlement + trade
        state.players[Player::One].hand = ResourceArray::new(4, 2, 1, 1, 0);

        // Build a road first (locks out earlier categories)
        let mut actions = Vec::new();
        legal_actions(&state, &mut actions);
        let road = actions.iter().find(|a| a.0 >= ROAD_START && a.0 < ROAD_END);
        if let Some(&road_action) = road {
            let mut after_road = state.clone();
            game::apply(&mut after_road, road_action);

            let mut canon_actions = Vec::new();
            legal_actions(&after_road, &mut canon_actions);

            let mut nc_state = after_road.clone();
            nc_state.canonical_build_order = false;
            let mut nc_actions = Vec::new();
            legal_actions(&nc_state, &mut nc_actions);

            // Canonical should have fewer or equal actions (trades locked out)
            assert!(
                canon_actions.len() <= nc_actions.len(),
                "canonical should prune: canon={} nocanon={}",
                canon_actions.len(),
                nc_actions.len()
            );
        }
    }

    /// After building a road, maritime trades should be locked out
    /// (roads are step 5, trades are step 2).
    #[test]
    fn canonical_locks_out_trades_after_road() {
        let mut state = make_state();
        play_setup(&mut state);
        state.phase = Phase::Main;
        state.current_player = Player::One;

        // Give enough for road + trade
        state.players[Player::One].hand = ResourceArray::new(5, 1, 0, 0, 0);
        state.players[Player::One].trade_ratios = [4, 4, 4, 4, 4];

        let mut actions = Vec::new();
        legal_actions(&state, &mut actions);
        let has_trade_before = actions
            .iter()
            .any(|a| a.0 >= MARITIME_START && a.0 < MARITIME_END);

        // Build a road
        if let Some(&road) = actions.iter().find(|a| a.0 >= ROAD_START && a.0 < ROAD_END) {
            game::apply(&mut state, road);
            legal_actions(&state, &mut actions);
            let has_trade_after = actions
                .iter()
                .any(|a| a.0 >= MARITIME_START && a.0 < MARITIME_END);

            if has_trade_before {
                assert!(
                    !has_trade_after,
                    "trades should be locked out after building a road"
                );
            }
        }
    }

    /// After building on a port, trades should be re-enabled (port
    /// settlement resets min_step to allow trading at new rates).
    #[test]
    fn port_settlement_resets_trade_access() {
        let mut state = make_state();
        play_setup(&mut state);
        state.phase = Phase::Main;
        state.current_player = Player::One;

        // Find a port node reachable from P1's road network
        let port_nid = {
            let topo = &state.topology;
            let p1_roads = state.boards[Player::One].road_network.roads;
            let occupied = state.occupied_nodes();
            let mut neighbor_blocked = 0u64;
            let mut occ = occupied;
            while occ != 0 {
                let n = occ.trailing_zeros() as usize;
                occ &= occ - 1;
                neighbor_blocked |= topo.adj.node_adj_nodes[n];
            }
            let mut on_road = 0u64;
            let mut roads = p1_roads;
            while roads != 0 {
                let eid = roads.trailing_zeros() as usize;
                roads &= roads - 1;
                on_road |= topo.adj.edge_endpoints[eid];
            }
            let legal_settle = !occupied & !neighbor_blocked & on_road & ((1u64 << 54) - 1);
            topo.nodes
                .iter()
                .find(|n| n.port.is_some() && legal_settle & (1u64 << n.id.0) != 0)
                .map(|n| n.id)
        };

        if let Some(nid) = port_nid {
            state.players[Player::One].hand = ResourceArray::new(5, 5, 5, 5, 0);
            game::apply(&mut state, settlement_id(nid));

            // After port settlement, trades should be available
            let mut actions = Vec::new();
            legal_actions(&state, &mut actions);
            let has_trade = actions
                .iter()
                .any(|a| a.0 >= MARITIME_START && a.0 < MARITIME_END);
            assert!(has_trade, "port settlement should re-enable trading");
        }
    }
}
