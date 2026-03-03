pub mod action;
pub mod board;
pub mod dev_card;
pub mod dice;
pub mod hex;
pub mod resource;
pub mod road;
pub mod state;
pub mod topology;

use canopy2::game::{Game, Status};
use canopy2::player::Player;

use action::{
    ActionId, BUY_DEV_CARD, CITY_END, CITY_START, DISCARD_END, DISCARD_START, END_TURN,
    MARITIME_END, MARITIME_START, MONOPOLY_END, MONOPOLY_START, PLAY_KNIGHT, PLAY_ROAD_BUILDING,
    ROAD_END, ROAD_START, ROBBER_END, ROBBER_START, SETTLEMENT_END, SETTLEMENT_START, STEAL,
    YOP_END, YOP_START,
};
use board::{EdgeId, NodeId, TileId};
use dev_card::DevCardKind;
use dice::Dice;
use resource::{
    ALL_RESOURCES, CITY_COST, DEV_CARD_COST, ROAD_COST, Resource, ResourceArray, SETTLEMENT_COST,
};
use state::{GameState, Phase};

const VP_TO_WIN: u8 = 15;
const MAX_TURNS: u16 = 2000;
pub(crate) const DISCARD_THRESHOLD: u8 = 9;
pub(crate) const FRIENDLY_ROBBER_VP: u8 = 3;

pub fn new_game(seed: u64, dice: Dice) -> GameState {
    GameState::from_seed(seed, dice)
}

const DICE_PROBS: [(usize, f32); 11] = [
    (0, 1.0 / 36.0),
    (1, 2.0 / 36.0),
    (2, 3.0 / 36.0),
    (3, 4.0 / 36.0),
    (4, 5.0 / 36.0),
    (5, 6.0 / 36.0),
    (6, 5.0 / 36.0),
    (7, 4.0 / 36.0),
    (8, 3.0 / 36.0),
    (9, 2.0 / 36.0),
    (10, 1.0 / 36.0),
];

impl Game for GameState {
    const NUM_ACTIONS: usize = action::ACTION_SPACE;

    fn status(&self) -> Status {
        match &self.phase {
            Phase::GameOver(p) => Status::Terminal(p.sign()),
            _ => Status::Ongoing(self.current_player),
        }
    }

    fn legal_actions(&self, buf: &mut Vec<usize>) {
        buf.clear();
        let mut abuf = Vec::new();
        action::legal_actions(self, &mut abuf);
        buf.extend(abuf.iter().map(|a| a.0 as usize));
    }

    fn apply_action(&mut self, action: usize) {
        match self.phase {
            Phase::Roll => {
                let total = (action + 2) as u8;
                if let Dice::Balanced(ref mut b) = self.dice {
                    b.draw(total, self.current_player);
                }
                if total == 7 {
                    handle_seven(self);
                } else {
                    distribute_resources(self, total);
                    self.phase = Phase::Main;
                }
            }
            Phase::StealResolve => {
                let resource = ALL_RESOURCES[action];
                let current = self.current_player;
                let target = current.opponent();
                self.players[target].hand[resource] -= 1;
                self.players[current].hand[resource] += 1;
                self.phase = Phase::Main;
            }
            _ => {
                apply(self, ActionId(action as u8));
            }
        }
    }

    fn chance_outcomes(&self, buf: &mut Vec<(usize, f32)>) {
        match self.phase {
            Phase::Roll => match &self.dice {
                Dice::Random => buf.extend_from_slice(&DICE_PROBS),
                Dice::Balanced(b) => {
                    for &(i, p) in &b.probabilities(self.current_player) {
                        buf.push((i as usize, p));
                    }
                }
            },
            Phase::StealResolve => {
                let target = self.current_player.opponent();
                let target_hand = &self.players[target].hand;
                let total = target_hand.total() as f32;
                if total > 0.0 {
                    for (i, &r) in ALL_RESOURCES.iter().enumerate() {
                        let count = target_hand[r];
                        if count > 0 {
                            buf.push((i, count as f32 / total));
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

#[allow(non_contiguous_range_endpoints)]
pub fn apply(state: &mut GameState, action: ActionId) {
    match action.0 {
        SETTLEMENT_START..SETTLEMENT_END => {
            let nid = action.settlement_node();
            apply_settlement(state, nid);
        }
        ROAD_START..ROAD_END => {
            let eid = action.road_edge();
            apply_road(state, eid);
        }
        CITY_START..CITY_END => {
            let nid = action.city_node();
            apply_build_city(state, nid);
        }
        END_TURN => apply_end_turn(state),
        BUY_DEV_CARD => apply_buy_dev_card(state),
        PLAY_KNIGHT => apply_play_knight(state),
        PLAY_ROAD_BUILDING => apply_play_road_building(state),
        YOP_START..YOP_END => {
            let (r1, r2) = action.year_of_plenty_resources();
            apply_year_of_plenty(state, r1, r2);
        }
        MONOPOLY_START..MONOPOLY_END => {
            let r = action.monopoly_resource();
            apply_monopoly(state, r);
        }
        ROBBER_START..ROBBER_END => {
            let tid = action.robber_tile();
            apply_move_robber(state, tid);
        }
        STEAL => apply_steal(state),
        DISCARD_START..DISCARD_END => {
            let r = action.discard_resource();
            apply_discard_resource(state, r);
        }
        MARITIME_START..MARITIME_END => {
            let (give, recv) = action.maritime_trade();
            apply_maritime_trade(state, give, recv);
        }
        _ => unreachable!("Invalid action id: {}", action.0),
    }
}

fn apply_settlement(state: &mut GameState, nid: NodeId) {
    if matches!(state.phase, Phase::PlaceSettlement) {
        apply_place_settlement(state, nid);
    } else {
        apply_build_settlement(state, nid);
    }
}

fn apply_road(state: &mut GameState, eid: EdgeId) {
    if matches!(state.phase, Phase::PlaceRoad) {
        apply_place_road(state, eid);
    } else {
        apply_build_road(state, eid);
    }
}

fn update_trade_ratios(state: &mut GameState, nid: NodeId, pid: Player) {
    let bit = 1u64 << nid.0;
    let adj = &state.topology.adj;
    for (ri, &mask) in adj.port_specific.iter().enumerate() {
        if bit & mask != 0 {
            state.players[pid].trade_ratios[ri] = state.players[pid].trade_ratios[ri].min(2);
        }
    }
    if bit & adj.port_generic != 0 {
        for r in &mut state.players[pid].trade_ratios {
            *r = (*r).min(3);
        }
    }
}

fn apply_place_settlement(state: &mut GameState, nid: NodeId) {
    let pid = state.current_player;
    let opp = pid.opponent();
    state.boards[pid].settlements |= 1u64 << nid.0;
    state.current_mut().settlements_left -= 1;
    state.current_mut().victory_points += 1;
    state.setup_count += 1;

    update_trade_ratios(state, nid, pid);

    // Update road network frontier
    let opp_roads = state.boards[opp].road_network.roads;
    state.boards[pid]
        .road_network
        .add_building(nid, &state.topology.adj, opp_roads);

    // Give resources for second settlement (settlements 2 and 3 are the second placements)
    let setup_count = state.setup_count as usize;
    if setup_count == 3 || setup_count == 4 {
        let mut to_give = ResourceArray::default();
        let topo = &state.topology;
        let node = &topo.nodes[nid.0 as usize];
        for &tid in &node.adjacent_tiles {
            let tile = &topo.tiles[tid.0 as usize];
            if let Some(resource) = tile.terrain.resource()
                && state.bank[resource] > to_give[resource]
            {
                to_give[resource] += 1;
            }
        }
        state.players[pid].hand.add(to_give);
        state.bank.sub(to_give);
    }
    state.phase = Phase::PlaceRoad;
}

fn apply_place_road(state: &mut GameState, eid: EdgeId) {
    let pid = state.current_player;
    let opp = pid.opponent();
    let opp_buildings = state.player_buildings(opp);
    let opp_roads = state.boards[opp].road_network.roads;
    state.boards[pid].road_network.add_road(
        eid,
        &state.topology.adj,
        opp_buildings,
        opp_roads,
        false,
    );
    state.boards[opp].road_network.remove_edge(eid);
    state.current_mut().roads_placed += 1;
    state.current_mut().roads_left -= 1;

    let setup_count = state.setup_count as usize;
    match setup_count {
        1 => {
            state.current_player = Player::Two;
            state.phase = Phase::PlaceSettlement;
        }
        2 => {
            state.phase = Phase::PlaceSettlement;
        }
        3 => {
            state.current_player = Player::One;
            state.phase = Phase::PlaceSettlement;
        }
        4 => {
            state.current_player = Player::One;
            state.phase = Phase::Roll;
            state.turn_number = 1;
        }
        _ => unreachable!("setup_count should be 1-4"),
    }
}

fn handle_seven(state: &mut GameState) {
    let current = state.current_player;
    let opponent = current.opponent();

    let current_total = state.players[current].hand.total();
    if current_total > DISCARD_THRESHOLD {
        state.phase = Phase::Discard {
            player: current,
            remaining: current_total / 2,
        };
    } else {
        let opp_total = state.players[opponent].hand.total();
        if opp_total > DISCARD_THRESHOLD {
            state.phase = Phase::Discard {
                player: opponent,
                remaining: opp_total / 2,
            };
        } else {
            state.phase = Phase::MoveRobber;
        }
    }
}

fn distribute_resources(state: &mut GameState, roll: u8) {
    let topo = &state.topology;

    // Pass 1: accumulate total demand per resource type and per-player gains.
    let mut total_demand = [0u8; 5];
    let mut player_gains = [[0u8; 5]; 2];

    for &tid in &topo.dice_to_tiles[roll as usize] {
        if tid == state.robber {
            continue;
        }
        let tile = &topo.tiles[tid.0 as usize];
        let resource = match tile.terrain.resource() {
            Some(r) => r,
            None => continue,
        };
        let ri = resource as usize;
        let tile_mask = topo.adj.tile_nodes[tid.0 as usize];

        for (pi, &pid) in [Player::One, Player::Two].iter().enumerate() {
            let s = (state.boards[pid].settlements & tile_mask).count_ones() as u8;
            let c = (state.boards[pid].cities & tile_mask).count_ones() as u8;
            let amount = s + c * 2;
            total_demand[ri] += amount;
            player_gains[pi][ri] += amount;
        }
    }

    // Pass 2: for each resource type, distribute only if bank can cover total demand.
    for &r in &ALL_RESOURCES {
        let ri = r as usize;
        if total_demand[ri] == 0 || state.bank[r] < total_demand[ri] {
            continue;
        }
        for (pi, &pid) in [Player::One, Player::Two].iter().enumerate() {
            let amount = player_gains[pi][ri];
            if amount > 0 {
                state.players[pid].hand[r] += amount;
                state.bank[r] -= amount;
            }
        }
    }
}

fn apply_end_turn(state: &mut GameState) {
    if state.turn_number >= MAX_TURNS {
        let vp = |p: Player| {
            state.players[p].victory_points + state.players[p].dev_cards[DevCardKind::VictoryPoint]
        };
        let winner = if vp(Player::Two) > vp(Player::One) {
            Player::Two
        } else {
            Player::One
        };
        state.phase = Phase::GameOver(winner);
        return;
    }

    let player = &mut state.players[state.current_player];
    for i in 0..5 {
        player.dev_cards_bought_this_turn.0[i] = 0;
    }
    player.has_played_dev_card_this_turn = false;

    state.current_player = state.current_player.opponent();
    state.turn_number += 1;
    state.phase = Phase::Roll;
}

fn apply_build_road(state: &mut GameState, eid: EdgeId) {
    let in_road_building = matches!(state.phase, Phase::RoadBuilding { .. });
    if !in_road_building {
        state.current_mut().hand.sub(ROAD_COST);
        state.bank.add(ROAD_COST);
    }

    let pid = state.current_player;
    let opp = pid.opponent();
    let opp_buildings = state.player_buildings(opp);
    let opp_roads = state.boards[opp].road_network.roads;
    state.boards[pid].road_network.add_road(
        eid,
        &state.topology.adj,
        opp_buildings,
        opp_roads,
        true,
    );
    state.boards[opp].road_network.remove_edge(eid);
    state.current_mut().roads_placed += 1;
    state.current_mut().roads_left -= 1;

    if let Phase::RoadBuilding { roads_left } = state.phase {
        let remaining = roads_left - 1;
        if remaining == 0 || state.current().roads_left == 0 {
            state.phase = Phase::Main;
        } else {
            state.phase = Phase::RoadBuilding {
                roads_left: remaining,
            };
        }
    }

    update_longest_road(state);
    check_victory(state);
}

fn apply_build_settlement(state: &mut GameState, nid: NodeId) {
    let pid = state.current_player;
    let opp = pid.opponent();
    state.current_mut().hand.sub(SETTLEMENT_COST);
    state.bank.add(SETTLEMENT_COST);
    state.boards[pid].settlements |= 1u64 << nid.0;
    state.current_mut().settlements_left -= 1;
    state.current_mut().victory_points += 1;

    update_trade_ratios(state, nid, pid);

    // Update road networks
    let opp_roads = state.boards[opp].road_network.roads;
    state.boards[pid]
        .road_network
        .add_building(nid, &state.topology.adj, opp_roads);

    let opp_own_buildings = state.boards[opp].settlements | state.boards[opp].cities;
    let pid_buildings = state.boards[pid].settlements | state.boards[pid].cities;
    state.boards[opp].road_network.on_opponent_build(
        nid,
        &state.topology.adj,
        opp_own_buildings,
        pid_buildings,
    );

    update_longest_road(state);
    check_victory(state);
}

fn apply_build_city(state: &mut GameState, nid: NodeId) {
    let pid = state.current_player;
    state.current_mut().hand.sub(CITY_COST);
    state.bank.add(CITY_COST);
    let bit = 1u64 << nid.0;
    state.boards[pid].settlements &= !bit;
    state.boards[pid].cities |= bit;
    state.current_mut().settlements_left += 1;
    state.current_mut().cities_left -= 1;
    state.current_mut().victory_points += 1;

    check_victory(state);
}

fn apply_buy_dev_card(state: &mut GameState) {
    state.current_mut().hand.sub(DEV_CARD_COST);
    state.bank.add(DEV_CARD_COST);
    let card = state.dev_deck.draw().unwrap();
    state.current_mut().dev_cards[card] += 1;
    state.current_mut().dev_cards_bought_this_turn[card] += 1;

    if card == DevCardKind::VictoryPoint {
        check_victory(state);
    }
}

fn apply_play_knight(state: &mut GameState) {
    state.current_mut().dev_cards[DevCardKind::Knight] -= 1;
    state.current_mut().has_played_dev_card_this_turn = true;
    state.current_mut().knights_played += 1;

    update_largest_army(state);
    state.phase = Phase::MoveRobber;
}

fn apply_play_road_building(state: &mut GameState) {
    state.current_mut().dev_cards[DevCardKind::RoadBuilding] -= 1;
    state.current_mut().has_played_dev_card_this_turn = true;
    state.phase = Phase::RoadBuilding { roads_left: 2 };
}

fn apply_year_of_plenty(state: &mut GameState, r1: Resource, r2: Resource) {
    state.current_mut().dev_cards[DevCardKind::YearOfPlenty] -= 1;
    state.current_mut().has_played_dev_card_this_turn = true;

    if state.bank[r1] > 0 {
        state.bank[r1] -= 1;
        state.current_mut().hand[r1] += 1;
    }
    if state.bank[r2] > 0 {
        state.bank[r2] -= 1;
        state.current_mut().hand[r2] += 1;
    }
}

fn apply_monopoly(state: &mut GameState, resource: Resource) {
    state.current_mut().dev_cards[DevCardKind::Monopoly] -= 1;
    state.current_mut().has_played_dev_card_this_turn = true;

    let current = state.current_player;
    let opponent = current.opponent();
    let stolen = state.players[opponent].hand[resource];
    state.players[opponent].hand[resource] = 0;
    state.players[current].hand[resource] += stolen;
}

fn apply_move_robber(state: &mut GameState, tid: TileId) {
    state.robber = tid;

    let opp = state.current_player.opponent();
    let tile_mask = state.topology.adj.tile_nodes[tid.0 as usize];
    let opp_buildings = state.player_buildings(opp);
    let has_target = (tile_mask & opp_buildings) != 0 && state.players[opp].hand.total() > 0;

    if has_target && state.public_vps(opp) >= FRIENDLY_ROBBER_VP {
        state.phase = Phase::Steal;
    } else {
        state.phase = Phase::Main;
    }
}

fn apply_steal(state: &mut GameState) {
    state.phase = Phase::StealResolve;
}

fn apply_discard_resource(state: &mut GameState, resource: Resource) {
    if let Phase::Discard { player, remaining } = state.phase {
        state.players[player].hand[resource] -= 1;
        state.bank[resource] += 1;

        let new_remaining = remaining - 1;
        if new_remaining > 0 {
            state.phase = Phase::Discard {
                player,
                remaining: new_remaining,
            };
        } else {
            // Check if the opponent also needs to discard
            let opponent = player.opponent();
            let opp_total = state.players[opponent].hand.total();
            if opp_total > DISCARD_THRESHOLD {
                state.phase = Phase::Discard {
                    player: opponent,
                    remaining: opp_total / 2,
                };
            } else {
                state.phase = Phase::MoveRobber;
            }
        }
    }
}

fn apply_maritime_trade(state: &mut GameState, give: Resource, receive: Resource) {
    let ratio = state.players[state.current_player].trade_ratios[give as usize];
    let give_array = {
        let mut a = ResourceArray::default();
        a[give] = ratio;
        a
    };
    state.current_mut().hand.sub(give_array);
    state.bank.add(give_array);
    state.current_mut().hand[receive] += 1;
    state.bank[receive] -= 1;
}

fn check_victory(state: &mut GameState) {
    if matches!(state.phase, Phase::GameOver(_)) {
        return;
    }
    let pid = state.current_player;
    let player = &state.players[pid];
    let mut vps = player.victory_points + player.dev_cards[DevCardKind::VictoryPoint];

    if let Some((lr_pid, _)) = state.longest_road
        && lr_pid == pid
    {
        vps += 2;
    }
    if let Some((la_pid, _)) = state.largest_army
        && la_pid == pid
    {
        vps += 2;
    }

    if vps >= VP_TO_WIN {
        state.phase = Phase::GameOver(pid);
    }
}

fn update_largest_army(state: &mut GameState) {
    let k1 = state.players[Player::One].knights_played;
    let k2 = state.players[Player::Two].knights_played;
    let new = match (k1 >= 3, k2 >= 3) {
        (false, false) => None,
        (true, false) => Some((Player::One, k1)),
        (false, true) => Some((Player::Two, k2)),
        (true, true) => {
            if k1 > k2 {
                Some((Player::One, k1))
            } else if k2 > k1 {
                Some((Player::Two, k2))
            } else {
                match state.largest_army {
                    Some((pid, _)) => Some((pid, k1)),
                    None => None,
                }
            }
        }
    };
    state.largest_army = new;
}

fn update_longest_road(state: &mut GameState) {
    let len1 = state.boards[Player::One].road_network.longest_road();
    let len2 = state.boards[Player::Two].road_network.longest_road();

    // Determine who (if anyone) holds longest road.
    // Must be >= 5 and strictly longer than the opponent to take it.
    // Current holder keeps on a tie.
    let new = match (len1 >= 5, len2 >= 5) {
        (false, false) => None,
        (true, false) => Some((Player::One, len1)),
        (false, true) => Some((Player::Two, len2)),
        (true, true) => {
            if len1 > len2 {
                Some((Player::One, len1))
            } else if len2 > len1 {
                Some((Player::Two, len2))
            } else {
                // Tie: current holder keeps it, or nobody gets it
                match state.longest_road {
                    Some((pid, _)) => Some((pid, len1)),
                    None => None,
                }
            }
        }
    };
    state.longest_road = new;
}

#[cfg(test)]
mod tests {
    use super::action::{
        self, BUY_DEV_CARD, END_TURN, PLAY_KNIGHT, ROAD_END, ROAD_START, SETTLEMENT_END,
        SETTLEMENT_START, maritime_id, road_id, robber_id, settlement_id,
    };
    use super::board::AdjacencyBitboards;
    use super::dev_card::{DevCardDeck, DevCardKind};
    use super::dice::Dice;
    use super::resource::{DEV_CARD_COST, Resource, SETTLEMENT_COST};
    use super::*;
    use canopy2::game::Game;
    use std::sync::Arc;

    fn sample_chance_outcome(buf: &[(usize, f32)], rng: &mut fastrand::Rng) -> usize {
        let mut roll = rng.f32();
        for &(outcome, prob) in buf {
            roll -= prob;
            if roll <= 0.0 {
                return outcome;
            }
        }
        buf.last().unwrap().0
    }

    fn make_state_with_seed(seed: u64) -> GameState {
        GameState::from_seed(seed, Dice::default())
    }

    /// Play through the 4-settlement setup phase using random actions.
    fn play_setup(state: &mut GameState) -> fastrand::Rng {
        let mut rng = fastrand::Rng::with_seed(0);
        let mut actions = Vec::new();
        for _ in 0..4 {
            action::legal_actions(state, &mut actions);
            let i = rng.usize(..actions.len());
            apply(state, actions[i]);
            action::legal_actions(state, &mut actions);
            let i = rng.usize(..actions.len());
            apply(state, actions[i]);
        }
        assert!(matches!(state.phase, Phase::Roll));
        rng
    }

    /// Fast-forward through non-Main phases (chance events + forced actions).
    fn fast_forward(state: &mut GameState, rng: &mut fastrand::Rng) {
        let mut actions = Vec::new();
        let mut chance_buf = Vec::new();
        while !matches!(state.phase, Phase::Main | Phase::GameOver(_)) {
            chance_buf.clear();
            state.chance_outcomes(&mut chance_buf);
            if !chance_buf.is_empty() {
                let outcome = sample_chance_outcome(&chance_buf, rng);
                state.apply_action(outcome);
                continue;
            }
            action::legal_actions(state, &mut actions);
            if actions.is_empty() {
                break;
            }
            apply(state, actions[0]);
        }
    }

    /// Find a linear chain of `length` edges starting from `start`, avoiding
    /// occupied nodes and already-placed roads.
    fn find_linear_path(
        adj: &AdjacencyBitboards,
        start: NodeId,
        length: usize,
        occupied: u64,
        used_roads: u128,
    ) -> Vec<EdgeId> {
        let mut path = Vec::new();
        let mut visited = occupied | (1u64 << start.0);
        let mut current = start;
        for _ in 0..length {
            let mut edges = adj.node_adj_edges[current.0 as usize] & !used_roads;
            let mut found = None;
            while edges != 0 {
                let eid = edges.trailing_zeros() as usize;
                edges &= edges - 1;
                let next = (adj.edge_endpoints[eid] & !(1u64 << current.0)).trailing_zeros() as u8;
                if visited & (1u64 << next) == 0 {
                    found = Some((EdgeId(eid as u8), next));
                    break;
                }
            }
            let (eid, next) = found.expect("should find next edge in linear path");
            path.push(eid);
            visited |= 1u64 << next;
            current = NodeId(next);
        }
        path
    }

    /// Building a contiguous chain of 5+ roads from a settlement triggers
    /// the Longest Road award (worth 2 VP).
    #[test]
    fn longest_road_awarded_after_building() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        // Find P1's first settlement and build a 5-edge linear path from it
        let start = NodeId(state.boards[Player::One].settlements.trailing_zeros() as u8);
        let path = find_linear_path(
            &state.topology.adj,
            start,
            5,
            state.occupied_nodes(),
            state.all_roads(),
        );

        for &eid in &path {
            state.phase = Phase::Main;
            state.current_player = Player::One;
            state.players[Player::One].hand.add(ROAD_COST);
            apply(&mut state, road_id(eid));
        }

        assert!(
            matches!(state.longest_road, Some((Player::One, len)) if len >= 5),
            "P1 should have longest road >= 5, got {:?}",
            state.longest_road
        );
    }

    /// An opponent settlement placed on an interior node of a road chain
    /// splits the chain, potentially revoking the Longest Road award if
    /// neither segment is long enough.
    #[test]
    fn settlement_cuts_longest_road() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        // Build a 7-edge linear road from P1's first settlement
        let start = NodeId(state.boards[Player::One].settlements.trailing_zeros() as u8);
        let path = find_linear_path(
            &state.topology.adj,
            start,
            7,
            state.occupied_nodes(),
            state.all_roads(),
        );

        for &eid in &path {
            state.phase = Phase::Main;
            state.current_player = Player::One;
            state.players[Player::One].hand.add(ROAD_COST);
            apply(&mut state, road_id(eid));
        }

        assert!(
            matches!(state.longest_road, Some((Player::One, _))),
            "P1 should hold longest road, got {:?}",
            state.longest_road
        );

        // Find a cuttable interior node: shared between consecutive path edges,
        // no existing building, and satisfies distance rule (no adjacent building).
        let adj = &state.topology.adj;
        let all_buildings = state.occupied_nodes();
        let mid = path.len() / 2;
        let mut cut_node = None;
        // Search from the middle outward for the best cut
        for offset in 0..path.len() {
            for &i in &[mid + offset, mid.wrapping_sub(offset + 1)] {
                if i == 0 || i >= path.len() {
                    continue;
                }
                let shared = adj.edge_endpoints[path[i - 1].0 as usize]
                    & adj.edge_endpoints[path[i].0 as usize];
                if shared == 0 {
                    continue;
                }
                let nid = shared.trailing_zeros() as u8;
                let bit = 1u64 << nid;
                if all_buildings & bit != 0 {
                    continue;
                }
                if adj.node_adj_nodes[nid as usize] & all_buildings != 0 {
                    continue;
                }
                cut_node = Some(NodeId(nid));
                break;
            }
            if cut_node.is_some() {
                break;
            }
        }
        let cut_node = cut_node.expect("should find a valid interior node to cut");

        // P2 builds a settlement at the cut node
        state.current_player = Player::Two;
        state.phase = Phase::Main;
        state.players[Player::Two].hand.add(SETTLEMENT_COST);
        apply(&mut state, settlement_id(cut_node));

        // P1's road is split — longest road should be revoked or transferred
        match state.longest_road {
            None => {}
            Some((Player::Two, _)) => {}
            Some((Player::One, len)) => {
                panic!("P1 should not keep longest road after cut, got length {len}");
            }
        }
    }

    /// After setup, rolling the dice distributes resources from tiles matching
    /// the roll to players with adjacent settlements. Over 50 rolls, at least
    /// some resources must be gained.
    #[test]
    fn distribute_resources_on_roll() {
        let mut state = make_state_with_seed(42);
        play_setup(&mut state);

        // Record P1 and P2 hands before rolling
        let hands_before = [
            state.players[Player::One].hand.total(),
            state.players[Player::Two].hand.total(),
        ];

        // Roll many times to ensure at least some resources are distributed
        let mut rng = fastrand::Rng::with_seed(123);
        let mut chance_buf = Vec::new();
        let mut total_gained = 0u16;
        for _ in 0..50 {
            state.phase = Phase::Roll;
            let before =
                state.players[Player::One].hand.total() + state.players[Player::Two].hand.total();
            // Use chance_outcomes + apply_action
            chance_buf.clear();
            state.chance_outcomes(&mut chance_buf);
            let outcome = sample_chance_outcome(&chance_buf, &mut rng);
            state.apply_action(outcome);
            let after =
                state.players[Player::One].hand.total() + state.players[Player::Two].hand.total();
            // Resources may be gained (non-7 roll) or lost (discard on 7)
            if after > before {
                total_gained += (after - before) as u16;
            }
            // Fast-forward past any discard/robber/steal/steal-resolve phases
            let mut inner_actions = Vec::new();
            while !matches!(state.phase, Phase::Main | Phase::GameOver(_)) {
                chance_buf.clear();
                state.chance_outcomes(&mut chance_buf);
                if !chance_buf.is_empty() {
                    let o = sample_chance_outcome(&chance_buf, &mut rng);
                    state.apply_action(o);
                    continue;
                }
                action::legal_actions(&state, &mut inner_actions);
                if inner_actions.is_empty() {
                    break;
                }
                apply(&mut state, inner_actions[0]);
            }
            if matches!(state.phase, Phase::GameOver(_)) {
                break;
            }
            // End turn to get back to Roll
            apply(&mut state, ActionId(END_TURN));
        }

        assert!(
            total_gained > 0,
            "after 50 rolls, some resources should have been distributed (started with {:?})",
            hands_before,
        );
    }

    /// Playing a full game with random actions must always terminate (reach
    /// GameOver) and never produce a state with zero legal actions.
    /// Runs 10 seeds for coverage.
    #[test]
    fn full_game_simulation() {
        for seed in 0..10u64 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut s = new_game(seed, Dice::default());
            let mut actions = Vec::new();
            let mut chance_buf = Vec::new();
            for _ in 0..10000 {
                if matches!(s.phase, Phase::GameOver(_)) {
                    break;
                }
                // Resolve chance events first
                loop {
                    chance_buf.clear();
                    s.chance_outcomes(&mut chance_buf);
                    if chance_buf.is_empty() {
                        break;
                    }
                    let outcome = sample_chance_outcome(&chance_buf, &mut rng);
                    s.apply_action(outcome);
                }
                if matches!(s.phase, Phase::GameOver(_)) {
                    break;
                }
                action::legal_actions(&s, &mut actions);
                assert!(!actions.is_empty(), "should always have legal actions");
                let action = actions[rng.usize(..actions.len())];
                apply(&mut s, action);
            }
            assert!(
                matches!(s.phase, Phase::GameOver(_)),
                "game with seed {} should complete",
                seed
            );
        }
    }

    /// Two games played with the same seed must produce the exact same
    /// action sequence, verifying that all game logic is deterministic.
    #[test]
    fn deterministic_replay() {
        let seed = 99u64;
        let mut results = Vec::new();
        for _ in 0..2 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut s = new_game(seed, Dice::default());
            let mut actions = Vec::new();
            let mut action_log = Vec::new();
            let mut chance_buf = Vec::new();
            for _ in 0..10000 {
                if matches!(s.phase, Phase::GameOver(_)) {
                    break;
                }
                // Resolve chance events first
                loop {
                    chance_buf.clear();
                    s.chance_outcomes(&mut chance_buf);
                    if chance_buf.is_empty() {
                        break;
                    }
                    let outcome = sample_chance_outcome(&chance_buf, &mut rng);
                    s.apply_action(outcome);
                }
                if matches!(s.phase, Phase::GameOver(_)) {
                    break;
                }
                action::legal_actions(&s, &mut actions);
                let action = actions[rng.usize(..actions.len())];
                action_log.push(action.0);
                apply(&mut s, action);
            }
            results.push(action_log);
        }
        assert_eq!(
            results[0], results[1],
            "same seed should produce same action sequence"
        );
    }

    // --- Longest Road tests ---

    /// Both players build 5+ roads to equal length. First achiever keeps LR on tie.
    #[test]
    fn longest_road_tie_holder_keeps() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        // P1 builds 5 roads
        let start1 = NodeId(state.boards[Player::One].settlements.trailing_zeros() as u8);
        let path1 = find_linear_path(
            &state.topology.adj,
            start1,
            5,
            state.occupied_nodes(),
            state.all_roads(),
        );
        for &eid in &path1 {
            state.phase = Phase::Main;
            state.current_player = Player::One;
            state.players[Player::One].hand.add(ROAD_COST);
            apply(&mut state, road_id(eid));
        }
        assert!(matches!(state.longest_road, Some((Player::One, _))));

        // P2 builds 5 roads to tie
        let start2 = NodeId(state.boards[Player::Two].settlements.trailing_zeros() as u8);
        let path2 = find_linear_path(
            &state.topology.adj,
            start2,
            5,
            state.occupied_nodes(),
            state.all_roads(),
        );
        for &eid in &path2 {
            state.phase = Phase::Main;
            state.current_player = Player::Two;
            state.players[Player::Two].hand.add(ROAD_COST);
            apply(&mut state, road_id(eid));
        }

        // P1 should keep LR on tie (first achiever holds)
        assert!(
            matches!(state.longest_road, Some((Player::One, _))),
            "first achiever should keep LR on tie, got {:?}",
            state.longest_road
        );
    }

    /// Build a road loop, verify longest road calculation handles cycles
    /// (counts longest simple path, not the cycle).
    #[test]
    fn longest_road_cycle() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);
        let topo = Arc::clone(&state.topology);

        // Find a settlement node of P1 and build roads forming a loop
        let start = NodeId(state.boards[Player::One].settlements.trailing_zeros() as u8);

        // Build a path of 5 edges from start
        let path = find_linear_path(
            &topo.adj,
            start,
            5,
            state.occupied_nodes(),
            state.all_roads(),
        );
        for &eid in &path {
            state.phase = Phase::Main;
            state.current_player = Player::One;
            state.players[Player::One].hand.add(ROAD_COST);
            apply(&mut state, road_id(eid));
        }

        // Try to close the loop: find an edge connecting end of path back toward start
        let end_node = {
            let last_edge = path.last().unwrap();
            let endpoints = topo.adj.edge_endpoints[last_edge.0 as usize];
            let prev_edge = path[path.len() - 2];
            let shared = endpoints & topo.adj.edge_endpoints[prev_edge.0 as usize];
            let end = endpoints & !shared;
            NodeId(end.trailing_zeros() as u8)
        };

        // Check if there's a closing edge back to start's neighbor
        let start_adj_edges = topo.adj.node_adj_edges[start.0 as usize];
        let end_adj_edges = topo.adj.node_adj_edges[end_node.0 as usize];
        let my_roads = state.boards[Player::One].road_network.roads;
        let closing_candidates = start_adj_edges & end_adj_edges & !my_roads;

        if closing_candidates != 0 {
            let close_eid = EdgeId(closing_candidates.trailing_zeros() as u8);
            state.phase = Phase::Main;
            state.current_player = Player::One;
            state.players[Player::One].hand.add(ROAD_COST);
            apply(&mut state, road_id(close_eid));
        }

        // Regardless of whether we closed the loop, LR should reflect the
        // longest simple path (5 or 6), not infinity
        if let Some((Player::One, len)) = state.longest_road {
            assert!(len <= 10, "cycle shouldn't produce unreasonable LR length");
        }
    }

    /// Build a Y-shaped road, verify only the longest branch path is counted.
    #[test]
    fn longest_road_branching() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);
        let topo = Arc::clone(&state.topology);

        let start = NodeId(state.boards[Player::One].settlements.trailing_zeros() as u8);

        // Build a trunk of 3 roads
        let trunk = find_linear_path(
            &topo.adj,
            start,
            3,
            state.occupied_nodes(),
            state.all_roads(),
        );
        for &eid in &trunk {
            state.phase = Phase::Main;
            state.current_player = Player::One;
            state.players[Player::One].hand.add(ROAD_COST);
            apply(&mut state, road_id(eid));
        }

        // Find the fork node (end of trunk)
        let fork_node = {
            let last_edge = trunk.last().unwrap();
            let endpoints = topo.adj.edge_endpoints[last_edge.0 as usize];
            let prev_edge = trunk[trunk.len() - 2];
            let shared = endpoints & topo.adj.edge_endpoints[prev_edge.0 as usize];
            NodeId((endpoints & !shared).trailing_zeros() as u8)
        };

        // Build two branches from the fork
        let branch1 = find_linear_path(
            &topo.adj,
            fork_node,
            2,
            state.occupied_nodes(),
            state.all_roads(),
        );
        for &eid in &branch1 {
            state.phase = Phase::Main;
            state.current_player = Player::One;
            state.players[Player::One].hand.add(ROAD_COST);
            apply(&mut state, road_id(eid));
        }

        // The longest path from start through trunk + branch = 3 + 2 = 5
        // but with the setup road it should be at least 5 total
        if let Some((Player::One, len)) = state.longest_road {
            // The trunk is 3 + longest branch is 2 = 5; the other branch
            // doesn't add to the longest simple path
            assert!(
                len >= 5,
                "Y-shape should have longest path >= 5, got {}",
                len
            );
        }
    }

    // --- Largest Army tests ---

    /// Playing 3rd knight triggers LA award.
    #[test]
    fn largest_army_awarded_at_three_knights() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        state.current_player = Player::One;
        state.players[Player::One].dev_cards[DevCardKind::Knight] = 3;

        for _ in 0..3 {
            state.phase = Phase::Main;
            state.current_player = Player::One;
            state.players[Player::One].has_played_dev_card_this_turn = false;
            apply(&mut state, ActionId(PLAY_KNIGHT));
            fast_forward(&mut state, &mut rng);
        }

        assert!(
            matches!(state.largest_army, Some((Player::One, 3))),
            "P1 should have LA after 3 knights, got {:?}",
            state.largest_army
        );
    }

    /// Both players at 3 knights, first achiever keeps LA.
    #[test]
    fn largest_army_tie_holder_keeps() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        // P1 plays 3 knights
        state.current_player = Player::One;
        state.players[Player::One].dev_cards[DevCardKind::Knight] = 3;
        for _ in 0..3 {
            state.phase = Phase::Main;
            state.current_player = Player::One;
            state.players[Player::One].has_played_dev_card_this_turn = false;
            apply(&mut state, ActionId(PLAY_KNIGHT));
            fast_forward(&mut state, &mut rng);
        }
        assert!(matches!(state.largest_army, Some((Player::One, 3))));

        // P2 plays 3 knights to tie
        state.players[Player::Two].dev_cards[DevCardKind::Knight] = 3;
        for _ in 0..3 {
            state.phase = Phase::Main;
            state.current_player = Player::Two;
            state.players[Player::Two].has_played_dev_card_this_turn = false;
            apply(&mut state, ActionId(PLAY_KNIGHT));
            fast_forward(&mut state, &mut rng);
        }

        // P1 should keep LA on tie
        assert!(
            matches!(state.largest_army, Some((Player::One, 3))),
            "first achiever should keep LA on tie, got {:?}",
            state.largest_army
        );
    }

    // --- Robber tests ---

    /// Roll a 7 with > 9 cards, verify full discard -> move robber -> steal sequence.
    #[test]
    fn seven_triggers_discard_robber_steal() {
        let mut state = make_state_with_seed(42);
        play_setup(&mut state);

        // Give P1 10 cards (> 9 threshold)
        state.players[Player::One].hand = ResourceArray::new(2, 2, 2, 2, 2);
        state.current_player = Player::One;

        // Directly call handle_seven instead of rolling (to avoid randomness)
        handle_seven(&mut state);

        assert!(
            matches!(
                state.phase,
                Phase::Discard {
                    player: Player::One,
                    remaining: 5
                }
            ),
            "P1 with 10 cards should discard 5, got {:?}",
            state.phase
        );

        // Discard 5 cards
        for _ in 0..5 {
            let mut actions = Vec::new();
            action::legal_actions(&state, &mut actions);
            apply(&mut state, actions[0]);
        }

        assert!(
            matches!(state.phase, Phase::MoveRobber),
            "after discarding, should move robber, got {:?}",
            state.phase
        );
    }

    /// Verify populate_move_robber excludes current robber tile.
    #[test]
    fn robber_must_move_to_different_tile() {
        let mut state = make_state_with_seed(42);
        play_setup(&mut state);
        // Disable friendly robber so all non-current tiles are legal
        state.players[Player::Two].victory_points = FRIENDLY_ROBBER_VP;
        state.phase = Phase::MoveRobber;

        let mut actions = Vec::new();
        action::legal_actions(&state, &mut actions);

        let current_robber = state.robber;
        for a in &actions {
            let tid = a.robber_tile();
            assert_ne!(tid, current_robber, "robber must move to a different tile");
        }
        assert_eq!(
            actions.len(),
            18,
            "should have 18 legal tiles (19 - current)"
        );
    }

    /// Move robber to opponent's tile when they have no cards; steal does nothing.
    #[test]
    fn steal_from_empty_hand_is_noop() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        state.current_player = Player::One;

        // Find a tile where P2 has a building
        let opp_buildings = state.player_buildings(Player::Two);
        let topo = &state.topology;
        let mut target_tile = None;
        for tile in &topo.tiles {
            if tile.id == state.robber {
                continue;
            }
            let tile_mask = topo.adj.tile_nodes[tile.id.0 as usize];
            if tile_mask & opp_buildings != 0 {
                target_tile = Some(tile.id);
                break;
            }
        }

        if let Some(tid) = target_tile {
            // Empty P2's hand
            state.players[Player::Two].hand = ResourceArray::default();
            // Give P2 enough public VP to bypass friendly robber
            state.players[Player::Two].victory_points = 3;
            state.phase = Phase::MoveRobber;

            let p1_hand_before = state.players[Player::One].hand;
            apply(&mut state, robber_id(tid));

            // With empty opponent hand, should go to Main, not Steal
            assert!(
                matches!(state.phase, Phase::Main),
                "empty opponent hand should skip steal, got {:?}",
                state.phase
            );
            assert_eq!(state.players[Player::One].hand, p1_hand_before);
        }
    }

    /// When opponent has <= 2 public VP, robber can't be placed on their tiles
    /// and steal is skipped.
    #[test]
    fn friendly_robber_blocks_targeting() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        state.current_player = Player::One;
        // P2 has 2 VP from setup settlements (< FRIENDLY_ROBBER_VP=3)
        assert!(
            state.public_vps(Player::Two) < FRIENDLY_ROBBER_VP,
            "P2 should have < 3 public VP after setup"
        );

        state.phase = Phase::MoveRobber;
        let mut actions = Vec::new();
        action::legal_actions(&state, &mut actions);

        // None of the robber actions should target tiles with P2 buildings
        let opp_buildings = state.player_buildings(Player::Two);
        let topo = &state.topology;
        for a in &actions {
            let tid = a.robber_tile();
            let tile_mask = topo.adj.tile_nodes[tid.0 as usize];
            assert_eq!(
                tile_mask & opp_buildings,
                0,
                "friendly robber should exclude tiles with opponent buildings"
            );
        }

        // Also verify: if we place robber on a tile adjacent to P2 buildings
        // via direct apply, it should skip steal
        let mut target_tile = None;
        for tile in &topo.tiles {
            if tile.id == state.robber {
                continue;
            }
            let tile_mask = topo.adj.tile_nodes[tile.id.0 as usize];
            if tile_mask & opp_buildings != 0 {
                target_tile = Some(tile.id);
                break;
            }
        }
        if let Some(tid) = target_tile {
            state.players[Player::Two].hand = ResourceArray::new(1, 1, 1, 1, 1);
            state.phase = Phase::MoveRobber;
            apply(&mut state, robber_id(tid));
            assert!(
                matches!(state.phase, Phase::Main),
                "friendly robber should skip steal phase, got {:?}",
                state.phase
            );
        }
    }

    /// When opponent has >= 3 public VP, normal robber rules apply.
    #[test]
    fn friendly_robber_allows_targeting_above_threshold() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        state.current_player = Player::One;
        // Give P2 enough VP to exceed friendly robber threshold
        state.players[Player::Two].victory_points = 3;

        state.phase = Phase::MoveRobber;
        let mut actions = Vec::new();
        action::legal_actions(&state, &mut actions);

        // Should include tiles with P2 buildings
        let opp_buildings = state.player_buildings(Player::Two);
        let topo = &state.topology;
        let has_opp_tile = actions.iter().any(|a| {
            let tid = a.robber_tile();
            let tile_mask = topo.adj.tile_nodes[tid.0 as usize];
            tile_mask & opp_buildings != 0
        });
        assert!(
            has_opp_tile,
            "above threshold, robber should be able to target opponent tiles"
        );

        // Place robber on opponent's tile with cards -> should enter Steal
        let target_tile = actions.iter().find_map(|a| {
            let tid = a.robber_tile();
            let tile_mask = topo.adj.tile_nodes[tid.0 as usize];
            if tile_mask & opp_buildings != 0 {
                Some(tid)
            } else {
                None
            }
        });
        if let Some(tid) = target_tile {
            state.players[Player::Two].hand = ResourceArray::new(1, 1, 1, 1, 1);
            state.phase = Phase::MoveRobber;
            apply(&mut state, robber_id(tid));
            assert!(
                matches!(state.phase, Phase::Steal),
                "above threshold with cards, should enter Steal, got {:?}",
                state.phase
            );
        }
    }

    /// Opponent has 2 settlements + 1 VP dev card (3 total but only 2 public).
    /// Friendly robber still protects them because VP cards don't count as public VP.
    #[test]
    fn friendly_robber_ignores_vp_cards() {
        let mut state = make_state_with_seed(42);
        play_setup(&mut state);

        state.current_player = Player::One;
        // P2 has 2 VP from settlements + 1 VP dev card, but only 2 public
        state.players[Player::Two].dev_cards[DevCardKind::VictoryPoint] = 1;
        assert_eq!(state.public_vps(Player::Two), 2);

        state.phase = Phase::MoveRobber;
        let mut actions = Vec::new();
        action::legal_actions(&state, &mut actions);

        // Should still block opponent tiles since public VP = 2 < 3
        let opp_buildings = state.player_buildings(Player::Two);
        let topo = &state.topology;
        for a in &actions {
            let tid = a.robber_tile();
            let tile_mask = topo.adj.tile_nodes[tid.0 as usize];
            assert_eq!(
                tile_mask & opp_buildings,
                0,
                "VP dev cards shouldn't count for friendly robber"
            );
        }
    }

    // --- Building tests ---

    /// After placing settlements, verify adjacent nodes are excluded from legal
    /// settlement actions (distance rule).
    #[test]
    fn distance_rule_enforced() {
        let mut state = make_state_with_seed(42);
        play_setup(&mut state);

        state.phase = Phase::Main;
        state.current_player = Player::One;

        let adj = &state.topology.adj;
        let occupied = state.occupied_nodes();

        // All occupied nodes and their neighbors should be blocked
        let mut blocked = occupied;
        let mut bits = occupied;
        while bits != 0 {
            let node = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            blocked |= adj.node_adj_nodes[node];
        }

        // Give P1 resources and roads to build everywhere
        state.players[Player::One].hand = ResourceArray::new(19, 19, 19, 19, 19);
        let mut actions = Vec::new();
        action::legal_actions(&state, &mut actions);
        let settlement_actions: Vec<_> = actions
            .iter()
            .filter(|a| a.0 >= SETTLEMENT_START && a.0 < SETTLEMENT_END)
            .collect();

        for a in &settlement_actions {
            let nid = a.settlement_node();
            assert_eq!(
                blocked & (1u64 << nid.0),
                0,
                "settlement at {:?} violates distance rule",
                nid
            );
        }
    }

    /// Place all 5 settlements, verify no more settlement actions appear.
    #[test]
    fn piece_limits_enforced() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        // P1 already has 2 settlements from setup, place 3 more
        for _ in 0..3 {
            state.phase = Phase::Main;
            state.current_player = Player::One;
            state.players[Player::One].hand.add(SETTLEMENT_COST);

            let mut actions = Vec::new();
            action::legal_actions(&state, &mut actions);
            let settle = actions
                .iter()
                .find(|a| a.0 >= SETTLEMENT_START && a.0 < SETTLEMENT_END);
            if let Some(&a) = settle {
                apply(&mut state, a);
            } else {
                break; // No more valid settlement spots on road
            }
        }

        // If all 5 are placed, no more settlement actions should appear
        if state.players[Player::One].settlements_left == 0 {
            state.phase = Phase::Main;
            state.current_player = Player::One;
            state.players[Player::One].hand = ResourceArray::new(19, 19, 19, 19, 19);
            let mut actions = Vec::new();
            action::legal_actions(&state, &mut actions);
            let has_settle = actions
                .iter()
                .any(|a| a.0 >= SETTLEMENT_START && a.0 < SETTLEMENT_END);
            assert!(
                !has_settle,
                "should not have settlement actions with 0 settlements left"
            );
        }
    }

    /// Verify disconnected edges don't appear in road actions.
    #[test]
    fn roads_must_connect_to_network() {
        let mut state = make_state_with_seed(42);
        play_setup(&mut state);

        state.phase = Phase::Main;
        state.current_player = Player::One;
        state.players[Player::One].hand.add(ROAD_COST);

        let mut actions = Vec::new();
        action::legal_actions(&state, &mut actions);
        let road_actions: Vec<_> = actions
            .iter()
            .filter(|a| a.0 >= ROAD_START && a.0 < ROAD_END)
            .collect();

        let my_buildings = state.boards[Player::One].settlements | state.boards[Player::One].cities;
        let my_roads = state.boards[Player::One].road_network.roads;
        let adj = &state.topology.adj;

        for a in &road_actions {
            let eid = a.road_edge();
            let endpoints = adj.edge_endpoints[eid.0 as usize];
            // Must connect to an existing building or road
            let touches_building = endpoints & my_buildings != 0;
            let mut touches_road = false;
            let mut ep = endpoints;
            while ep != 0 {
                let node = ep.trailing_zeros() as usize;
                ep &= ep - 1;
                if adj.node_adj_edges[node] & my_roads != 0 {
                    touches_road = true;
                    break;
                }
            }
            assert!(
                touches_building || touches_road,
                "road {:?} is disconnected from network",
                eid
            );
        }
    }

    /// Play Road Building with only 1 road left, verify only 1 road built
    /// then back to Main.
    #[test]
    fn road_building_card_partial() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        state.current_player = Player::One;
        state.phase = Phase::Main;
        state.players[Player::One].roads_left = 1;
        state.players[Player::One].dev_cards[DevCardKind::RoadBuilding] = 1;
        state.players[Player::One].has_played_dev_card_this_turn = false;

        apply(&mut state, ActionId(PLAY_ROAD_BUILDING));
        assert!(matches!(state.phase, Phase::RoadBuilding { roads_left: 2 }));

        // Build one road
        let mut actions = Vec::new();
        action::legal_actions(&state, &mut actions);
        let road = actions
            .iter()
            .find(|a| a.0 >= ROAD_START && a.0 < ROAD_END)
            .copied();
        if let Some(r) = road {
            apply(&mut state, r);
            // Should be back to Main since roads_left is now 0
            assert!(
                matches!(state.phase, Phase::Main),
                "with 0 roads left, should return to Main, got {:?}",
                state.phase
            );
        }
    }

    /// Road Building card is not offered when all 15 roads have been placed.
    #[test]
    fn road_building_unavailable_with_no_roads_left() {
        let mut state = make_state_with_seed(42);
        play_setup(&mut state);

        state.current_player = Player::One;
        state.phase = Phase::Main;
        state.players[Player::One].roads_left = 0;
        state.players[Player::One].dev_cards[DevCardKind::RoadBuilding] = 1;
        state.players[Player::One].has_played_dev_card_this_turn = false;

        let mut actions = Vec::new();
        action::legal_actions(&state, &mut actions);
        assert!(
            !actions.contains(&ActionId(PLAY_ROAD_BUILDING)),
            "should not offer Road Building when roads_left == 0"
        );
    }

    /// With 14 roads placed (1 left), Road Building allows exactly 1 placement
    /// then returns to Main.
    #[test]
    fn road_building_with_one_road_left() {
        let mut state = make_state_with_seed(42);
        play_setup(&mut state);

        state.current_player = Player::One;
        state.phase = Phase::Main;
        state.players[Player::One].roads_left = 1;
        state.players[Player::One].dev_cards[DevCardKind::RoadBuilding] = 1;
        state.players[Player::One].has_played_dev_card_this_turn = false;

        // Card should be playable with 1 road left
        let mut actions = Vec::new();
        action::legal_actions(&state, &mut actions);
        assert!(
            actions.contains(&ActionId(PLAY_ROAD_BUILDING)),
            "should offer Road Building when roads_left == 1"
        );

        // Play the card — enters RoadBuilding phase with roads_left: 2
        apply(&mut state, ActionId(PLAY_ROAD_BUILDING));
        assert!(matches!(state.phase, Phase::RoadBuilding { roads_left: 2 }));

        // Should have road actions available
        action::legal_actions(&state, &mut actions);
        let road = actions
            .iter()
            .find(|a| a.0 >= ROAD_START && a.0 < ROAD_END)
            .copied()
            .expect("should have at least one legal road placement");

        // Place the one road — should return to Main since roads_left hits 0
        apply(&mut state, road);
        assert!(
            matches!(state.phase, Phase::Main),
            "should return to Main after placing the only remaining road, got {:?}",
            state.phase
        );
        assert_eq!(state.players[Player::One].roads_left, 0);
    }

    // --- Dev Card tests ---

    /// Buy a knight, verify it doesn't appear in playable actions same turn.
    #[test]
    fn cannot_play_card_bought_this_turn() {
        let mut state = make_state_with_seed(42);
        play_setup(&mut state);

        state.current_player = Player::One;
        state.phase = Phase::Main;
        state.players[Player::One].has_played_dev_card_this_turn = false;

        // Manually give P1 a knight as if bought this turn
        state.players[Player::One].dev_cards[DevCardKind::Knight] = 1;
        state.players[Player::One].dev_cards_bought_this_turn[DevCardKind::Knight] = 1;

        let mut actions = Vec::new();
        action::legal_actions(&state, &mut actions);
        assert!(
            !actions.contains(&ActionId(PLAY_KNIGHT)),
            "should not be able to play knight bought this turn"
        );
    }

    /// Play a knight, verify no other dev cards are playable that turn.
    #[test]
    fn one_dev_card_per_turn() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        state.current_player = Player::One;
        state.phase = Phase::Main;
        state.players[Player::One].dev_cards[DevCardKind::Knight] = 2;
        state.players[Player::One].dev_cards[DevCardKind::Monopoly] = 1;
        state.players[Player::One].has_played_dev_card_this_turn = false;

        // Play a knight
        apply(&mut state, ActionId(PLAY_KNIGHT));

        // Fast-forward through robber
        fast_forward(&mut state, &mut rng);

        let mut actions = Vec::new();
        action::legal_actions(&state, &mut actions);
        assert!(
            !actions.contains(&ActionId(PLAY_KNIGHT)),
            "should not play second dev card in same turn"
        );
        let has_monopoly = actions.iter().any(|a| a.0 >= 200 && a.0 < 205); // MONOPOLY range
        assert!(
            !has_monopoly,
            "should not play monopoly after knight same turn"
        );
    }

    /// Set player to 14 VP, buy a VP dev card, verify GameOver.
    #[test]
    fn vp_card_wins_immediately() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        state.current_player = Player::One;
        state.phase = Phase::Main;

        // Set P1 to 14 VP (12 from buildings + 2 VP dev cards)
        state.players[Player::One].victory_points = 12;
        state.players[Player::One].dev_cards[DevCardKind::VictoryPoint] = 2;

        // Stack the deck with a VP card
        state.dev_deck = DevCardDeck::from_cards(&[DevCardKind::VictoryPoint]);
        state.players[Player::One].hand.add(DEV_CARD_COST);

        apply(&mut state, ActionId(BUY_DEV_CARD));

        assert!(
            matches!(state.phase, Phase::GameOver(Player::One)),
            "should win immediately on VP card purchase, got {:?}",
            state.phase
        );
    }

    // --- Setup tests ---

    /// Verify turn order: P1, P2, P2, P1 during setup.
    #[test]
    fn setup_snake_draft_order() {
        let mut state = make_state_with_seed(42);
        let mut actions = Vec::new();

        // Settlement 1: P1
        assert_eq!(state.current_player, Player::One);
        action::legal_actions(&state, &mut actions);
        apply(&mut state, actions[0]);
        // Road 1: still P1
        action::legal_actions(&state, &mut actions);
        apply(&mut state, actions[0]);

        // Settlement 2: P2
        assert_eq!(state.current_player, Player::Two);
        action::legal_actions(&state, &mut actions);
        apply(&mut state, actions[0]);
        // Road 2: still P2
        action::legal_actions(&state, &mut actions);
        apply(&mut state, actions[0]);

        // Settlement 3: P2 again (snake)
        assert_eq!(state.current_player, Player::Two);
        action::legal_actions(&state, &mut actions);
        apply(&mut state, actions[0]);
        action::legal_actions(&state, &mut actions);
        apply(&mut state, actions[0]);

        // Settlement 4: P1 again
        assert_eq!(state.current_player, Player::One);
    }

    /// Verify settlements 3 and 4 (second placements) give starting resources.
    #[test]
    fn second_settlement_grants_resources() {
        let mut state = make_state_with_seed(42);
        let mut actions = Vec::new();

        // Play first 2 settlement+road pairs (no resources given)
        for _ in 0..2 {
            action::legal_actions(&state, &mut actions);
            apply(&mut state, actions[0]);
            action::legal_actions(&state, &mut actions);
            apply(&mut state, actions[0]);
        }

        // Record hands before 3rd settlement
        let hand_before_3 = state.players[Player::Two].hand.total();
        action::legal_actions(&state, &mut actions);
        apply(&mut state, actions[0]); // 3rd settlement (P2)
        let hand_after_3 = state.players[Player::Two].hand.total();
        assert!(
            hand_after_3 > hand_before_3,
            "3rd settlement should grant resources"
        );

        // Road for 3rd settlement
        action::legal_actions(&state, &mut actions);
        apply(&mut state, actions[0]);

        // 4th settlement (P1)
        let hand_before_4 = state.players[Player::One].hand.total();
        action::legal_actions(&state, &mut actions);
        apply(&mut state, actions[0]);
        let hand_after_4 = state.players[Player::One].hand.total();
        assert!(
            hand_after_4 > hand_before_4,
            "4th settlement should grant resources"
        );
    }

    /// Verify road actions in PlaceRoad phase only include edges adjacent to
    /// the just-placed settlement.
    #[test]
    fn setup_road_must_touch_settlement() {
        let mut state = make_state_with_seed(42);

        // Place first settlement
        let mut actions = Vec::new();
        action::legal_actions(&state, &mut actions);
        let settle_action = actions[0];
        let settle_node = settle_action.settlement_node();
        apply(&mut state, settle_action);

        assert!(matches!(state.phase, Phase::PlaceRoad));

        action::legal_actions(&state, &mut actions);

        let adj = &state.topology.adj;
        let node_edges = adj.node_adj_edges[settle_node.0 as usize];

        for a in &actions {
            let eid = a.road_edge();
            assert_ne!(
                node_edges & (1u128 << eid.0),
                0,
                "road {:?} doesn't touch settlement {:?}",
                eid,
                settle_node
            );
        }
    }

    // --- Resource Production tests ---

    /// Place robber on a tile, roll its number, verify no resources distributed.
    #[test]
    fn robber_blocks_production() {
        let mut state = make_state_with_seed(42);
        play_setup(&mut state);
        let topo = Arc::clone(&state.topology);

        // Find a tile that has a P1 settlement and a dice number
        let p1_buildings = state.player_buildings(Player::One);
        let mut target = None;
        for tile in &topo.tiles {
            let tile_mask = topo.adj.tile_nodes[tile.id.0 as usize];
            if tile_mask & p1_buildings != 0 && tile.terrain.resource().is_some() {
                for roll in 2..=12u8 {
                    if topo.dice_to_tiles[roll as usize].contains(&tile.id) {
                        target = Some((tile.id, roll));
                        break;
                    }
                }
                if target.is_some() {
                    break;
                }
            }
        }

        let (tid, roll) = target.expect("should find a tile with P1 settlement");
        // Clear hands, place robber, and distribute
        state.players[Player::One].hand = ResourceArray::default();
        state.players[Player::Two].hand = ResourceArray::default();
        state.robber = tid;
        distribute_resources(&mut state, roll);

        // Now compare: remove robber and distribute again
        let hand_with_robber = state.players[Player::One].hand;
        state.players[Player::One].hand = ResourceArray::default();
        state.players[Player::Two].hand = ResourceArray::default();
        // Move robber to a tile that doesn't produce this roll
        let other_tile = topo
            .tiles
            .iter()
            .find(|t| !topo.dice_to_tiles[roll as usize].contains(&t.id))
            .unwrap();
        state.robber = other_tile.id;
        distribute_resources(&mut state, roll);
        let hand_without_robber = state.players[Player::One].hand;

        // With robber blocking, P1 should get fewer (or equal) resources
        assert!(
            hand_with_robber.total() <= hand_without_robber.total(),
            "robber should block production: with={}, without={}",
            hand_with_robber.total(),
            hand_without_robber.total(),
        );
    }

    /// City on a tile produces 2 resources vs settlement's 1.
    #[test]
    fn city_produces_double() {
        let mut state = make_state_with_seed(42);
        play_setup(&mut state);

        let topo = &state.topology;
        let p1_settlements = state.boards[Player::One].settlements;

        // Find a P1 settlement on a producing tile
        let mut target = None;
        let mut bits = p1_settlements;
        while bits != 0 {
            let nid = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            let node = &topo.nodes[nid as usize];
            for &tid in &node.adjacent_tiles {
                let tile = &topo.tiles[tid.0 as usize];
                if let Some(resource) = tile.terrain.resource() {
                    if tid != state.robber {
                        for roll in 2..=12u8 {
                            if topo.dice_to_tiles[roll as usize].contains(&tid) {
                                target = Some((NodeId(nid), tid, roll, resource));
                                break;
                            }
                        }
                    }
                }
                if target.is_some() {
                    break;
                }
            }
            if target.is_some() {
                break;
            }
        }

        if let Some((nid, _tid, roll, resource)) = target {
            // Record settlement production
            state.players[Player::One].hand = ResourceArray::default();
            state.players[Player::Two].hand = ResourceArray::default();
            distribute_resources(&mut state, roll);
            let settlement_yield = state.players[Player::One].hand[resource];

            // Upgrade to city
            let bit = 1u64 << nid.0;
            state.boards[Player::One].settlements &= !bit;
            state.boards[Player::One].cities |= bit;

            state.players[Player::One].hand = ResourceArray::default();
            state.players[Player::Two].hand = ResourceArray::default();
            distribute_resources(&mut state, roll);
            let city_yield = state.players[Player::One].hand[resource];

            assert_eq!(
                city_yield,
                settlement_yield * 2,
                "city should produce double: city={}, settlement={}",
                city_yield,
                settlement_yield
            );
        }
    }

    /// When bank can't cover total demand for a resource, nobody gets it.
    #[test]
    fn insufficient_bank_skips_resource_type() {
        let mut state = make_state_with_seed(42);
        play_setup(&mut state);

        let topo = &state.topology;

        // Find a roll that produces at least one resource from a tile adjacent to P1.
        let p1_buildings = state.player_buildings(Player::One);
        let mut target_roll = None;
        let mut scarce_resource = None;
        let mut other_resource = None;

        'outer: for roll in 2..=12u8 {
            for &tid in &topo.dice_to_tiles[roll as usize] {
                if tid == state.robber {
                    continue;
                }
                let tile_mask = topo.adj.tile_nodes[tid.0 as usize];
                if tile_mask & p1_buildings != 0 {
                    if let Some(r) = topo.tiles[tid.0 as usize].terrain.resource() {
                        if scarce_resource.is_none() {
                            scarce_resource = Some(r);
                        } else if Some(r) != scarce_resource && other_resource.is_none() {
                            other_resource = Some(r);
                        }
                    }
                }
            }
            if scarce_resource.is_some() {
                target_roll = Some(roll);
                break 'outer;
            }
        }

        let roll = target_roll.expect("should find a producing roll for P1");
        let scarce = scarce_resource.expect("should find a scarce resource");

        // Compute demand for the scarce resource on this roll.
        let mut demand = 0u8;
        for &tid in &topo.dice_to_tiles[roll as usize] {
            if tid == state.robber {
                continue;
            }
            if topo.tiles[tid.0 as usize].terrain.resource() != Some(scarce) {
                continue;
            }
            let tile_mask = topo.adj.tile_nodes[tid.0 as usize];
            for &pid in &[Player::One, Player::Two] {
                let s = (state.boards[pid].settlements & tile_mask).count_ones() as u8;
                let c = (state.boards[pid].cities & tile_mask).count_ones() as u8;
                demand += s + c * 2;
            }
        }

        // Set bank just below total demand so nobody gets it.
        state.bank[scarce] = demand.saturating_sub(1);
        state.players[Player::One].hand = ResourceArray::default();
        state.players[Player::Two].hand = ResourceArray::default();

        distribute_resources(&mut state, roll);

        // Nobody should receive the scarce resource.
        assert_eq!(
            state.players[Player::One].hand[scarce],
            0,
            "P1 should not receive scarce resource when bank < total demand"
        );
        assert_eq!(
            state.players[Player::Two].hand[scarce],
            0,
            "P2 should not receive scarce resource when bank < total demand"
        );

        // If another resource was also produced by this roll, it should be distributed normally.
        if let Some(other) = other_resource {
            let p1_got = state.players[Player::One].hand[other];
            let p2_got = state.players[Player::Two].hand[other];
            assert!(
                p1_got > 0 || p2_got > 0,
                "other resource ({other}) should still be distributed"
            );
        }
    }

    // --- Trading tests ---

    /// Execute a maritime trade, verify hand and bank change correctly.
    #[test]
    fn maritime_trade_execution() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        state.current_player = Player::One;
        state.phase = Phase::Main;
        // Give P1 4 lumber for a 4:1 trade
        state.players[Player::One].trade_ratios = [4; 5];
        state.players[Player::One].hand = ResourceArray::new(4, 0, 0, 0, 0);

        let give = Resource::Lumber;
        let recv = Resource::Brick;
        let bank_lumber_before = state.bank[Resource::Lumber];
        let bank_brick_before = state.bank[Resource::Brick];

        apply(&mut state, maritime_id(give, recv));

        assert_eq!(state.players[Player::One].hand[Resource::Lumber], 0);
        assert_eq!(state.players[Player::One].hand[Resource::Brick], 1);
        assert_eq!(state.bank[Resource::Lumber], bank_lumber_before + 4);
        assert_eq!(state.bank[Resource::Brick], bank_brick_before - 1);
    }

    /// Execute two maritime trades in one turn, verify both succeed.
    #[test]
    fn multiple_maritime_trades_per_turn() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        state.current_player = Player::One;
        state.phase = Phase::Main;
        state.players[Player::One].trade_ratios = [4; 5];
        state.players[Player::One].hand = ResourceArray::new(8, 0, 0, 0, 0);

        apply(&mut state, maritime_id(Resource::Lumber, Resource::Brick));
        assert_eq!(state.players[Player::One].hand[Resource::Lumber], 4);
        assert_eq!(state.players[Player::One].hand[Resource::Brick], 1);

        apply(&mut state, maritime_id(Resource::Lumber, Resource::Wool));
        assert_eq!(state.players[Player::One].hand[Resource::Lumber], 0);
        assert_eq!(state.players[Player::One].hand[Resource::Wool], 1);
    }

    // --- Winning tests ---

    /// Verify check_victory only triggers for current player.
    #[test]
    fn win_only_on_own_turn() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        // Give P2 enough VP to win
        state.players[Player::Two].victory_points = 13;
        state.players[Player::Two].dev_cards[DevCardKind::VictoryPoint] = 2;

        // But it's P1's turn — building something should not trigger P2's win
        state.current_player = Player::One;
        state.phase = Phase::Main;
        state.players[Player::One].hand.add(ROAD_COST);

        let mut actions = Vec::new();
        action::legal_actions(&state, &mut actions);
        let road = actions
            .iter()
            .find(|a| a.0 >= ROAD_START && a.0 < ROAD_END)
            .copied();
        if let Some(r) = road {
            apply(&mut state, r);
            assert!(
                !matches!(state.phase, Phase::GameOver(Player::Two)),
                "P2 should not win on P1's turn"
            );
        }
    }

    /// Reach 15 VP through various sources, verify GameOver.
    #[test]
    fn win_at_fifteen_vp() {
        let mut state = make_state_with_seed(42);
        let mut rng = play_setup(&mut state);

        state.current_player = Player::One;
        state.phase = Phase::Main;

        // 12 from buildings + 2 VP dev cards = 14 total
        // Buy a VP dev card → 15 → win
        state.players[Player::One].victory_points = 12;
        state.players[Player::One].dev_cards[DevCardKind::VictoryPoint] = 2;
        state.dev_deck = DevCardDeck::from_cards(&[DevCardKind::VictoryPoint]);
        state.players[Player::One].hand.add(DEV_CARD_COST);

        apply(&mut state, ActionId(BUY_DEV_CARD));

        assert!(
            matches!(state.phase, Phase::GameOver(Player::One)),
            "P1 should win at 15 VP, got {:?}",
            state.phase
        );
    }
}
