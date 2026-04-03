use std::fmt;
use std::sync::Arc;

use canopy::player::{PerPlayer, Player};

use super::board::{NodeId, TileId};
use super::dev_card::{DevCardArray, DevCardDeck, DevCardKind};

/// Original deck composition: 14 knights, 5 VP, 2 road building, 2 year of plenty, 2 monopoly.
pub const ORIGINAL_DEV_DECK: [u8; 5] = [14, 5, 2, 2, 2];
use super::dice::Dice;
use super::resource::ResourceArray;
use super::road::RoadNetwork;
use super::topology::Topology;

/// Bitboard representation of a player's buildings and road network.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerBoards {
    /// Bitmask of node positions with settlements (bits 0..54).
    pub settlements: u64,
    /// Bitmask of node positions with cities (bits 0..54).
    pub cities: u64,
    /// Road network: placed roads, reachable frontier, and cached longest road.
    pub road_network: RoadNetwork,
}

/// Per-player game state: resources, pieces, dev cards, and scoring.
#[derive(Clone, Debug)]
pub struct PlayerState {
    /// Resource cards currently in hand.
    pub hand: ResourceArray,
    /// Development cards held but not yet played (includes VP cards).
    pub dev_cards: DevCardArray,
    /// Dev cards bought this turn (cannot be played until next turn).
    pub dev_cards_bought_this_turn: DevCardArray,
    /// Total knight cards played (counts toward largest army; threshold is 3).
    pub knights_played: u8,
    /// Total road segments placed on the board.
    pub roads_placed: u8,
    /// Remaining settlement pieces (starts at 5).
    pub settlements_left: u8,
    /// Remaining city pieces (starts at 4).
    pub cities_left: u8,
    /// Remaining road pieces (starts at 15).
    pub roads_left: u8,
    /// Whether a dev card has been played this turn (limit one per turn).
    pub has_played_dev_card_this_turn: bool,
    /// Cumulative count of each dev card type played (for information tracking).
    pub dev_cards_played: DevCardArray,
    /// Dev cards bought but not yet revealed. In colonist mode, set from
    /// the opponent's purchases. During self-play search, `determinize()`
    /// temporarily moves the opponent's known cards here before resampling.
    pub hidden_dev_cards: u8,
    /// How many of `hidden_dev_cards` were bought this turn (cannot be played).
    /// When card identities are revealed, this transfers to `dev_cards_bought_this_turn`.
    pub hidden_dev_cards_bought_this_turn: u8,
    /// VP from buildings only: +1 per settlement, +1 more per city upgrade.
    /// Does **not** include longest road, largest army, or VP dev cards.
    /// For total VP, combine with `dev_cards[VictoryPoint]` and the
    /// `longest_road`/`largest_army` awards on `GameState`.
    pub building_vps: u8,
    /// Best maritime trade ratio per resource type (indexed by `Resource`).
    /// Defaults to 4:1; improved to 3:1 by a generic port or 2:1 by a
    /// resource-specific port.
    pub trade_ratios: [u8; 5],
}

impl Default for PlayerState {
    fn default() -> Self {
        Self {
            hand: ResourceArray::default(),
            dev_cards: DevCardArray::default(),
            dev_cards_bought_this_turn: DevCardArray::default(),
            knights_played: 0,
            roads_placed: 0,
            settlements_left: 5,
            cities_left: 4,
            roads_left: 15,
            has_played_dev_card_this_turn: false,
            dev_cards_played: DevCardArray::default(),
            hidden_dev_cards: 0,
            hidden_dev_cards_bought_this_turn: 0,
            building_vps: 0,
            trade_ratios: [4; 5],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Phase {
    PlaceSettlement,
    PlaceRoad,
    PreRoll,
    Roll,
    Discard {
        player: Player,
        remaining: u8,
        roller: Player,
        /// Lexicographic lower bound on resource index to prevent
        /// transpositions (discard lumber before brick before wool …).
        min_resource: u8,
    },
    MoveRobber,
    StealResolve,
    Main,
    DevCardDraw,
    RoadBuilding {
        roads_left: u8,
    },
    GameOver(Player),
}

#[derive(Clone)]
pub struct GameState {
    pub topology: Arc<Topology>,

    pub players: PerPlayer<PlayerState>,
    pub current_player: Player,

    pub boards: PerPlayer<PlayerBoards>,

    pub bank: ResourceArray,
    pub dev_deck: DevCardDeck,
    pub robber: TileId,

    pub phase: Phase,
    pub turn_number: u16,
    pub pre_roll: bool,

    pub setup_count: u8,

    pub longest_road: Option<(Player, u8)>,
    pub largest_army: Option<(Player, u8)>,
    pub dice: Dice,
    pub vp_limit: u8,
    pub discard_threshold: u8,

    /// The node of the most recently placed setup settlement. Used by
    /// `populate_place_road` to determine which settlement needs a road.
    pub last_setup_node: Option<NodeId>,

    // ── Canonical build ordering (transposition elimination) ─────────
    /// Minimum ordered action type still allowed this turn.
    /// 0 = all (dev buy, city, settle), 1 = city+settle, 2 = settle only.
    pub min_build_type: u8,
    /// Ordered cities must target nodes > this value.
    pub min_city_node: u8,
    /// Ordered (non-port) settlements must target nodes > this value.
    pub min_settle_node: u8,
    /// Snapshot of current player's settlement bitmask at turn start.
    /// Cities on nodes in this mask are "ordered"; others are "unordered"
    /// (same-turn settlements that must precede their city upgrade).
    pub settlements_at_turn_start: u64,
    /// When false, `legal_actions` skips canonical ordering filters and
    /// apply functions skip updating ordering fields. Used during colonist
    /// replay where the event log may use non-canonical action orderings.
    pub canonical_build_order: bool,
}

impl fmt::Debug for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GameState")
            .field("phase", &self.phase)
            .field("current_player", &self.current_player)
            .field("turn_number", &self.turn_number)
            .field(
                "topology",
                &format_args!(
                    "Topology({} tiles, {} nodes, {} edges)",
                    self.topology.tiles.len(),
                    self.topology.nodes.len(),
                    self.topology.edges.len()
                ),
            )
            .field("players", &self.players)
            .field("bank", &self.bank)
            .field("robber", &self.robber)
            .field("longest_road", &self.longest_road)
            .field("largest_army", &self.largest_army)
            .finish()
    }
}

impl GameState {
    pub fn from_seed(seed: u64, dice: Dice) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let topology = Arc::new(Topology::from_seed(rng.u64(..)));
        let dev_deck = DevCardDeck::new();
        Self::new(topology, dev_deck, dice)
    }

    pub fn new(topology: Arc<Topology>, dev_deck: DevCardDeck, dice: Dice) -> Self {
        let robber = topology.robber_start;

        Self {
            topology,
            players: PerPlayer::default(),
            current_player: Player::One,
            boards: PerPlayer::default(),
            bank: ResourceArray::new(19, 19, 19, 19, 19),
            dev_deck,
            robber,
            phase: Phase::PlaceSettlement,
            turn_number: 0,
            pre_roll: false,
            setup_count: 0,
            longest_road: None,
            largest_army: None,
            dice,
            vp_limit: 15,
            discard_threshold: 9,
            last_setup_node: None,
            min_build_type: 0,
            min_city_node: 0,
            min_settle_node: 0,
            settlements_at_turn_start: 0,
            canonical_build_order: true,
        }
    }

    pub fn current(&self) -> &PlayerState {
        &self.players[self.current_player]
    }

    pub fn current_mut(&mut self) -> &mut PlayerState {
        &mut self.players[self.current_player]
    }

    pub fn occupied_nodes(&self) -> u64 {
        (self.boards[Player::One].settlements | self.boards[Player::One].cities)
            | (self.boards[Player::Two].settlements | self.boards[Player::Two].cities)
    }

    pub fn all_roads(&self) -> u128 {
        self.boards[Player::One].road_network.roads | self.boards[Player::Two].road_network.roads
    }

    pub fn player_buildings(&self, pid: Player) -> u64 {
        self.boards[pid].settlements | self.boards[pid].cities
    }

    /// VP visible to all players: buildings + longest road + largest army.
    /// Does not include hidden VP dev cards.
    pub fn public_vps(&self, pid: Player) -> u8 {
        let mut vps = self.players[pid].building_vps;
        if let Some((lr_pid, _)) = self.longest_road {
            if lr_pid == pid {
                vps += 2;
            }
        }
        if let Some((la_pid, _)) = self.largest_army {
            if la_pid == pid {
                vps += 2;
            }
        }
        vps
    }

    /// True total VP: buildings + longest road + largest army + VP dev cards.
    pub fn total_vps(&self, pid: Player) -> u8 {
        self.public_vps(pid) + self.players[pid].dev_cards[DevCardKind::VictoryPoint]
    }

    /// Cards not accounted for by any player's known hand or played pile.
    ///
    /// Includes the bank plus any hidden cards. During self-play search,
    /// `determinize()` hides the opponent's cards first, so the pool
    /// reflects the searching player's uncertainty.
    pub fn unknown_dev_pool(&self) -> [u8; 5] {
        let mut pool = ORIGINAL_DEV_DECK;
        for pid in [Player::One, Player::Two] {
            for t in 0..5 {
                let cards = self.players[pid].dev_cards.0[t] as u16;
                let played = self.players[pid].dev_cards_played.0[t] as u16;
                // Saturating: during SO-ISMCTS sims, played counts can exceed
                // the original deck because interior tree nodes may replay dev
                // card actions from a different determinization.
                pool[t] = pool[t].saturating_sub((cards + played).min(255) as u8);
            }
        }
        pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::dev_card::DevCardDeck;
    use crate::game::topology::Topology;

    fn make_state() -> GameState {
        let topo = Arc::new(Topology::from_seed(42));
        let deck = DevCardDeck::new();
        GameState::new(topo, deck, Dice::default())
    }

    /// A fresh game state has full bank, empty hands, all pieces available,
    /// default 4:1 trade ratios, and begins in the PlaceSettlement phase.
    #[test]
    fn initial_state_valid() {
        let state = make_state();
        assert_eq!(state.bank, ResourceArray::new(19, 19, 19, 19, 19));
        assert_eq!(state.phase, Phase::PlaceSettlement);
        assert_eq!(state.turn_number, 0);
        assert_eq!(state.setup_count, 0);
        assert!(state.longest_road.is_none());
        assert!(state.largest_army.is_none());
        assert_eq!(state.occupied_nodes(), 0);
        assert_eq!(state.all_roads(), 0);
        for p in &state.players.0 {
            assert_eq!(p.hand, ResourceArray::default());
            assert_eq!(p.building_vps, 0);
            assert_eq!(p.settlements_left, 5);
            assert_eq!(p.cities_left, 4);
            assert_eq!(p.roads_left, 15);
            assert_eq!(p.trade_ratios, [4; 5]);
        }
    }

    /// PerPlayer indexes Player::One to 0, Player::Two to 1.
    #[test]
    fn per_player_indexing() {
        let pp = PerPlayer([10, 20]);
        assert_eq!(pp[Player::One], 10);
        assert_eq!(pp[Player::Two], 20);
    }
}
