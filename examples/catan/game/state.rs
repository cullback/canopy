use std::fmt;
use std::sync::Arc;

use canopy2::player::{PerPlayer, Player};

use super::board::TileId;
use super::dev_card::{DevCardArray, DevCardDeck};
use super::dice::Dice;
use super::resource::ResourceArray;
use super::road::RoadNetwork;
use super::topology::Topology;

#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerBoards {
    pub settlements: u64,          // bits 0..54
    pub cities: u64,               // bits 0..54
    pub road_network: RoadNetwork, // roads + reachable frontier + cached longest
}

#[derive(Clone, Debug)]
pub struct PlayerState {
    pub hand: ResourceArray,
    pub dev_cards: DevCardArray,
    pub dev_cards_bought_this_turn: DevCardArray,
    pub knights_played: u8,
    pub roads_placed: u8,
    pub settlements_left: u8,
    pub cities_left: u8,
    pub roads_left: u8,
    pub has_played_dev_card_this_turn: bool,
    pub victory_points: u8,
    pub hidden_vps: u8,
    pub trade_ratios: [u8; 5], // best maritime ratio per resource
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
            victory_points: 0,
            hidden_vps: 0,
            trade_ratios: [4; 5],
        }
    }
}

impl PlayerState {
    pub fn total_vps(&self) -> u8 {
        self.victory_points + self.hidden_vps
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Phase {
    PlaceSettlement,
    PlaceRoad,
    Roll,
    Discard { player: Player, remaining: u8 },
    MoveRobber,
    Steal,
    StealResolve,
    Main,
    RoadBuilding { roads_left: u8 },
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

    pub setup_count: u8,

    pub longest_road: Option<(Player, u8)>,
    pub largest_army: Option<(Player, u8)>,
    pub dice: Dice,
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
            setup_count: 0,
            longest_road: None,
            largest_army: None,
            dice,
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

    pub fn public_vps(&self, pid: Player) -> u8 {
        let mut vps = self.players[pid].victory_points;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::dev_card::DevCardDeck;
    use crate::game::topology::Topology;

    fn make_state() -> GameState {
        let topo = Arc::new(Topology::from_seed(42));
        let mut rng = fastrand::Rng::with_seed(42);
        let deck = DevCardDeck::new(&mut rng);
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
            assert_eq!(p.victory_points, 0);
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
