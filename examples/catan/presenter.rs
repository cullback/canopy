use std::path::{Path, PathBuf};

use canopy::player::Player;
use canopy::server::GamePresenter;

use crate::game;
use crate::game::action::ActionId;
use crate::game::dev_card::DevCardKind;
use crate::game::dice::Dice;
use crate::game::resource::ALL_RESOURCES;
use crate::game::state::{GameState, Phase};
use crate::visualize;

const RES_NAMES: [&str; 5] = ["lumber", "brick", "wool", "grain", "ore"];

/// Compute per-player resource gains for a dice roll without mutating state.
fn roll_gains(state: &GameState, roll: u8) -> [[u8; 5]; 2] {
    let topo = &state.topology;
    let mut total_demand = [0u8; 5];
    let mut gains = [[0u8; 5]; 2];

    for &tid in &topo.dice_to_tiles[roll as usize] {
        if tid == state.robber {
            continue;
        }
        let ri = match topo.tiles[tid.0 as usize].terrain.resource() {
            Some(r) => r as usize,
            None => continue,
        };
        let tile_mask = topo.adj.tile_nodes[tid.0 as usize];
        for (pi, &pid) in [Player::One, Player::Two].iter().enumerate() {
            let s = (state.boards[pid].settlements & tile_mask).count_ones() as u8;
            let c = (state.boards[pid].cities & tile_mask).count_ones() as u8;
            let amount = s + c * 2;
            total_demand[ri] += amount;
            gains[pi][ri] += amount;
        }
    }
    // Zero out resources where bank can't cover demand.
    for ri in 0..5 {
        if total_demand[ri] == 0 || state.bank.0[ri] < total_demand[ri] {
            gains[0][ri] = 0;
            gains[1][ri] = 0;
        }
    }
    gains
}

/// Format resource gains for one player, e.g. "2 wool, 1 grain".
fn fmt_gains(gains: &[u8; 5]) -> String {
    let mut parts = Vec::new();
    for (i, &count) in gains.iter().enumerate() {
        if count > 0 {
            parts.push(format!("{} {}", count, RES_NAMES[i]));
        }
    }
    parts.join(", ")
}

/// Compute expected hidden dev card distribution for each player.
///
/// Returns `[[f32; 5]; 2]` — un-normalized expected counts per dev card type.
/// For a player with no hidden cards, the row is all zeros (exact counts are
/// already in `PlayerFrame.dev_cards`).
///
/// Bank estimate is `None` when no hidden cards exist (the unknown pool IS
/// the bank — no uncertainty), so the frontend falls back to exact counts.
fn expected_hidden_dev_cards(state: &GameState) -> ([[f32; 5]; 2], Option<[f32; 5]>) {
    let pool = state.unknown_dev_pool();
    let pool_total: f32 = pool.iter().sum::<u8>() as f32;

    let mut players = [[0.0f32; 5]; 2];
    let total_hidden: u8 =
        state.players[Player::One].hidden_dev_cards + state.players[Player::Two].hidden_dev_cards;

    if total_hidden == 0 {
        return (players, None);
    }

    let bank_cards = (state.dev_deck.total as f32) - (total_hidden as f32);

    for (i, &pid) in [Player::One, Player::Two].iter().enumerate() {
        let hidden = state.players[pid].hidden_dev_cards as f32;
        if hidden > 0.0 && pool_total > 0.0 {
            for t in 0..5 {
                players[i][t] = pool[t] as f32 * hidden / pool_total;
            }
        }
    }

    let mut bank = [0.0f32; 5];
    if bank_cards > 0.0 && pool_total > 0.0 {
        for t in 0..5 {
            bank[t] = pool[t] as f32 * bank_cards / pool_total;
        }
    }

    (players, Some(bank))
}

pub struct CatanPresenter {
    static_dir: PathBuf,
    dice: Dice,
    /// Optional player names: [P1 name, P2 name].
    player_names: Option<[String; 2]>,
}

impl CatanPresenter {
    pub fn new(static_dir: PathBuf, dice: Dice) -> Self {
        Self {
            static_dir,
            dice,
            player_names: None,
        }
    }

    pub fn with_player_names(mut self, names: [String; 2]) -> Self {
        self.player_names = Some(names);
        self
    }
}

impl GamePresenter<GameState> for CatanPresenter {
    fn serialize_state(&self, state: &GameState) -> serde_json::Value {
        let board = visualize::build_board(state);
        let frame = visualize::capture_frame(state, "", state.current_player as u8, None);

        let (expected_dev, expected_bank_dev) = expected_hidden_dev_cards(state);

        // Balanced dice info: normalized probabilities and deck state.
        let dice_info = match &state.dice {
            Dice::Balanced(b) => {
                let ws = b.weights(state.current_player);
                let total: f64 = ws.iter().map(|(_, w)| *w as f64).sum();
                let probs: Vec<f64> = ws
                    .iter()
                    .map(|(_, w)| if total > 0.0 { *w as f64 / total } else { 0.0 })
                    .collect();
                Some(serde_json::json!({
                    "probs": probs,
                    "cards_left": b.cards_left(),
                    "total_cards": 36,
                }))
            }
            Dice::Random => None,
        };

        let mut v = serde_json::json!({
            "board": board,
            "frame": frame,
            "turn": state.turn_number,
            "current_player": state.current_player as u8,
            "p1_vp": state.total_vps(Player::One),
            "p2_vp": state.total_vps(Player::Two),
            "expected_dev": expected_dev,
        });
        if let Some(bank_dev) = expected_bank_dev {
            v["expected_bank_dev"] = serde_json::json!(bank_dev);
        }
        if let Some(dice) = dice_info {
            v["dice"] = dice;
        }
        if let Some(ref names) = self.player_names {
            v["player_names"] = serde_json::json!(names);
        }
        v
    }

    fn action_label(&self, state: &GameState, action: usize) -> String {
        visualize::format_action_desc(ActionId(action as u8), state)
    }

    fn action_description(&self, state: &GameState, action: usize) -> String {
        visualize::format_action_desc(ActionId(action as u8), state)
    }

    fn chance_label(&self, state: &GameState, outcome: usize) -> String {
        match state.phase {
            Phase::Roll => {
                let roll = (outcome + 2) as u8;
                if roll == 7 {
                    return format!("Rolled {roll}");
                }
                let gains = roll_gains(state, roll);
                let p1 = fmt_gains(&gains[0]);
                let p2 = fmt_gains(&gains[1]);
                let mut label = format!("Rolled {roll}");
                if !p1.is_empty() {
                    label.push_str(&format!("\n  P1: {p1}"));
                }
                if !p2.is_empty() {
                    label.push_str(&format!("\n  P2: {p2}"));
                }
                label
            }
            Phase::StealResolve => {
                if let Some(&r) = ALL_RESOURCES.get(outcome) {
                    format!("Stole {r}")
                } else {
                    String::new()
                }
            }
            Phase::DevCardDraw => {
                if let Some(&kind) = DevCardKind::ALL.get(outcome) {
                    format!("Drew {kind:?}")
                } else {
                    String::new()
                }
            }
            _ => String::new(),
        }
    }

    fn phase_label(&self, state: &GameState) -> String {
        visualize::format_phase(&state.phase)
    }

    fn static_dir(&self) -> &Path {
        &self.static_dir
    }

    fn new_game(&self, seed: u64) -> GameState {
        game::new_game(seed, self.dice, 15, 9)
    }
}
