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

/// Compute expected hidden dev card distribution for each player.
///
/// Returns `[[f32; 5]; 2]` — un-normalized expected counts per dev card type.
/// For a player with no hidden cards, the row is all zeros (exact counts are
/// already in `PlayerFrame.dev_cards`).
fn expected_hidden_dev_cards(state: &GameState) -> [[f32; 5]; 2] {
    let pool = state.unknown_dev_pool();
    let pool_total: f32 = pool.iter().sum::<u8>() as f32;

    let mut result = [[0.0f32; 5]; 2];
    for (i, &pid) in [Player::One, Player::Two].iter().enumerate() {
        let hidden = state.players[pid].hidden_dev_cards as f32;
        if hidden > 0.0 && pool_total > 0.0 {
            for t in 0..5 {
                result[i][t] = pool[t] as f32 * hidden / pool_total;
            }
        }
    }
    result
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

        let expected_dev = expected_hidden_dev_cards(state);

        let mut v = serde_json::json!({
            "board": board,
            "frame": frame,
            "turn": state.turn_number,
            "current_player": state.current_player as u8,
            "p1_vp": state.total_vps(Player::One),
            "p2_vp": state.total_vps(Player::Two),
            "expected_dev": expected_dev,
        });
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
            Phase::Roll => format!("Rolled {}", outcome + 2),
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
