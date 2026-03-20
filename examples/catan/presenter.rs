use std::path::{Path, PathBuf};

use canopy::player::Player;
use canopy::server::GamePresenter;

use crate::game;
use crate::game::action::ActionId;
use crate::game::dice::Dice;
use crate::game::resource::ALL_RESOURCES;
use crate::game::state::{GameState, Phase};
use crate::visualize;

pub struct CatanPresenter {
    static_dir: PathBuf,
    dice: Dice,
}

impl CatanPresenter {
    pub fn new(static_dir: PathBuf, dice: Dice) -> Self {
        Self { static_dir, dice }
    }
}

impl GamePresenter<GameState> for CatanPresenter {
    fn serialize_state(&self, state: &GameState) -> serde_json::Value {
        let board = visualize::build_board(state);
        let frame = visualize::capture_frame(state, "", state.current_player as u8, None);

        serde_json::json!({
            "board": board,
            "frame": frame,
            "turn": state.turn_number,
            "current_player": state.current_player as u8,
            "p1_vp": state.total_vps(Player::One),
            "p2_vp": state.total_vps(Player::Two),
        })
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
        game::new_game(seed, self.dice)
    }
}
