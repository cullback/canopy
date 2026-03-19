use serde::Serialize;
use std::path::Path;

use canopy::game::Game;
use canopy::game_log::GameLog;

use crate::game::{DESTROY_A, DESTROY_B, DONE, K, Phase, TenureGame, optimal_value};

#[derive(Serialize)]
struct ReplayData {
    k: usize,
    initial_value: f32,
    frames: Vec<ReplayFrame>,
    result: String,
}

#[derive(Serialize)]
struct ReplayFrame {
    action: String,
    phase: String,
    board: Vec<u8>,
    partition: Vec<u8>,
    score: u8,
    board_value: f32,
    partition_value: f32,
}

fn format_action(action: usize, state: &TenureGame) -> String {
    let player = match state.phase {
        Phase::Attacker => "Attacker",
        Phase::Defender => "Defender",
    };
    let desc = match action {
        DONE => "Done partitioning".to_string(),
        DESTROY_A => "Destroy board (keep partition)".to_string(),
        DESTROY_B => "Destroy partition (keep board)".to_string(),
        level if level < K => format!("Move piece from level {level}"),
        _ => format!("Unknown action {action}"),
    };
    format!("{player}: {desc}")
}

fn capture_frame(state: &TenureGame, action: &str) -> ReplayFrame {
    ReplayFrame {
        action: action.to_string(),
        phase: match state.phase {
            Phase::Attacker => "Attacker".to_string(),
            Phase::Defender => "Defender".to_string(),
        },
        board: state.board.to_vec(),
        partition: state.partition.to_vec(),
        score: state.score,
        board_value: optimal_value(&state.board),
        partition_value: optimal_value(&state.partition),
    }
}

/// Render a game log into a self-contained HTML replay file.
pub fn render(log: &GameLog, output: &Path) {
    let mut rng = fastrand::Rng::with_seed(log.seed);
    let mut state = TenureGame::random(&mut rng);
    let initial_value = state.initial_value;

    let mut frames = vec![capture_frame(&state, "Game start")];
    for &action in &log.actions {
        let desc = format_action(action, &state);
        state.apply_action(action);
        frames.push(capture_frame(&state, &desc));
    }

    let result = if state.is_terminal() {
        match state.terminal_reward() {
            r if r > 0.0 => "Attacker wins!".to_string(),
            r if r < 0.0 => "Defender wins!".to_string(),
            _ => "Draw!".to_string(),
        }
    } else {
        "Game in progress".to_string()
    };

    let data = ReplayData {
        k: K,
        initial_value,
        frames,
        result,
    };

    let json = serde_json::to_string(&data).expect("failed to serialize replay data");
    let html = include_str!("visualize_template.html").replace("\"__REPLAY_DATA__\"", &json);
    std::fs::write(output, html).expect("failed to write HTML file");
}
