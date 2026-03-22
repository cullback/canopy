//! # 2048 — solo evaluation & tournament
//!
//! Single-player stochastic tile-sliding game. Supports MCTS solo play,
//! tournament mode (for framework compatibility), and training.
//!
//! ```text
//! cargo run --example twenty48 -- solo -n 10 --simulations 100 --eval rollout
//! cargo run --example twenty48 -- --p1-sims 200 --p2-sims 200
//! cargo run --example twenty48 -- train --iterations 5 --games 20
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use clap::{Arg, Command};
use indicatif::{ProgressBar, ProgressStyle};

use canopy::cli::GameCli;
use canopy::eval::Evaluation;
use canopy::game::{Game, Status};
use canopy::mcts::{Config, Search, Step};
use canopy::train::TrainConfig;

mod encoder;
mod game;
mod model;

use game::Board;

impl Game for Board {
    const NUM_ACTIONS: usize = game::NUM_ACTIONS;

    fn status(&self) -> Status {
        if self.awaiting_spawn() {
            return Status::Ongoing;
        }
        if game::has_legal_move(self.tiles()) {
            Status::Ongoing
        } else {
            let score = game::board_score(self.tiles()) as f32;
            Status::Terminal((score / 1_000_000.0).min(1.0))
        }
    }

    fn legal_actions(&self, buf: &mut Vec<usize>) {
        if self.awaiting_spawn() {
            return;
        }
        let tiles = self.tiles();
        for dir in 0..4 {
            if game::execute_move(tiles, dir) != tiles {
                buf.push(dir);
            }
        }
    }

    fn apply_action(&mut self, action: usize) {
        if self.awaiting_spawn() {
            // Decode chance outcome: action = pos * 2 + tile_type
            let pos = (action / 2) as u32;
            let val: u8 = if action % 2 == 0 { 1 } else { 2 }; // 1 = "2", 2 = "4"
            let tiles = game::set_nibble(self.tiles(), pos, val);
            self.0 = tiles; // Decision: store directly
        } else {
            let new_tiles = game::execute_move(self.tiles(), action);
            self.0 = !new_tiles; // Chance: store complement
        }
    }

    fn chance_outcomes(&self, buf: &mut Vec<(usize, u32)>) {
        if !self.awaiting_spawn() {
            return;
        }
        for pos in game::empty_positions(self.tiles()) {
            buf.push(((pos as usize) * 2, 9)); // 90% chance of "2"
            buf.push(((pos as usize) * 2 + 1, 1)); // 10% chance of "4"
        }
    }

    fn sample_chance(&self, rng: &mut fastrand::Rng) -> Option<usize> {
        if !self.awaiting_spawn() {
            return None;
        }
        let empties = game::empty_positions(self.tiles());
        let pos = empties[rng.usize(..empties.len())];
        let tile_type = if rng.u32(0..10) == 0 { 1 } else { 0 }; // 0 = "2", 1 = "4"
        Some((pos as usize) * 2 + tile_type)
    }

    fn state_key(&self) -> Option<u64> {
        if self.awaiting_spawn() {
            None
        } else {
            Some(self.0)
        }
    }
}

fn solo_command() -> Command {
    let d = Config::default();
    Command::new("solo")
        .about("Play N solo games and report score statistics")
        .arg(
            Arg::new("num-games")
                .short('n')
                .long("num-games")
                .default_value("10"),
        )
        .arg(
            Arg::new("simulations")
                .long("simulations")
                .default_value(d.num_simulations.to_string()),
        )
        .arg(
            Arg::new("eval")
                .long("eval")
                .default_value("rollout")
                .help("Evaluator to use"),
        )
        .arg(
            Arg::new("gumbel-m")
                .long("gumbel-m")
                .default_value(d.num_sampled_actions.to_string()),
        )
}

fn run_solo(matches: &clap::ArgMatches, setup: &GameCli<Board>) {
    let num_games: u32 = matches
        .get_one::<String>("num-games")
        .unwrap()
        .parse()
        .unwrap();
    let simulations: u32 = matches
        .get_one::<String>("simulations")
        .unwrap()
        .parse()
        .unwrap();
    let eval_name = matches.get_one::<String>("eval").unwrap();
    let gumbel_m: u32 = matches
        .get_one::<String>("gumbel-m")
        .unwrap()
        .parse()
        .unwrap();

    let evaluator = setup.evaluators().get(eval_name);

    let config = Config {
        num_simulations: simulations,
        num_sampled_actions: gumbel_m,
        gumbel_scale: 0.0, // No exploration noise during evaluation
        ..Default::default()
    };

    println!("=== Solo: {eval_name}, {simulations} simulations, {num_games} games ===\n");

    let mut rng = fastrand::Rng::new();
    let mut scores = Vec::with_capacity(num_games as usize);
    let mut max_tiles = Vec::with_capacity(num_games as usize);

    let pb = ProgressBar::new(num_games as u64);
    pb.set_style(
        ProgressStyle::with_template("{bar:30} {pos}/{len} | avg {msg} | {elapsed} < {eta}")
            .unwrap(),
    );
    pb.set_message("...");

    for _ in 0..num_games {
        let mut state = Board::new(&mut rng);

        loop {
            match state.status() {
                Status::Terminal(_) => break,
                Status::Ongoing => {}
            }

            if let Some(action) = state.sample_chance(&mut rng) {
                state.apply_action(action);
            } else {
                let mut search = Search::new(state, config.clone());
                let mut evals: Vec<Evaluation> = vec![];
                let result = loop {
                    match search.step(&evals, &mut rng) {
                        Step::NeedsEval(states) => {
                            let refs: Vec<&Board> = states.iter().collect();
                            evals = evaluator.evaluate_batch(&refs, &mut rng);
                        }
                        Step::Done(r) => break r,
                    }
                };
                state.apply_action(result.selected_action);
            }
        }

        let tiles = state.tiles();
        let score = game::board_score(tiles);
        let mt = game::max_tile(tiles);
        scores.push(score);
        max_tiles.push(mt);

        let avg = scores.iter().sum::<u32>() as f32 / scores.len() as f32;
        pb.set_message(format!("{avg:.0}"));
        pb.inc(1);
    }

    pb.finish();
    println!();

    // Summary statistics
    let avg = scores.iter().sum::<u32>() as f32 / scores.len() as f32;
    let max = *scores.iter().max().unwrap();
    let min = *scores.iter().min().unwrap();

    println!("Score:  avg {avg:.0}  |  min {min}  |  max {max}");

    // Max-tile distribution
    let mut tile_counts: HashMap<u32, u32> = HashMap::new();
    for &mt in &max_tiles {
        *tile_counts.entry(mt).or_default() += 1;
    }
    let mut tiles: Vec<u32> = tile_counts.keys().copied().collect();
    tiles.sort();
    println!("Max tile distribution:");
    for tile in tiles {
        let count = tile_counts[&tile];
        let pct = count as f32 / num_games as f32 * 100.0;
        println!("  {tile:>5}: {count:>3} ({pct:.1}%)");
    }
}

fn main() {
    let mut setup = GameCli::new("twenty48", "2048 tile-sliding game");
    setup.add_encoder("default", Arc::new(encoder::Twenty48Encoder));
    setup.add_model("default", |device, _cfg| model::init_twenty48(device));
    setup.add_config(
        "default",
        TrainConfig {
            iterations: 200,
            games_per_iter: 100,
            mcts_sims: 200,
            mcts_sims_start: 50,
            epochs: 3,
            train_batch_size: 128,
            replay_window: 10,
            warmup_iters: 20,
            concurrent_games: 10,
            leaf_batch_size: 1,
            explore_moves: 0,
            ..TrainConfig::default()
        },
    );

    let matches = setup.command().subcommand(solo_command()).get_matches();

    if let Some(sub) = matches.subcommand_matches("solo") {
        run_solo(sub, &setup);
        return;
    }

    setup.run(&matches, |seed| {
        Board::new(&mut fastrand::Rng::with_seed(seed))
    });
}
