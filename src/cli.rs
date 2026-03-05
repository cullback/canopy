use std::path::PathBuf;

use clap::{Arg, ArgMatches, Command};

use crate::mcts::Config;
use crate::player::PerPlayer;

const PREFIXES: [&str; 2] = ["p1", "p2"];

fn prefixed_args(prefix: &str) -> Vec<Arg> {
    let d = Config::default();
    vec![
        Arg::new(format!("{prefix}-simulations"))
            .long(format!("{prefix}-simulations"))
            .default_value(d.num_simulations.to_string()),
        Arg::new(format!("{prefix}-gumbel-m"))
            .long(format!("{prefix}-gumbel-m"))
            .default_value(d.num_sampled_actions.to_string()),
        Arg::new(format!("{prefix}-c-visit"))
            .long(format!("{prefix}-c-visit"))
            .default_value(d.c_visit.to_string()),
        Arg::new(format!("{prefix}-c-scale"))
            .long(format!("{prefix}-c-scale"))
            .default_value(d.c_scale.to_string()),
    ]
}

fn parse_one(matches: &ArgMatches, prefix: &str) -> Config {
    let get = |name: &str| -> String {
        let key = format!("{prefix}-{name}");
        matches.get_one::<String>(&key).unwrap().clone()
    };
    Config {
        num_simulations: get("simulations").parse().unwrap(),
        num_sampled_actions: get("gumbel-m").parse().unwrap(),
        c_visit: get("c-visit").parse().unwrap(),
        c_scale: get("c-scale").parse().unwrap(),
    }
}

/// Returns clap [`Arg`]s for two MCTS configs (`--p1-*` and `--p2-*`).
pub fn config_args() -> Vec<Arg> {
    PREFIXES.iter().flat_map(|p| prefixed_args(p)).collect()
}

/// Parse both MCTS [`Config`]s from clap [`ArgMatches`].
pub fn parse_configs(matches: &ArgMatches) -> PerPlayer<Config> {
    PerPlayer(PREFIXES.map(|p| parse_one(matches, p)))
}

/// Returns a `train` subcommand with all game-agnostic args.
///
/// The caller can append game-specific args via `.arg()`.
#[cfg(feature = "nn")]
pub fn train_command() -> Command {
    Command::new("train")
        .about("Run AlphaZero-style self-play training")
        .arg(
            Arg::new("iterations")
                .long("iterations")
                .default_value("1000"),
        )
        .arg(
            Arg::new("games")
                .long("games")
                .default_value("500")
                .help("Self-play games per iteration"),
        )
        .arg(
            Arg::new("train-mcts")
                .long("train-mcts")
                .default_value("800")
                .help("MCTS simulations per move during self-play"),
        )
        .arg(
            Arg::new("epochs")
                .long("epochs")
                .default_value("3")
                .help("Training epochs per iteration"),
        )
        .arg(
            Arg::new("batch-size")
                .long("batch-size")
                .default_value("256"),
        )
        .arg(Arg::new("lr").long("lr").default_value("0.001"))
        .arg(
            Arg::new("lr-min")
                .long("lr-min")
                .help("Minimum learning rate at end of cosine schedule (default: lr/10)"),
        )
        .arg(
            Arg::new("replay-window")
                .long("replay-window")
                .default_value("40"),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .default_value("checkpoints"),
        )
        .arg(
            Arg::new("resume")
                .long("resume")
                .help("Resume training from checkpoint path (e.g. checkpoints/run/model_iter_10)"),
        )
        .arg(
            Arg::new("q-blend-gen")
                .long("q-blend-gen")
                .default_value("100"),
        )
        .arg(
            Arg::new("bench-games")
                .long("bench-games")
                .default_value("0")
                .help("Benchmark games vs rollout bot per iteration (0 to skip)"),
        )
        .arg(
            Arg::new("gumbel-m")
                .long("gumbel-m")
                .default_value("16")
                .help("Gumbel-Top-k sampled actions at root"),
        )
        .arg(Arg::new("c-visit").long("c-visit").default_value("50.0"))
        .arg(Arg::new("c-scale").long("c-scale").default_value("1.0"))
        .arg(
            Arg::new("explore-moves")
                .long("explore-moves")
                .default_value("30")
                .help("Early-game turns where action is sampled from improved policy"),
        )
        .arg(
            Arg::new("playout-cap-prob")
                .long("playout-cap-prob")
                .default_value("0.25")
                .help("Probability of full search per move (playout cap randomization)"),
        )
        .arg(
            Arg::new("playout-cap-fast-sims")
                .long("playout-cap-fast-sims")
                .default_value("32")
                .help("Simulations for fast (non-full) search moves"),
        )
        .arg(
            Arg::new("bench-baseline-sims")
                .long("bench-baseline-sims")
                .default_value("200")
                .help("MCTS simulations for benchmark baseline opponent"),
        )
        .arg(
            Arg::new("bench-baseline-rollouts")
                .long("bench-baseline-rollouts")
                .default_value("1")
                .help("Rollouts per evaluation for benchmark baseline opponent"),
        )
}

/// Parse all common training fields from clap [`ArgMatches`].
#[cfg(feature = "nn")]
pub fn parse_train_config(matches: &ArgMatches) -> crate::train::TrainConfig {
    let parse = |name: &str| -> String { matches.get_one::<String>(name).unwrap().clone() };

    let lr: f64 = parse("lr").parse().unwrap();

    crate::train::TrainConfig {
        iterations: parse("iterations").parse().unwrap(),
        games_per_iter: parse("games").parse().unwrap(),
        mcts_sims: parse("train-mcts").parse().unwrap(),
        epochs: parse("epochs").parse().unwrap(),
        batch_size: parse("batch-size").parse().unwrap(),
        lr,
        lr_min: matches
            .get_one::<String>("lr-min")
            .map(|s| s.parse().unwrap())
            .unwrap_or(lr / 10.0),
        replay_window: parse("replay-window").parse().unwrap(),
        output_dir: parse("output").into(),
        resume: matches.get_one::<String>("resume").map(PathBuf::from),
        q_blend_generations: parse("q-blend-gen").parse().unwrap(),
        bench_games: parse("bench-games").parse().unwrap(),
        gumbel_m: parse("gumbel-m").parse().unwrap(),
        c_visit: parse("c-visit").parse().unwrap(),
        c_scale: parse("c-scale").parse().unwrap(),
        explore_moves: parse("explore-moves").parse().unwrap(),
        playout_cap_full_prob: parse("playout-cap-prob").parse().unwrap(),
        playout_cap_fast_sims: parse("playout-cap-fast-sims").parse().unwrap(),
        bench_baseline_sims: parse("bench-baseline-sims").parse().unwrap(),
        bench_baseline_rollouts: parse("bench-baseline-rollouts").parse().unwrap(),
    }
}
