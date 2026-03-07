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
        ..Default::default()
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
    use crate::train::TrainConfig;
    let d = TrainConfig::default();

    Command::new("train")
        .about("Run AlphaZero-style self-play training")
        .arg(
            Arg::new("iterations")
                .long("iterations")
                .default_value(d.iterations.to_string()),
        )
        .arg(
            Arg::new("games")
                .long("games")
                .default_value(d.games_per_iter.to_string())
                .help("Self-play games per iteration"),
        )
        .arg(
            Arg::new("train-mcts")
                .long("train-mcts")
                .default_value(d.mcts_sims.to_string())
                .help("MCTS simulations per move during self-play"),
        )
        .arg(
            Arg::new("epochs")
                .long("epochs")
                .default_value(d.epochs.to_string())
                .help("Training epochs per iteration"),
        )
        .arg(
            Arg::new("batch-size")
                .long("batch-size")
                .default_value(d.batch_size.to_string()),
        )
        .arg(Arg::new("lr").long("lr").default_value(d.lr.to_string()))
        .arg(
            Arg::new("lr-min")
                .long("lr-min")
                .help("Minimum learning rate at end of cosine schedule (default: lr/10)"),
        )
        .arg(
            Arg::new("replay-window")
                .long("replay-window")
                .default_value(d.replay_window.to_string()),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .default_value(d.output_dir.to_string_lossy().into_owned()),
        )
        .arg(
            Arg::new("resume")
                .long("resume")
                .help("Resume training from checkpoint path (e.g. checkpoints/run/model_iter_10)"),
        )
        .arg(
            Arg::new("q-blend-gen")
                .long("q-blend-gen")
                .default_value(d.q_blend_generations.to_string()),
        )
        .arg(
            Arg::new("bench-games")
                .long("bench-games")
                .default_value(d.bench_games.to_string())
                .help("Benchmark games vs rollout bot (0 to skip)"),
        )
        .arg(
            Arg::new("bench-interval")
                .long("bench-interval")
                .default_value(d.bench_interval.to_string())
                .help("Run benchmark every N iterations"),
        )
        .arg(
            Arg::new("gumbel-m")
                .long("gumbel-m")
                .default_value(d.gumbel_m.to_string())
                .help("Gumbel-Top-k sampled actions at root"),
        )
        .arg(
            Arg::new("c-visit")
                .long("c-visit")
                .default_value(d.c_visit.to_string()),
        )
        .arg(
            Arg::new("c-scale")
                .long("c-scale")
                .default_value(d.c_scale.to_string()),
        )
        .arg(
            Arg::new("leaf-batch-size")
                .long("leaf-batch-size")
                .default_value(d.leaf_batch_size.to_string())
                .help("Leaves to collect per MCTS batch before requesting evaluation"),
        )
        .arg(
            Arg::new("explore-moves")
                .long("explore-moves")
                .default_value(d.explore_moves.to_string())
                .help("Early-game turns where action is sampled from improved policy"),
        )
        .arg(
            Arg::new("playout-cap-prob")
                .long("playout-cap-prob")
                .default_value(d.playout_cap_full_prob.to_string())
                .help("Probability of full search per move (playout cap randomization)"),
        )
        .arg(
            Arg::new("playout-cap-fast-sims")
                .long("playout-cap-fast-sims")
                .default_value(d.playout_cap_fast_sims.to_string())
                .help("Simulations for fast (non-full) search moves"),
        )
        .arg(
            Arg::new("mcts-sims-start")
                .long("mcts-sims-start")
                .default_value(d.mcts_sims_start.to_string())
                .help("Starting MCTS sims for progressive ramp (ramps linearly to --train-mcts)"),
        )
        .arg(
            Arg::new("bench-baseline-sims")
                .long("bench-baseline-sims")
                .default_value(d.bench_baseline_sims.to_string())
                .help("MCTS simulations for benchmark baseline opponent"),
        )
        .arg(
            Arg::new("bench-baseline-rollouts")
                .long("bench-baseline-rollouts")
                .default_value(d.bench_baseline_rollouts.to_string())
                .help("Rollouts per evaluation for benchmark baseline opponent"),
        )
}

/// Parse CLI overrides on top of a base [`TrainConfig`].
///
/// Only fields explicitly passed on the command line are overridden;
/// everything else keeps the value from `config`.
#[cfg(feature = "nn")]
pub fn parse_train_config(
    matches: &ArgMatches,
    mut config: crate::train::TrainConfig,
) -> crate::train::TrainConfig {
    use clap::parser::ValueSource;

    let set = |name: &str| matches.value_source(name) == Some(ValueSource::CommandLine);
    let val = |name: &str| -> String { matches.get_one::<String>(name).unwrap().clone() };

    if set("iterations") {
        config.iterations = val("iterations").parse().unwrap();
    }
    if set("games") {
        config.games_per_iter = val("games").parse().unwrap();
    }
    if set("train-mcts") {
        config.mcts_sims = val("train-mcts").parse().unwrap();
    }
    if set("epochs") {
        config.epochs = val("epochs").parse().unwrap();
    }
    if set("batch-size") {
        config.batch_size = val("batch-size").parse().unwrap();
    }
    if set("lr") {
        config.lr = val("lr").parse().unwrap();
        // Recalculate lr_min unless it was also explicitly set
        if !set("lr-min") {
            config.lr_min = config.lr / 10.0;
        }
    }
    if set("lr-min") {
        config.lr_min = val("lr-min").parse().unwrap();
    }
    if set("replay-window") {
        config.replay_window = val("replay-window").parse().unwrap();
    }
    if set("output") {
        config.output_dir = val("output").into();
    }
    if set("resume") {
        config.resume = matches.get_one::<String>("resume").map(PathBuf::from);
    }
    if set("q-blend-gen") {
        config.q_blend_generations = val("q-blend-gen").parse().unwrap();
    }
    if set("bench-games") {
        config.bench_games = val("bench-games").parse().unwrap();
    }
    if set("bench-interval") {
        config.bench_interval = val("bench-interval").parse().unwrap();
    }
    if set("gumbel-m") {
        config.gumbel_m = val("gumbel-m").parse().unwrap();
    }
    if set("c-visit") {
        config.c_visit = val("c-visit").parse().unwrap();
    }
    if set("c-scale") {
        config.c_scale = val("c-scale").parse().unwrap();
    }
    if set("leaf-batch-size") {
        config.leaf_batch_size = val("leaf-batch-size").parse().unwrap();
    }
    if set("explore-moves") {
        config.explore_moves = val("explore-moves").parse().unwrap();
    }
    if set("playout-cap-prob") {
        config.playout_cap_full_prob = val("playout-cap-prob").parse().unwrap();
    }
    if set("playout-cap-fast-sims") {
        config.playout_cap_fast_sims = val("playout-cap-fast-sims").parse().unwrap();
    }
    if set("mcts-sims-start") {
        config.mcts_sims_start = val("mcts-sims-start").parse().unwrap();
    }
    if set("bench-baseline-sims") {
        config.bench_baseline_sims = val("bench-baseline-sims").parse().unwrap();
    }
    if set("bench-baseline-rollouts") {
        config.bench_baseline_rollouts = val("bench-baseline-rollouts").parse().unwrap();
    }

    config
}
