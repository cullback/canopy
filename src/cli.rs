use std::path::PathBuf;

use clap::{Arg, ArgMatches, Command};

use crate::game::Game;
use crate::mcts::Config;
pub use crate::tournament::TournamentOptions;

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
pub fn parse_configs(matches: &ArgMatches) -> [Config; 2] {
    PREFIXES.map(|p| parse_one(matches, p))
}

/// Standard tournament args: `--num-games`, `--log-dir`, plus per-player MCTS config.
pub fn tournament_args() -> Vec<Arg> {
    let mut args = vec![
        Arg::new("num-games")
            .short('n')
            .long("num-games")
            .default_value("20"),
        Arg::new("log-dir")
            .long("log-dir")
            .help("Directory to write game logs"),
        Arg::new("p1-eval")
            .long("p1-eval")
            .default_value("rollout")
            .help("Evaluator for player 1"),
        Arg::new("p2-eval")
            .long("p2-eval")
            .default_value("rollout")
            .help("Evaluator for player 2"),
    ];
    args.extend(config_args());
    args
}

/// Build a complete [`Command`] with all standard tournament args.
pub fn tournament_command(name: &str, about: &str) -> Command {
    let mut cmd = Command::new(name.to_string()).about(about.to_string());
    for arg in tournament_args() {
        cmd = cmd.arg(arg);
    }
    cmd
}

/// Parse standard tournament options from clap [`ArgMatches`].
pub fn parse_tournament(matches: &ArgMatches) -> TournamentOptions {
    TournamentOptions {
        num_games: matches
            .get_one::<String>("num-games")
            .unwrap()
            .parse()
            .unwrap(),
        configs: parse_configs(matches),
        log_dir: matches.get_one::<String>("log-dir").map(PathBuf::from),
        eval_names: [
            matches.get_one::<String>("p1-eval").unwrap().clone(),
            matches.get_one::<String>("p2-eval").unwrap().clone(),
        ],
    }
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
            Arg::new("warmup-iters")
                .long("warmup-iters")
                .default_value(d.warmup_iters.to_string())
                .help("Iterations to ramp MCTS sims and Z→Q blend together"),
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
            Arg::new("bench-baseline")
                .long("bench-baseline")
                .default_value("rollout")
                .help("Evaluator name for benchmark baseline opponent"),
        )
}

// ---------------------------------------------------------------------------
// GameSetup — independent registries for evaluators, encoders, models, configs
// ---------------------------------------------------------------------------

#[cfg(feature = "nn")]
type ModelFactory<G> = Box<
    dyn Fn(
            std::sync::Arc<dyn crate::nn::StateEncoder<G>>,
            &crate::train::Device,
        ) -> Box<dyn crate::train::TrainableModel<G>>
        + Send,
>;

#[cfg(feature = "nn")]
pub struct GameSetup<G: Game> {
    name: String,
    about: String,
    evaluators: crate::eval::Evaluators<G>,
    encoders: Vec<(String, std::sync::Arc<dyn crate::nn::StateEncoder<G>>)>,
    models: Vec<(String, ModelFactory<G>)>,
    configs: Vec<(String, crate::train::TrainConfig)>,
}

#[cfg(feature = "nn")]
impl<G: Game + 'static> GameSetup<G> {
    pub fn new(name: impl Into<String>, about: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            about: about.into(),
            evaluators: crate::eval::Evaluators::new(),
            encoders: Vec::new(),
            models: Vec::new(),
            configs: Vec::new(),
        }
    }

    pub fn add_evaluator(
        &mut self,
        name: impl Into<String>,
        eval: impl crate::eval::Evaluator<G> + Sync + 'static,
    ) {
        self.evaluators.add(name, eval);
    }

    pub fn add_evaluator_arc(
        &mut self,
        name: impl Into<String>,
        eval: std::sync::Arc<dyn crate::eval::Evaluator<G> + Sync>,
    ) {
        self.evaluators.add_arc(name, eval);
    }

    pub fn evaluators(&self) -> &crate::eval::Evaluators<G> {
        &self.evaluators
    }

    pub fn evaluators_mut(&mut self) -> &mut crate::eval::Evaluators<G> {
        &mut self.evaluators
    }

    pub fn add_encoder(
        &mut self,
        name: impl Into<String>,
        encoder: std::sync::Arc<dyn crate::nn::StateEncoder<G>>,
    ) {
        self.encoders.push((name.into(), encoder));
    }

    pub fn add_model(
        &mut self,
        name: impl Into<String>,
        factory: impl Fn(
            std::sync::Arc<dyn crate::nn::StateEncoder<G>>,
            &crate::train::Device,
        ) -> Box<dyn crate::train::TrainableModel<G>>
        + Send
        + 'static,
    ) {
        self.models.push((name.into(), Box::new(factory)));
    }

    pub fn add_config(&mut self, name: impl Into<String>, config: crate::train::TrainConfig) {
        self.configs.push((name.into(), config));
    }

    /// Returns a `train` subcommand with `--model`, `--encoder`, `--config`
    /// args populated from registered names (first registered = default).
    pub fn train_command(&self) -> Command {
        let mut cmd = train_command();

        fn registry_arg(
            cmd: Command,
            id: &'static str,
            help: &'static str,
            names: &[String],
        ) -> Command {
            if names.is_empty() {
                return cmd;
            }
            cmd.arg(
                Arg::new(id)
                    .long(id)
                    .default_value(names[0].clone())
                    .value_parser(names.to_vec())
                    .help(help),
            )
        }

        let model_names: Vec<String> = self.models.iter().map(|(n, _)| n.clone()).collect();
        cmd = registry_arg(cmd, "model", "Model architecture", &model_names);

        let encoder_names: Vec<String> = self.encoders.iter().map(|(n, _)| n.clone()).collect();
        cmd = registry_arg(cmd, "encoder", "State encoder", &encoder_names);

        let config_names: Vec<String> = self.configs.iter().map(|(n, _)| n.clone()).collect();
        cmd = registry_arg(cmd, "config", "Training config preset", &config_names);

        cmd
    }

    /// Returns the full CLI command: tournament args + train subcommand.
    pub fn command(&self) -> Command {
        tournament_command(&self.name, &self.about).subcommand(self.train_command())
    }

    /// Look up the selected `--config`, `--encoder`, `--model`, merge CLI
    /// overrides, build the model, and run training.
    pub fn run_train(
        &self,
        matches: &ArgMatches,
        new_state: impl Fn(&mut fastrand::Rng) -> G + Send + Sync + 'static,
    ) {
        // Look up config
        let config_name = matches
            .get_one::<String>("config")
            .map(|s| s.as_str())
            .unwrap_or_else(|| self.configs[0].0.as_str());
        let base_config = self
            .configs
            .iter()
            .find(|(n, _)| n == config_name)
            .unwrap_or_else(|| {
                let names: Vec<&str> = self.configs.iter().map(|(n, _)| n.as_str()).collect();
                panic!("unknown config '{config_name}', available: {names:?}");
            })
            .1
            .clone();
        let config = parse_train_config(matches, base_config);

        // Look up encoder
        let encoder_name = matches
            .get_one::<String>("encoder")
            .map(|s| s.as_str())
            .unwrap_or_else(|| self.encoders[0].0.as_str());
        let encoder = self
            .encoders
            .iter()
            .find(|(n, _)| n == encoder_name)
            .unwrap_or_else(|| {
                let names: Vec<&str> = self.encoders.iter().map(|(n, _)| n.as_str()).collect();
                panic!("unknown encoder '{encoder_name}', available: {names:?}");
            })
            .1
            .clone();

        // Look up model factory and build
        let model_name = matches
            .get_one::<String>("model")
            .map(|s| s.as_str())
            .unwrap_or_else(|| self.models[0].0.as_str());
        let factory = &self
            .models
            .iter()
            .find(|(n, _)| n == model_name)
            .unwrap_or_else(|| {
                let names: Vec<&str> = self.models.iter().map(|(n, _)| n.as_str()).collect();
                panic!("unknown model '{model_name}', available: {names:?}");
            })
            .1;

        let device = crate::train::default_device();
        let mut trainable = factory(encoder, &device);
        crate::train::run_training(config, trainable.as_mut(), new_state, &self.evaluators);
    }

    /// Parse tournament options and run.
    pub fn run_tournament(&self, matches: &ArgMatches, new_game: impl Fn(u64) -> G) {
        let opts = parse_tournament(matches);
        opts.run(new_game, &self.evaluators);
    }
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
    if set("warmup-iters") {
        config.warmup_iters = val("warmup-iters").parse().unwrap();
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
    if set("bench-baseline") {
        config.bench_baseline_name = val("bench-baseline");
    }

    config
}
