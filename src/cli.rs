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
            Arg::new("concurrent-games")
                .long("concurrent-games")
                .default_value(d.concurrent_games.to_string())
                .help("Maximum async game tasks running concurrently during self-play"),
        )
        .arg(
            Arg::new("simulations")
                .long("simulations")
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
            Arg::new("checkpoint-interval")
                .long("checkpoint-interval")
                .default_value(d.checkpoint_interval.to_string())
                .help("Save model checkpoint every N iterations (last iteration always saved)"),
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
            Arg::new("simulations-start")
                .long("simulations-start")
                .default_value(d.mcts_sims_start.to_string())
                .help("Starting MCTS sims for progressive ramp (ramps linearly to --simulations)"),
        )
        .arg(
            Arg::new("bench-sims")
                .long("bench-sims")
                .default_value(d.bench_sims.to_string())
                .help("MCTS simulations for benchmark opponent"),
        )
        .arg(
            Arg::new("bench-eval")
                .long("bench-eval")
                .default_value("rollout")
                .help("Evaluator name for benchmark opponent"),
        )
        .arg(
            Arg::new("gpus")
                .long("gpus")
                .default_value("1")
                .help("Number of GPUs for parallel inference during self-play"),
        )
}

// ---------------------------------------------------------------------------
// GameCli — independent registries for evaluators, encoders, models, configs
// ---------------------------------------------------------------------------

/// Add a clap arg that selects from a list of registered names.
#[cfg(feature = "nn")]
fn registry_arg(cmd: Command, id: &str, help: &str, names: &[String]) -> Command {
    if names.is_empty() {
        return cmd;
    }
    cmd.arg(
        Arg::new(id.to_string())
            .long(id.to_string())
            .default_value(names[0].clone())
            .value_parser(names.to_vec())
            .help(help.to_string()),
    )
}

#[cfg(feature = "nn")]
type ModelFactory<G> = Box<dyn Fn() -> Box<dyn crate::train::TrainableModel<G>> + Send>;

#[cfg(feature = "nn")]
pub struct GameCli<G: Game> {
    name: String,
    about: String,
    evaluators: crate::eval::Evaluators<G>,
    encoders: Vec<(String, std::sync::Arc<dyn crate::nn::StateEncoder<G>>)>,
    models: Vec<(String, ModelFactory<G>)>,
    configs: Vec<(String, crate::train::TrainConfig)>,
}

#[cfg(feature = "nn")]
impl<G: Game + 'static> GameCli<G> {
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

    pub fn add_model<M>(
        &mut self,
        name: impl Into<String>,
        model_init: impl Fn(&crate::train::Device) -> M + Send + Sync + 'static,
    ) where
        M: burn::module::AutodiffModule<crate::train::TrainBackend>
            + crate::nn::PolicyValueNet<crate::train::TrainBackend>
            + 'static,
        M::InnerModule: crate::nn::PolicyValueNet<crate::train::InferBackend>,
    {
        let init = std::sync::Arc::new(model_init);
        self.models.push((
            name.into(),
            Box::new(move || {
                let init = init.clone();
                Box::new(crate::train::BurnTrainableModel::new(move |d| init(d)))
            }),
        ));
    }

    pub fn add_config(&mut self, name: impl Into<String>, config: crate::train::TrainConfig) {
        self.configs.push((name.into(), config));
    }

    /// Returns a `train` subcommand with `--model`, `--encoder`, `--config`
    /// args populated from registered names (first registered = default).
    pub fn train_command(&self) -> Command {
        let mut cmd = train_command();

        let model_names: Vec<String> = self.models.iter().map(|(n, _)| n.clone()).collect();
        cmd = registry_arg(cmd, "model", "Model architecture", &model_names);

        let encoder_names: Vec<String> = self.encoders.iter().map(|(n, _)| n.clone()).collect();
        cmd = registry_arg(cmd, "encoder", "State encoder", &encoder_names);

        let config_names: Vec<String> = self.configs.iter().map(|(n, _)| n.clone()).collect();
        cmd = registry_arg(cmd, "config", "Training config preset", &config_names);

        cmd
    }

    /// Returns the full CLI command: tournament args + train subcommand.
    ///
    /// When encoders and models are registered, `--nn-model`, `--encoder`,
    /// and `--model` args are added to the root (tournament) command so that
    /// any game can load a trained checkpoint without custom boilerplate.
    pub fn command(&self) -> Command {
        let mut cmd = tournament_command(&self.name, &self.about);

        if !self.encoders.is_empty() && !self.models.is_empty() {
            cmd = cmd.arg(
                Arg::new("nn-model")
                    .long("nn-model")
                    .help("Path to neural network checkpoint (for nn evaluator)"),
            );

            let encoder_names: Vec<String> = self.encoders.iter().map(|(n, _)| n.clone()).collect();
            cmd = registry_arg(
                cmd,
                "encoder",
                "State encoder for nn evaluator",
                &encoder_names,
            );

            let model_names: Vec<String> = self.models.iter().map(|(n, _)| n.clone()).collect();
            cmd = registry_arg(
                cmd,
                "model",
                "Model architecture for nn evaluator",
                &model_names,
            );
        }

        cmd.subcommand(self.train_command())
    }

    /// If `--nn-model` was provided, load the checkpoint and register an
    /// `"nn"` evaluator. No-op if the flag was not given.
    pub fn load_nn_evaluator(&mut self, matches: &ArgMatches) {
        let Some(path) = matches.get_one::<String>("nn-model") else {
            return;
        };

        let checkpoint_path = std::path::Path::new(path.as_str());
        let stem = checkpoint_path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("invalid checkpoint path");
        let iteration = stem
            .strip_prefix("model_iter_")
            .expect("checkpoint filename must be model_iter_N")
            .parse::<usize>()
            .expect("failed to parse iteration from checkpoint filename");
        let checkpoint_dir = checkpoint_path.parent().unwrap();

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

        let mut trainable = factory();
        trainable.load(checkpoint_dir, iteration);
        let evaluator = trainable.evaluator(encoder);
        self.evaluators.add_arc("nn", evaluator);
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

        let mut trainable = factory();
        crate::train::run_training(
            config,
            trainable.as_mut(),
            encoder,
            new_state,
            &self.evaluators,
        );
    }

    /// Dispatch to train or tournament based on subcommand.
    ///
    /// Accepts the train-form closure (`Fn(&mut Rng) -> G`) and adapts it
    /// for tournament by constructing an Rng from the seed.
    pub fn run(
        &mut self,
        matches: &ArgMatches,
        new_game: impl Fn(&mut fastrand::Rng) -> G + Send + Sync + 'static,
    ) {
        if let Some(sub) = matches.subcommand_matches("train") {
            self.run_train(sub, new_game);
            return;
        }
        self.load_nn_evaluator(matches);
        let new_game = std::sync::Arc::new(new_game);
        self.run_tournament(matches, move |seed| {
            let mut rng = fastrand::Rng::with_seed(seed);
            new_game(&mut rng)
        });
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
    if set("concurrent-games") {
        config.concurrent_games = val("concurrent-games").parse().unwrap();
    }
    if set("simulations") {
        config.mcts_sims = val("simulations").parse().unwrap();
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
    if set("checkpoint-interval") {
        config.checkpoint_interval = val("checkpoint-interval").parse().unwrap();
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
    if set("simulations-start") {
        config.mcts_sims_start = val("simulations-start").parse().unwrap();
    }
    if set("bench-sims") {
        config.bench_sims = val("bench-sims").parse().unwrap();
    }
    if set("bench-eval") {
        config.bench_eval = val("bench-eval");
    }
    if set("gpus") {
        config.inference_workers = val("gpus").parse().unwrap();
    }

    config
}
