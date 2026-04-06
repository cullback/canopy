use std::path::PathBuf;

use clap::{Arg, ArgMatches, Command};

use crate::game::Game;
use crate::mcts::Config;
pub use crate::tournament::TournamentOptions;

const PREFIXES: [&str; 2] = ["p1", "p2"];

fn prefixed_args(prefix: &str) -> Vec<Arg> {
    let d = Config::default();
    vec![
        Arg::new(format!("{prefix}-sims"))
            .long(format!("{prefix}-sims"))
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
        Arg::new(format!("{prefix}-leaf-batch"))
            .long(format!("{prefix}-leaf-batch"))
            .default_value("8"),
    ]
}

fn parse_one(matches: &ArgMatches, prefix: &str) -> Config {
    let get = |name: &str| -> String {
        let key = format!("{prefix}-{name}");
        matches.get_one::<String>(&key).unwrap().clone()
    };
    Config {
        num_simulations: get("sims").parse().unwrap(),
        num_sampled_actions: get("gumbel-m").parse().unwrap(),
        c_visit: get("c-visit").parse().unwrap(),
        c_scale: get("c-scale").parse().unwrap(),
        leaf_batch_size: get("leaf-batch").parse().unwrap(),
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
            .help("Evaluator for player 1 (name or checkpoint path)"),
        Arg::new("p2-eval")
            .long("p2-eval")
            .default_value("rollout")
            .help("Evaluator for player 2 (name or checkpoint path)"),
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

/// Returns a `serve` subcommand for the web analysis board.
#[cfg(feature = "server")]
pub fn serve_command() -> Command {
    Command::new("serve")
        .about("Launch the web analysis board")
        .arg(
            Arg::new("port")
                .long("port")
                .default_value("3000")
                .help("HTTP port"),
        )
        .arg(
            Arg::new("eval")
                .long("eval")
                .default_value("rollout")
                .help("Evaluator (name or checkpoint path)"),
        )
        .arg(
            Arg::new("human")
                .long("human")
                .default_value("none")
                .value_parser(["p1", "p2", "both", "none"])
                .help("Which players are human-controlled"),
        )
        .arg(
            Arg::new("replay")
                .long("replay")
                .help("Game log file to replay (seed + actions)"),
        )
}

/// Parsed serve options.
#[cfg(feature = "server")]
pub struct ServeOptions {
    pub port: u16,
    pub eval_name: String,
    pub human_players: [bool; 2],
    pub replay: Option<PathBuf>,
}

/// Parse serve subcommand options.
#[cfg(feature = "server")]
pub fn parse_serve(matches: &ArgMatches) -> ServeOptions {
    let port: u16 = matches
        .get_one::<String>("port")
        .unwrap()
        .parse()
        .expect("invalid port");
    let eval_name = matches.get_one::<String>("eval").unwrap().clone();
    let human = matches.get_one::<String>("human").unwrap().as_str();
    let human_players = match human {
        "p1" => [true, false],
        "p2" => [false, true],
        "both" => [true, true],
        _ => [false, false],
    };
    let replay = matches.get_one::<String>("replay").map(PathBuf::from);
    ServeOptions {
        port,
        eval_name,
        human_players,
        replay,
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
            Arg::new("train-samples")
                .long("train-samples")
                .default_value(d.train_samples_per_iter.to_string())
                .help("Fresh self-play samples to collect before each training step"),
        )
        .arg(
            Arg::new("replay-buffer-samples")
                .long("replay-buffer-samples")
                .default_value(d.replay_buffer_samples.to_string())
                .help("Maximum samples retained in the replay buffer"),
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
                .help("MCTS simulations per action during self-play"),
        )
        .arg(
            Arg::new("epochs")
                .long("epochs")
                .default_value(d.epochs.to_string())
                .help("Training epochs per iteration"),
        )
        .arg(
            Arg::new("train-batch-size")
                .long("train-batch-size")
                .default_value(d.train_batch_size.to_string())
                .help("Mini-batch size for training gradient steps"),
        )
        .arg(
            Arg::new("inference-batch-size")
                .long("inference-batch-size")
                .default_value(d.inference_batch_size.to_string())
                .help("Maximum evaluations per GPU forward pass during self-play"),
        )
        .arg(Arg::new("lr").long("lr").default_value(d.lr.to_string()))
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
            Arg::new("q-weight-ramp-iters")
                .long("q-weight-ramp-iters")
                .default_value(d.q_weight_ramp_iters.to_string())
                .help("Iterations to ramp Q-weight from 0 to q_weight_max"),
        )
        .arg(
            Arg::new("checkpoint-interval")
                .long("checkpoint-interval")
                .default_value(d.checkpoint_interval.to_string())
                .help("Save model checkpoint every N iterations (last iteration always saved)"),
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
            Arg::new("max-actions")
                .long("max-actions")
                .default_value(d.max_actions.to_string())
                .help("Terminate games exceeding this many actions as a draw (0 = no limit)"),
        )
        .arg(
            Arg::new("explore-actions")
                .long("explore-actions")
                .default_value(d.explore_actions.to_string())
                .help("Early-game actions where action is sampled from improved policy"),
        )
        .arg(
            Arg::new("playout-cap-prob")
                .long("playout-cap-prob")
                .default_value(d.playout_cap_full_prob.to_string())
                .help("Probability of full search per action (playout cap randomization)"),
        )
        .arg(
            Arg::new("playout-cap-fast-sims")
                .long("playout-cap-fast-sims")
                .default_value(d.playout_cap_fast_sims.to_string())
                .help("Simulations for fast (non-full) search actions"),
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
type ModelFactory<G> =
    Box<dyn Fn(&crate::train::TrainConfig) -> Box<dyn crate::train::TrainableModel<G>> + Send>;

#[cfg(feature = "nn")]
pub struct GameCli<G: Game> {
    name: String,
    about: String,
    evaluators: crate::eval::Evaluators<G>,
    encoders: Vec<(String, std::sync::Arc<dyn crate::nn::StateEncoder<G>>)>,
    models: Vec<(String, ModelFactory<G>)>,
    configs: Vec<(String, crate::train::TrainConfig)>,
    /// Encoders for checkpoint-loaded evaluators, keyed by evaluator label.
    checkpoint_encoders: Vec<(String, std::sync::Arc<dyn crate::nn::StateEncoder<G>>)>,
}

#[cfg(feature = "nn")]
impl<G: Game + 'static> GameCli<G> {
    pub fn new(name: impl Into<String>, about: impl Into<String>) -> Self {
        let mut evaluators = crate::eval::Evaluators::new();
        evaluators.add("random", crate::eval::RandomEvaluator);
        evaluators.add("rollout", crate::eval::RolloutEvaluator::default());
        Self {
            name: name.into(),
            about: about.into(),
            evaluators,
            encoders: Vec::new(),
            models: Vec::new(),
            configs: Vec::new(),
            checkpoint_encoders: Vec::new(),
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
        model_init: impl Fn(&crate::train::Device, &crate::train::TrainConfig) -> M
        + Send
        + Sync
        + 'static,
    ) where
        M: burn::module::AutodiffModule<crate::train::TrainBackend>
            + crate::nn::PolicyValueNet<crate::train::TrainBackend>
            + 'static,
        M::InnerModule: crate::nn::PolicyValueNet<crate::train::InferBackend>,
    {
        let init = std::sync::Arc::new(model_init);
        self.models.push((
            name.into(),
            Box::new(move |cfg: &crate::train::TrainConfig| {
                let cfg = cfg.clone();
                let init = init.clone();
                Box::new(crate::train::BurnTrainableModel::new(move |d| {
                    init(d, &cfg)
                }))
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
    /// `--p1-eval` and `--p2-eval` accept either a named evaluator
    /// (random, rollout) or a checkpoint file path. Model and encoder
    /// names are read from checkpoint metadata (falls back to first
    /// registered).
    pub fn command(&self) -> Command {
        let mut cmd = tournament_command(&self.name, &self.about);

        cmd = cmd.subcommand(self.train_command());

        #[cfg(feature = "server")]
        {
            cmd = cmd.subcommand(serve_command());
        }

        cmd
    }

    /// Load a checkpoint file and return the evaluator + encoder pair.
    ///
    /// Reads `CheckpointMeta` for model/encoder names; falls back to the
    /// first registered model/encoder when metadata is absent (old checkpoints).
    pub fn load_checkpoint(
        &self,
        path: &str,
    ) -> (
        std::sync::Arc<dyn crate::eval::Evaluator<G> + Sync>,
        std::sync::Arc<dyn crate::nn::StateEncoder<G>>,
    ) {
        let checkpoint_path = std::path::Path::new(path);
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

        // Try to read metadata for model/encoder names
        let meta_path = checkpoint_dir.join(format!("checkpoint_iter_{iteration}.json"));
        let meta: Option<crate::train::CheckpointMeta> = std::fs::read(&meta_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok());

        // Resolve encoder
        let encoder_name = meta
            .as_ref()
            .and_then(|m| m.encoder.as_deref())
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

        // Resolve model
        let model_name = meta
            .as_ref()
            .and_then(|m| m.model.as_deref())
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

        let config = &self.configs[0].1;
        let mut trainable = factory(config);
        trainable.load_weights(checkpoint_dir, iteration);
        let evaluator = trainable.evaluator(encoder.clone());
        (evaluator, encoder)
    }

    /// Resolve an eval spec: if it looks like a file path (contains `/` or
    /// ends with `.mpk`), load the checkpoint and register the evaluator under
    /// `label`, returning Some(encoder). Otherwise it is a named evaluator and
    /// returns None.
    pub fn resolve_eval_spec(
        &mut self,
        spec: &str,
        label: &str,
    ) -> Option<std::sync::Arc<dyn crate::nn::StateEncoder<G>>> {
        if spec.contains('/') || spec.ends_with(".mpk") {
            let (evaluator, encoder) = self.load_checkpoint(spec);
            self.evaluators.add_arc(label, evaluator);
            self.checkpoint_encoders
                .push((label.to_string(), encoder.clone()));
            Some(encoder)
        } else {
            None
        }
    }

    /// Look up the selected `--config`, `--encoder`, `--model`, merge CLI
    /// overrides, build the model, and run training.
    pub fn run_train(
        &self,
        matches: &ArgMatches,
        new_state: impl Fn(u64) -> G + Send + Sync + 'static,
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
        let mut config = parse_train_config(matches, base_config);
        config.game_name = self.name.clone();

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

        let mut trainable = factory(&config);
        crate::train::run_training(
            config,
            trainable.as_mut(),
            encoder,
            new_state,
            Some(model_name),
            Some(encoder_name),
        );
    }

    /// Dispatch to train or tournament based on subcommand.
    pub fn run(
        &mut self,
        matches: &ArgMatches,
        new_game: impl Fn(u64) -> G + Send + Sync + 'static,
    ) {
        crate::init_logging();
        if let Some(sub) = matches.subcommand_matches("train") {
            self.run_train(sub, new_game);
            return;
        }

        // Resolve --p1-eval and --p2-eval: checkpoint paths get loaded
        // and registered under labels "p1-nn" / "p2-nn".
        let p1_spec = matches.get_one::<String>("p1-eval").unwrap().clone();
        let p2_spec = matches.get_one::<String>("p2-eval").unwrap().clone();
        self.resolve_eval_spec(&p1_spec, "p1-nn");
        self.resolve_eval_spec(&p2_spec, "p2-nn");

        // Remap eval names: checkpoint paths become their registered labels.
        let mut opts = parse_tournament(matches);
        if p1_spec.contains('/') || p1_spec.ends_with(".mpk") {
            opts.eval_names[0] = "p1-nn".to_string();
        }
        if p2_spec.contains('/') || p2_spec.ends_with(".mpk") {
            opts.eval_names[1] = "p2-nn".to_string();
        }

        self.run_tournament_opts(&opts, new_game);
    }

    /// Launch the web analysis board server (serve subcommand).
    #[cfg(feature = "server")]
    pub fn run_serve(
        &mut self,
        serve_matches: &ArgMatches,
        presenter: std::sync::Arc<dyn crate::server::GamePresenter<G>>,
    ) {
        crate::init_logging();
        let opts = parse_serve(serve_matches);

        // Resolve --eval: checkpoint path or named evaluator
        let eval_spec = opts.eval_name.clone();
        self.resolve_eval_spec(&eval_spec, "serve-nn");
        let eval_name = if eval_spec.contains('/') || eval_spec.ends_with(".mpk") {
            "serve-nn"
        } else {
            &opts.eval_name
        };

        let evaluator = self.evaluators.get_arc(eval_name);
        let replay = opts
            .replay
            .map(|path| crate::game_log::GameLog::read(&path));
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(crate::server::serve(
            opts.port,
            evaluator,
            eval_name,
            presenter,
            opts.human_players,
            replay,
        ));
    }

    /// Parse tournament options and run.
    pub fn run_tournament(&self, matches: &ArgMatches, new_game: impl Fn(u64) -> G + Sync) {
        let opts = parse_tournament(matches);
        self.run_tournament_opts(&opts, new_game);
    }

    /// Run tournament with pre-built options.
    fn run_tournament_opts(
        &self,
        opts: &crate::tournament::TournamentOptions,
        new_game: impl Fn(u64) -> G + Sync,
    ) {
        if self.checkpoint_encoders.is_empty() {
            opts.run(new_game, &self.evaluators);
        } else {
            self.run_tournament_batched(opts, new_game);
        }
    }

    /// Look up a checkpoint encoder by evaluator label.
    fn checkpoint_encoder(
        &self,
        label: &str,
    ) -> Option<std::sync::Arc<dyn crate::nn::StateEncoder<G>>> {
        self.checkpoint_encoders
            .iter()
            .find(|(n, _)| n == label)
            .map(|(_, e)| e.clone())
    }

    /// Run tournament with batched nn evaluation for any checkpoint-loaded evaluators.
    fn run_tournament_batched(
        &self,
        opts: &crate::tournament::TournamentOptions,
        new_game: impl Fn(u64) -> G + Sync,
    ) {
        use crate::tournament::BatchedEvaluator;
        use crate::train::inference::InferencePipeline;

        // Set up a pipeline for each distinct checkpoint-loaded evaluator used in the tournament.
        let mut pipelines: Vec<(String, InferencePipeline)> = Vec::new();
        for name in &opts.eval_names {
            if self.checkpoint_encoder(name).is_some() && !pipelines.iter().any(|(n, _)| n == name)
            {
                let eval = self.evaluators.get_arc(name);
                let pipeline = InferencePipeline::start::<G>(vec![eval], 256);
                pipelines.push((name.clone(), pipeline));
            }
        }

        // Wrap checkpoint-loaded players in BatchedEvaluator; leave others direct.
        let make_batched = |name: &str| -> Option<BatchedEvaluator<G>> {
            if let Some(encoder) = self.checkpoint_encoder(name) {
                let eval = self.evaluators.get_arc(name);
                let sender = pipelines
                    .iter()
                    .find(|(n, _)| n == name)
                    .unwrap()
                    .1
                    .sender();
                Some(BatchedEvaluator::new(eval, encoder, sender))
            } else {
                None
            }
        };

        let batched: [Option<BatchedEvaluator<G>>; 2] = [
            make_batched(&opts.eval_names[0]),
            make_batched(&opts.eval_names[1]),
        ];

        let eval_refs: [&(dyn crate::eval::Evaluator<G> + Sync); 2] = [
            batched[0]
                .as_ref()
                .map_or_else(|| self.evaluators.get(&opts.eval_names[0]), |b| b),
            batched[1]
                .as_ref()
                .map_or_else(|| self.evaluators.get(&opts.eval_names[1]), |b| b),
        ];

        let mut rng = fastrand::Rng::new();

        tracing::info!(
            "=== Tournament: {} ({}) vs {} ({}) simulations, {} games (batched) ===",
            opts.eval_names[0],
            opts.configs[0].num_simulations,
            opts.eval_names[1],
            opts.configs[1].num_simulations,
            opts.num_games,
        );

        // Use the first available pipeline's stats for eval counting.
        let eval_counter = pipelines.first().map(|(_, p)| p.stats().evals_counter());
        let logs = crate::tournament::tournament_with_stats(
            new_game,
            &eval_refs,
            &opts.configs,
            opts.num_games,
            &mut rng,
            eval_counter,
        );

        if let Some(dir) = &opts.log_dir {
            crate::tournament::save_game_logs(&logs, dir);
        }

        // Drop batched evaluators (their cloned senders) before joining pipelines.
        drop(batched);
        for (_, p) in pipelines {
            p.join();
        }
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
    if set("train-samples") {
        config.train_samples_per_iter = val("train-samples").parse().unwrap();
    }
    if set("replay-buffer-samples") {
        config.replay_buffer_samples = val("replay-buffer-samples").parse().unwrap();
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
    if set("train-batch-size") {
        config.train_batch_size = val("train-batch-size").parse().unwrap();
    }
    if set("inference-batch-size") {
        config.inference_batch_size = val("inference-batch-size").parse().unwrap();
    }
    if set("lr") {
        config.lr = val("lr").parse().unwrap();
    }
    if set("output") {
        config.output_dir = val("output").into();
    }
    if set("resume") {
        config.resume = matches.get_one::<String>("resume").map(PathBuf::from);
    }
    if set("q-weight-ramp-iters") {
        config.q_weight_ramp_iters = val("q-weight-ramp-iters").parse().unwrap();
    }
    if set("checkpoint-interval") {
        config.checkpoint_interval = val("checkpoint-interval").parse().unwrap();
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
    if set("max-actions") {
        config.max_actions = val("max-actions").parse().unwrap();
    }
    if set("explore-actions") {
        config.explore_actions = val("explore-actions").parse().unwrap();
    }
    if set("playout-cap-prob") {
        config.playout_cap_full_prob = val("playout-cap-prob").parse().unwrap();
    }
    if set("playout-cap-fast-sims") {
        config.playout_cap_fast_sims = val("playout-cap-fast-sims").parse().unwrap();
    }
    if set("gpus") {
        config.inference_workers = val("gpus").parse().unwrap();
    }

    config
}
