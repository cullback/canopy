use clap::{Arg, ArgMatches};

use crate::mcts::Config;
use crate::player::PerPlayer;

const PREFIXES: [&str; 2] = ["p1", "p2"];

fn prefixed_args(prefix: &str) -> Vec<Arg> {
    let d = Config::default();
    vec![
        Arg::new(format!("{prefix}-simulations"))
            .long(format!("{prefix}-simulations"))
            .default_value(d.num_simulations.to_string()),
        Arg::new(format!("{prefix}-cpuct"))
            .long(format!("{prefix}-cpuct"))
            .default_value(d.cpuct.to_string()),
        Arg::new(format!("{prefix}-fpu-reduction"))
            .long(format!("{prefix}-fpu-reduction"))
            .default_value(d.fpu_reduction.to_string()),
        Arg::new(format!("{prefix}-dirichlet-alpha"))
            .long(format!("{prefix}-dirichlet-alpha"))
            .default_value(d.dirichlet_alpha.to_string()),
        Arg::new(format!("{prefix}-dirichlet-epsilon"))
            .long(format!("{prefix}-dirichlet-epsilon"))
            .default_value(d.dirichlet_epsilon.to_string()),
    ]
}

fn parse_one(matches: &ArgMatches, prefix: &str) -> Config {
    let get = |name: &str| -> String {
        let key = format!("{prefix}-{name}");
        matches.get_one::<String>(&key).unwrap().clone()
    };
    Config {
        num_simulations: get("simulations").parse().unwrap(),
        cpuct: get("cpuct").parse().unwrap(),
        fpu_reduction: get("fpu-reduction").parse().unwrap(),
        dirichlet_alpha: get("dirichlet-alpha").parse().unwrap(),
        dirichlet_epsilon: get("dirichlet-epsilon").parse().unwrap(),
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
