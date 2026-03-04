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
