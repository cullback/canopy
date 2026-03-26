use std::sync::Arc;

use crate::eval::{Evaluation, Evaluator};
use crate::game::{Game, Status};
use crate::game_log::GameLog;
use crate::mcts::{Config, Search, SearchResult, SearchSnapshot, Step};

use super::protocol::{ActionInfo, ClientMsg, ServerMsg};
use super::traits::GamePresenter;

/// Per-player configuration.
struct PlayerConfig {
    simulations: u32,
    /// Whether this player is human-controlled.
    human: bool,
}

/// A checkpoint in the game timeline (player decision or chance outcome).
struct HistoryEntry<G> {
    state: G,        // state before this action
    action: usize,   // action taken from this state
    label: String,   // human-readable label (empty for unlabeled)
    is_chance: bool, // true for auto-resolved chance outcomes
    /// Pre-computed next state for timeline entries where the action can't be
    /// replayed through `apply_action` (e.g. colonist replay snapshots).
    next_state: Option<G>,
}

/// Owns the game state, search tree, evaluator, and presenter.
/// Processes client messages and produces server responses.
pub struct GameSession<G: Game> {
    search: Search<G>,
    evaluator: Arc<dyn Evaluator<G> + Sync>,
    presenter: Arc<dyn GamePresenter<G>>,
    rng: fastrand::Rng,
    configs: [PlayerConfig; 2],
    history: Vec<HistoryEntry<G>>,
    cursor: usize, // 0..=history.len(), points past the last applied entry
    seed: u64,
}

impl<G: Game + 'static> GameSession<G> {
    pub fn new(
        evaluator: Arc<dyn Evaluator<G> + Sync>,
        presenter: Arc<dyn GamePresenter<G>>,
        default_sims: u32,
        human_players: [bool; 2],
    ) -> Self {
        let seed = fastrand::u64(..);
        let state = presenter.new_game(seed);
        let config = Config {
            num_simulations: default_sims,
            ..Config::default()
        };
        let search = Search::new(state, config);

        Self {
            search,
            evaluator,
            presenter,
            rng: fastrand::Rng::new(),
            configs: [
                PlayerConfig {
                    simulations: default_sims,
                    human: human_players[0],
                },
                PlayerConfig {
                    simulations: default_sims,
                    human: human_players[1],
                },
            ],
            history: Vec::new(),
            cursor: 0,
            seed,
        }
    }

    /// Create a session with an explicit initial state.
    pub fn with_state(
        state: G,
        evaluator: Arc<dyn Evaluator<G> + Sync>,
        presenter: Arc<dyn GamePresenter<G>>,
        default_sims: u32,
        human_players: [bool; 2],
    ) -> Self {
        let config = Config {
            num_simulations: default_sims,
            ..Config::default()
        };
        let search = Search::new(state, config);

        Self {
            search,
            evaluator,
            presenter,
            rng: fastrand::Rng::new(),
            configs: [
                PlayerConfig {
                    simulations: default_sims,
                    human: human_players[0],
                },
                PlayerConfig {
                    simulations: default_sims,
                    human: human_players[1],
                },
            ],
            history: Vec::new(),
            cursor: 0,
            seed: 0,
        }
    }

    /// Load a recorded game log for replay.
    ///
    /// Resets the game with the log's seed, replays all actions into history
    /// (recording labels and chance detection), then rewinds cursor to 0 so
    /// the UI starts at the beginning with full redo available.
    pub fn load_replay(&mut self, log: &GameLog) {
        let state = self.presenter.new_game(log.seed);
        self.search.reset(state);
        self.history.clear();
        self.cursor = 0;
        self.seed = log.seed;

        // Replay all actions into history.
        for &action in &log.actions {
            self.apply_action(action);
        }

        // Rewind: reset to initial state, keep history for redo.
        self.cursor = 0;
        let state = self.presenter.new_game(log.seed);
        self.search.reset(state);
    }

    /// Load an externally-built timeline (e.g. from colonist.io replay).
    ///
    /// Each entry is `(label, state)`. The first entry is the initial state;
    /// subsequent entries are states after events described by their label.
    /// Redo navigates forward using `next_state` (no `apply_action`).
    pub fn load_timeline(&mut self, timeline: Vec<(String, G)>) {
        if timeline.is_empty() {
            return;
        }
        self.search.reset(timeline[0].1.clone());
        self.history.clear();
        self.cursor = 0;

        for i in 0..timeline.len() - 1 {
            self.history.push(HistoryEntry {
                state: timeline[i].1.clone(),
                action: usize::MAX, // sentinel — not a real game action
                label: timeline[i + 1].0.clone(),
                is_chance: false,
                next_state: Some(timeline[i + 1].1.clone()),
            });
        }
    }

    /// Process a client message and return response messages.
    pub fn handle(&mut self, msg: ClientMsg) -> Vec<ServerMsg> {
        match msg {
            ClientMsg::NewGame { seed } => {
                self.seed = seed.unwrap_or_else(|| fastrand::u64(..));
                let state = self.presenter.new_game(self.seed);
                self.search.reset(state);
                self.history.clear();
                self.cursor = 0;
                self.auto_resolve_chance();
                vec![self.state_msg()]
            }
            ClientMsg::GetState => {
                vec![self.state_msg()]
            }
            ClientMsg::PlayAction { action } => {
                if self.is_terminal() {
                    return vec![ServerMsg::Error {
                        message: "Game is over".into(),
                    }];
                }
                let legal = self.legal_actions();
                if !legal.contains(&action) {
                    return vec![ServerMsg::Error {
                        message: format!("Illegal action: {action}"),
                    }];
                }
                self.apply_action(action);
                self.auto_resolve_chance();
                vec![self.state_msg()]
            }
            ClientMsg::BotMove { simulations } => {
                if self.is_terminal() {
                    return vec![ServerMsg::Error {
                        message: "Game is over".into(),
                    }];
                }
                if self.is_chance() {
                    return vec![ServerMsg::Error {
                        message: "Current state is a chance node".into(),
                    }];
                }
                let player_idx = self.current_player_idx();
                let sims = simulations.unwrap_or(self.configs[player_idx].simulations);
                self.search.set_num_simulations(sims);

                let result = self.run_search();
                let action = result.selected_action;
                let label = self.presenter.action_label(self.search.state(), action);
                let snapshot = self.search.snapshot();
                let action_labels = snapshot
                    .as_ref()
                    .map(|s| self.edge_labels(&s.edges))
                    .unwrap_or_default();
                self.apply_action(action);
                self.auto_resolve_chance();
                vec![
                    ServerMsg::BotAction {
                        action,
                        label,
                        snapshot,
                        action_labels,
                    },
                    self.state_msg(),
                ]
            }
            ClientMsg::RunSims { count } => {
                if self.is_terminal() || self.is_chance() {
                    return vec![ServerMsg::Error {
                        message: "Cannot run sims on chance/terminal state".into(),
                    }];
                }
                self.search.set_num_simulations(count);
                let _result = self.run_search();
                let snapshot = self.search.snapshot();
                match snapshot {
                    Some(snap) => {
                        let labels = self.edge_labels(&snap.edges);
                        vec![ServerMsg::Snapshot {
                            snapshot: snap,
                            action_labels: labels,
                        }]
                    }
                    None => vec![ServerMsg::Error {
                        message: "No snapshot available".into(),
                    }],
                }
            }
            ClientMsg::GetSnapshot => match self.search.snapshot() {
                Some(snap) => {
                    let labels = self.edge_labels(&snap.edges);
                    vec![ServerMsg::Snapshot {
                        snapshot: snap,
                        action_labels: labels,
                    }]
                }
                None => vec![ServerMsg::Error {
                    message: "No snapshot available".into(),
                }],
            },
            ClientMsg::ExploreSubtree { action_path, depth } => {
                match self.search.snapshot_at_path(&action_path, depth) {
                    Some(mut tree) => {
                        self.label_subtree(&mut tree);
                        vec![ServerMsg::Subtree { tree }]
                    }
                    None => vec![ServerMsg::Error {
                        message: "Path not found in tree".into(),
                    }],
                }
            }
            ClientMsg::TakeOver { player } => {
                if let Some(cfg) = self.configs.get_mut(player as usize) {
                    cfg.human = true;
                    vec![self.state_msg()]
                } else {
                    vec![ServerMsg::Error {
                        message: format!("Invalid player: {player}"),
                    }]
                }
            }
            ClientMsg::ReleaseControl { player } => {
                if let Some(cfg) = self.configs.get_mut(player as usize) {
                    cfg.human = false;
                    vec![self.state_msg()]
                } else {
                    vec![ServerMsg::Error {
                        message: format!("Invalid player: {player}"),
                    }]
                }
            }
            ClientMsg::SetAutoplay { .. } => {
                // Autoplay is handled client-side by sending BotMove in a loop.
                vec![self.state_msg()]
            }
            ClientMsg::Undo => {
                if self.cursor > 0 {
                    // Skip back over chance entries to the previous decision point.
                    loop {
                        self.cursor -= 1;
                        if self.cursor == 0 || !self.history[self.cursor].is_chance {
                            break;
                        }
                    }
                    self.search.reset(self.history[self.cursor].state.clone());
                    vec![self.state_msg()]
                } else {
                    vec![ServerMsg::Error {
                        message: "Nothing to undo".into(),
                    }]
                }
            }
            ClientMsg::Redo => {
                if self.cursor < self.history.len() {
                    // Replay the stored decision + any following chance outcomes.
                    loop {
                        let entry = &self.history[self.cursor];
                        if let Some(ref next) = entry.next_state {
                            self.search.reset(next.clone());
                        } else {
                            self.search.apply_action(entry.action);
                        }
                        self.cursor += 1;
                        let at_chance =
                            self.cursor < self.history.len() && self.history[self.cursor].is_chance;
                        if !at_chance {
                            break;
                        }
                    }
                    vec![self.state_msg()]
                } else {
                    vec![ServerMsg::Error {
                        message: "Nothing to redo".into(),
                    }]
                }
            }
            ClientMsg::SetConfig {
                player,
                simulations,
            } => {
                if let Some(cfg) = self.configs.get_mut(player as usize) {
                    cfg.simulations = simulations;
                    vec![self.state_msg()]
                } else {
                    vec![ServerMsg::Error {
                        message: format!("Invalid player: {player}"),
                    }]
                }
            }
        }
    }

    /// Build a GameState server message for the current state.
    fn state_msg(&self) -> ServerMsg {
        let state = self.search.state();
        let is_terminal = matches!(state.status(), Status::Terminal(_));
        let is_chance = self.is_chance();

        let legal = if is_terminal || is_chance {
            Vec::new()
        } else {
            let actions = self.legal_actions();
            actions
                .iter()
                .map(|&a| ActionInfo {
                    action: a,
                    label: self.presenter.action_label(state, a),
                })
                .collect()
        };

        let result = if let Status::Terminal(reward) = state.status() {
            Some(if reward > 0.0 {
                "P1 wins".into()
            } else if reward < 0.0 {
                "P2 wins".into()
            } else {
                "Draw".into()
            })
        } else {
            None
        };

        let action_log: Vec<String> = self.history[..self.cursor]
            .iter()
            .filter(|e| !e.label.is_empty())
            .map(|e| e.label.clone())
            .collect();

        ServerMsg::GameState {
            state: self.presenter.serialize_state(state),
            legal_actions: legal,
            current_player: self.current_player_idx() as u8,
            phase: self.presenter.phase_label(state),
            is_chance,
            is_terminal,
            result,
            action_log,
            can_undo: self.cursor > 0,
            can_redo: self.cursor < self.history.len(),
        }
    }

    fn legal_actions(&self) -> Vec<usize> {
        let mut buf = Vec::new();
        self.search.state().legal_actions(&mut buf);
        buf
    }

    fn is_terminal(&self) -> bool {
        matches!(self.search.state().status(), Status::Terminal(_))
    }

    fn is_chance(&self) -> bool {
        let mut buf = Vec::new();
        self.search.state().chance_outcomes(&mut buf);
        !buf.is_empty()
    }

    fn current_player_idx(&self) -> usize {
        let sign = self.search.state().current_sign();
        if sign > 0.0 { 0 } else { 1 }
    }

    /// Returns true if current player is a human.
    pub fn current_is_human(&self) -> bool {
        let idx = self.current_player_idx();
        self.configs[idx].human
    }

    fn apply_action(&mut self, action: usize) {
        let state = self.search.state();
        let mut chances = Vec::new();
        state.chance_outcomes(&mut chances);
        let is_chance = !chances.is_empty();
        let label = if !is_chance {
            let player = if state.current_sign() > 0.0 {
                "P1"
            } else {
                "P2"
            };
            let action_label = self.presenter.action_label(state, action);
            format!("{player}: {action_label}")
        } else {
            self.presenter.chance_label(state, action)
        };
        // If the next history entry matches this action, preserve redo history.
        if self.cursor < self.history.len() && self.history[self.cursor].action == action {
            self.cursor += 1;
            self.search.apply_action(action);
            return;
        }
        // Truncate redo future and push new entry.
        self.history.truncate(self.cursor);
        self.history.push(HistoryEntry {
            state: state.clone(),
            action,
            label,
            is_chance,
            next_state: None,
        });
        self.cursor += 1;
        self.search.apply_action(action);
    }

    /// Auto-resolve chance nodes (dice, steals) until we reach a decision
    /// point or terminal state.
    fn auto_resolve_chance(&mut self) {
        loop {
            if self.is_terminal() {
                break;
            }
            // If the log is still intact and the next entry is a chance action,
            // replay it instead of re-sampling to avoid state divergence.
            if self.cursor < self.history.len() && self.history[self.cursor].is_chance {
                let a = self.history[self.cursor].action;
                self.apply_action(a);
                continue;
            }
            let action = self.search.state().sample_chance(&mut self.rng);
            match action {
                Some(a) => self.apply_action(a),
                None => break,
            }
        }
    }

    /// Run MCTS to completion and return the result.
    fn run_search(&mut self) -> crate::mcts::SearchResult {
        let mut evals = vec![];
        loop {
            match self.search.step(&evals, &mut self.rng) {
                Step::NeedsEval(states) => {
                    let refs: Vec<&G> = states.iter().collect();
                    evals = self.evaluator.evaluate_batch(&refs, &mut self.rng);
                }
                Step::Done(result) => return result,
            }
        }
    }

    /// Validate state and prepare search for streaming.
    /// Returns `Ok(sims_total)` or `Err(error_msgs)`.
    pub fn begin_search(&mut self, msg: &ClientMsg) -> Result<u32, Vec<ServerMsg>> {
        match msg {
            ClientMsg::BotMove { simulations } => {
                if self.is_terminal() {
                    return Err(vec![ServerMsg::Error {
                        message: "Game is over".into(),
                    }]);
                }
                if self.is_chance() {
                    return Err(vec![ServerMsg::Error {
                        message: "Current state is a chance node".into(),
                    }]);
                }
                let player_idx = self.current_player_idx();
                let sims = simulations.unwrap_or(self.configs[player_idx].simulations);
                self.search.set_num_simulations(sims);
                Ok(sims)
            }
            ClientMsg::RunSims { count } => {
                if self.is_terminal() || self.is_chance() {
                    return Err(vec![ServerMsg::Error {
                        message: "Cannot run sims on chance/terminal state".into(),
                    }]);
                }
                self.search.set_num_simulations(*count);
                Ok(*count)
            }
            _ => Err(vec![ServerMsg::Error {
                message: "begin_search called with non-search message".into(),
            }]),
        }
    }

    /// Run one step of MCTS. Returns `Some(result)` when search is complete.
    pub fn search_tick(&mut self, evals: &mut Vec<Evaluation>) -> Option<SearchResult> {
        match self.search.step(evals, &mut self.rng) {
            Step::NeedsEval(states) => {
                let refs: Vec<&G> = states.iter().collect();
                *evals = self.evaluator.evaluate_batch(&refs, &mut self.rng);
                None
            }
            Step::Done(result) => Some(result),
        }
    }

    /// Finish a streaming search: apply action (BotMove) or return snapshot (RunSims).
    pub fn finish_search(&mut self, msg: &ClientMsg, result: SearchResult) -> Vec<ServerMsg> {
        match msg {
            ClientMsg::BotMove { .. } => {
                let action = result.selected_action;
                let label = self.presenter.action_label(self.search.state(), action);
                let snapshot = self.search.snapshot();
                let action_labels = snapshot
                    .as_ref()
                    .map(|s| self.edge_labels(&s.edges))
                    .unwrap_or_default();
                self.apply_action(action);
                self.auto_resolve_chance();
                vec![
                    ServerMsg::BotAction {
                        action,
                        label,
                        snapshot,
                        action_labels,
                    },
                    self.state_msg(),
                ]
            }
            ClientMsg::RunSims { .. } => match self.search.snapshot() {
                Some(snap) => {
                    let labels = self.edge_labels(&snap.edges);
                    vec![ServerMsg::Snapshot {
                        snapshot: snap,
                        action_labels: labels,
                    }]
                }
                None => vec![ServerMsg::Error {
                    message: "No snapshot available".into(),
                }],
            },
            _ => vec![],
        }
    }

    /// Get current snapshot with action labels.
    pub fn snapshot_with_labels(&self) -> Option<(SearchSnapshot, Vec<String>)> {
        self.search.snapshot().map(|snap| {
            let labels = self.edge_labels(&snap.edges);
            (snap, labels)
        })
    }

    /// Get action labels for edge snapshots.
    fn edge_labels(&self, edges: &[crate::mcts::EdgeSnapshot]) -> Vec<String> {
        let state = self.search.state();
        edges
            .iter()
            .map(|e| self.presenter.action_label(state, e.action))
            .collect()
    }

    /// Label all nodes in a subtree by simulating actions from the root state.
    fn label_subtree(&self, tree: &mut crate::mcts::TreeNodeSnapshot) {
        let state = self.search.state().clone();
        label_subtree_walk(tree, state, &*self.presenter);
    }
}

fn label_subtree_walk<G: Game + Clone>(
    node: &mut crate::mcts::TreeNodeSnapshot,
    state: G, // state from which node.action was taken
    presenter: &dyn GamePresenter<G>,
) {
    // Label this node using the state before its action.
    if let Some(action) = node.action {
        node.label = Some(presenter.action_description(&state, action));
    }
    // Advance state through this node's action for children.
    let next = if let Some(action) = node.action {
        let mut s = state;
        s.apply_action(action);
        s
    } else {
        state
    };
    for child in &mut node.children {
        label_subtree_walk(child, next.clone(), presenter);
    }
}
