use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Snapshot types (wire format for the JS frontend) ────────────────

/// Root-level search snapshot sent to the frontend.
#[derive(Debug, Serialize)]
pub struct SearchSnapshot {
    pub total_simulations: u32,
    pub pv_depth: u32,
    pub root_wdl: [f32; 3],
    pub network_value: f32,
    pub edges: Vec<EdgeSnapshot>,
}

/// Per-edge data in a search snapshot.
#[derive(Debug, Serialize)]
pub struct EdgeSnapshot {
    pub action: usize,
    pub visits: u32,
    pub q: Option<f32>,
    pub improved_policy: f32,
    pub depth: Option<u32>,
}

/// Recursive tree node for the subtree explorer.
#[derive(Debug, Serialize)]
pub struct TreeNodeSnapshot {
    pub action: Option<usize>,
    pub label: Option<String>,
    pub kind: String,
    pub visits: u32,
    pub wdl: [f32; 3],
    pub player: Option<u8>,
    pub children: Vec<TreeNodeSnapshot>,
}

// ── Client → Server ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMsg {
    /// Start a new game (optionally with a seed).
    NewGame { seed: Option<u64> },
    /// Human plays an action.
    PlayAction { action: usize },
    /// Request the bot to play an action.
    BotMove { simulations: Option<u32> },
    /// Run N additional simulations on current state (step debugger).
    RunSims { count: u32 },
    /// Request current search snapshot.
    GetSnapshot,
    /// Explore a subtree by following an action path.
    ExploreSubtree {
        action_path: Vec<usize>,
        depth: usize,
    },
    /// Take over control of a player (human overrides bot).
    TakeOver { player: u8 },
    /// Release control back to bot.
    ReleaseControl { player: u8 },
    /// Toggle autoplay mode.
    SetAutoplay {
        enabled: bool,
        delay_ms: Option<u64>,
    },
    /// Poll external state (e.g. colonist.io CDP). Default: returns current state.
    PollState,
    /// Request current game state.
    GetState,
    /// Undo last action.
    Undo,
    /// Redo previously undone action.
    Redo,
    /// Jump to a specific log entry (0-based index into labeled entries).
    SetLogCursor { index: usize },
    /// Configure per-player settings.
    SetConfig { player: u8, simulations: u32 },
    /// Enable/disable continuous background search with a sim target.
    SetAutoSearch { enabled: bool, target: u32 },
}

// ── Server → Client ──────────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ServerMsg {
    /// Full game state update.
    GameState {
        state: Value,
        legal_actions: Vec<ActionInfo>,
        current_player: u8,
        phase: String,
        is_chance: bool,
        is_terminal: bool,
        result: Option<String>,
        action_log: Vec<String>,
        can_undo: bool,
        can_redo: bool,
    },
    /// MCTS search snapshot.
    Snapshot {
        snapshot: SearchSnapshot,
        action_labels: Vec<String>,
    },
    /// Subtree exploration result (labels embedded in tree nodes).
    Subtree { tree: TreeNodeSnapshot },
    /// Bot played an action.
    BotAction {
        action: usize,
        label: String,
        snapshot: Option<SearchSnapshot>,
        action_labels: Vec<String>,
    },
    /// Live search progress update.
    SearchProgress {
        snapshot: SearchSnapshot,
        action_labels: Vec<String>,
        sims_total: u32,
    },
    /// Error message.
    Error { message: String },
}

#[derive(Debug, Serialize)]
pub struct ActionInfo {
    pub action: usize,
    pub label: String,
}
