/// Pig dice game.
///
/// Two players take turns rolling a single die. On your turn you can
/// ROLL (risk your turn total) or HOLD (bank it). Rolling a 1 busts
/// and ends your turn with nothing banked. First to the target wins.

pub const ROLL: usize = 0;
pub const HOLD: usize = 1;
pub const NUM_ACTIONS: usize = 2;
pub const NUM_DIE_FACES: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Player {
    One,
    Two,
}

impl Player {
    fn opponent(self) -> Self {
        match self {
            Player::One => Player::Two,
            Player::Two => Player::One,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Player::One => 0,
            Player::Two => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PigGame {
    scores: [u32; 2],
    current: Player,
    turn_total: u32,
    rolling: bool,
    target: u32,
}

impl std::fmt::Display for PigGame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pig:{}", self.target)
    }
}

impl std::str::FromStr for PigGame {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let target = s
            .strip_prefix("pig:")
            .ok_or("expected pig:<target>")?
            .parse()
            .map_err(|e| format!("invalid target: {e}"))?;
        Ok(Self::new(target))
    }
}

impl PigGame {
    pub fn new(target: u32) -> Self {
        Self {
            scores: [0, 0],
            current: Player::One,
            turn_total: 0,
            rolling: false,
            target,
        }
    }

    pub fn current_player(&self) -> Player {
        self.current
    }

    pub fn winner(&self) -> Option<Player> {
        if self.scores[0] >= self.target {
            Some(Player::One)
        } else if self.scores[1] >= self.target {
            Some(Player::Two)
        } else {
            None
        }
    }

    /// True when the game is waiting for a die roll (chance event),
    /// false when the current player must choose ROLL or HOLD.
    pub fn scores(&self) -> [u32; 2] {
        self.scores
    }

    pub fn turn_total(&self) -> u32 {
        self.turn_total
    }

    pub fn is_rolling(&self) -> bool {
        self.rolling
    }

    /// Apply a player decision (ROLL or HOLD).
    pub fn apply_decision(&mut self, action: usize) {
        debug_assert!(!self.rolling);
        match action {
            ROLL => self.rolling = true,
            HOLD => {
                self.scores[self.current.index()] += self.turn_total;
                self.pass_turn();
            }
            _ => panic!("invalid action {action}"),
        }
    }

    /// Apply a die outcome (0..6 mapping to faces 1..=6).
    pub fn apply_roll(&mut self, outcome: usize) {
        debug_assert!(self.rolling);
        let face = outcome as u32 + 1;
        if face == 1 {
            self.pass_turn();
        } else {
            self.turn_total += face;
            self.rolling = false;
        }
    }

    fn pass_turn(&mut self) {
        self.current = self.current.opponent();
        self.turn_total = 0;
        self.rolling = false;
    }
}
