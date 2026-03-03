use std::ops::{Index, IndexMut};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Player {
    One,
    Two,
}

impl std::fmt::Display for Player {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Player::One => write!(f, "P1"),
            Player::Two => write!(f, "P2"),
        }
    }
}

impl Player {
    pub fn opponent(self) -> Self {
        match self {
            Player::One => Player::Two,
            Player::Two => Player::One,
        }
    }

    pub fn sign(self) -> f32 {
        match self {
            Player::One => 1.0,
            Player::Two => -1.0,
        }
    }
}

/// A two-element array indexed by [`Player`] instead of `usize`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PerPlayer<T>(pub [T; 2]);

impl<T> Index<Player> for PerPlayer<T> {
    type Output = T;
    fn index(&self, p: Player) -> &T {
        &self.0[p as usize]
    }
}

impl<T> IndexMut<Player> for PerPlayer<T> {
    fn index_mut(&mut self, p: Player) -> &mut T {
        &mut self.0[p as usize]
    }
}
