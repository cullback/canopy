use std::ops::{Index, IndexMut};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum DevCardKind {
    Knight = 0,
    VictoryPoint = 1,
    RoadBuilding = 2,
    YearOfPlenty = 3,
    Monopoly = 4,
}

impl DevCardKind {
    pub const ALL: [DevCardKind; 5] = [
        DevCardKind::Knight,
        DevCardKind::VictoryPoint,
        DevCardKind::RoadBuilding,
        DevCardKind::YearOfPlenty,
        DevCardKind::Monopoly,
    ];
}

/// Fixed-size array indexed by `DevCardKind`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Debug)]
pub struct DevCardArray(pub [u8; 5]);

impl Index<DevCardKind> for DevCardArray {
    type Output = u8;
    fn index(&self, k: DevCardKind) -> &u8 {
        &self.0[k as usize]
    }
}

impl IndexMut<DevCardKind> for DevCardArray {
    fn index_mut(&mut self, k: DevCardKind) -> &mut u8 {
        &mut self.0[k as usize]
    }
}

/// Count-based development card deck tracking remaining cards by type.
#[derive(Clone, Copy, Debug)]
pub struct DevCardDeck {
    remaining: DevCardArray,
}

impl DevCardDeck {
    /// Standard deck: 14 knights, 5 VP, 2 road building, 2 year of plenty, 2 monopoly.
    pub fn new() -> Self {
        Self {
            remaining: DevCardArray([14, 5, 2, 2, 2]),
        }
    }

    /// Total cards remaining in the deck.
    pub fn total_remaining(&self) -> u8 {
        self.remaining.0.iter().sum()
    }

    /// Count of a specific card type remaining.
    pub fn remaining_of(&self, kind: DevCardKind) -> u8 {
        self.remaining[kind]
    }

    /// Reference to the remaining counts array.
    pub fn remaining_counts(&self) -> &DevCardArray {
        &self.remaining
    }

    /// Remove one card of a specific type from the deck.
    pub fn remove(&mut self, kind: DevCardKind) {
        debug_assert!(self.remaining[kind] > 0, "no {kind:?} cards left in deck");
        self.remaining[kind] -= 1;
    }

    /// Remove one unknown card — decrements the most common type.
    /// Used for colonist replay where we observe a buy but not the result.
    pub fn remove_unknown(&mut self) {
        let mut best_idx = 0;
        let mut best_count = 0;
        for (i, &c) in self.remaining.0.iter().enumerate() {
            if c > best_count {
                best_count = c;
                best_idx = i;
            }
        }
        debug_assert!(best_count > 0, "deck is empty");
        self.remaining.0[best_idx] -= 1;
    }

    pub fn is_empty(&self) -> bool {
        self.total_remaining() == 0
    }

    #[cfg(test)]
    pub fn from_counts(counts: [u8; 5]) -> Self {
        Self {
            remaining: DevCardArray(counts),
        }
    }
}
