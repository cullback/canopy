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

/// A shuffled deck of development cards. Pop from the end to draw.
#[derive(Clone, Copy, Debug)]
pub struct DevCardDeck {
    cards: [DevCardKind; 25],
    len: u8,
}

impl DevCardDeck {
    /// Standard deck: 14 knights, 5 VP, 2 road building, 2 year of plenty, 2 monopoly.
    pub fn new(rng: &mut fastrand::Rng) -> Self {
        let mut cards = [DevCardKind::Knight; 25];
        // 14 knights already filled, set the rest
        for i in 14..19 {
            cards[i] = DevCardKind::VictoryPoint;
        }
        cards[19] = DevCardKind::RoadBuilding;
        cards[20] = DevCardKind::RoadBuilding;
        cards[21] = DevCardKind::YearOfPlenty;
        cards[22] = DevCardKind::YearOfPlenty;
        cards[23] = DevCardKind::Monopoly;
        cards[24] = DevCardKind::Monopoly;
        rng.shuffle(&mut cards);
        Self { cards, len: 25 }
    }

    #[cfg(test)]
    pub fn from_cards(cards: &[DevCardKind]) -> Self {
        let mut deck = Self {
            cards: [DevCardKind::Knight; 25],
            len: cards.len() as u8,
        };
        deck.cards[..cards.len()].copy_from_slice(cards);
        deck
    }

    pub fn draw(&mut self) -> Option<DevCardKind> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(self.cards[self.len as usize])
    }

    pub fn remaining(&self) -> u8 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
