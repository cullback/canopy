use std::ops::{Index, IndexMut};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Resource {
    Lumber = 0,
    Brick = 1,
    Wool = 2,
    Grain = 3,
    Ore = 4,
}

impl std::fmt::Display for Resource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Resource::Lumber => write!(f, "lumber"),
            Resource::Brick => write!(f, "brick"),
            Resource::Wool => write!(f, "wool"),
            Resource::Grain => write!(f, "grain"),
            Resource::Ore => write!(f, "ore"),
        }
    }
}

pub const ALL_RESOURCES: [Resource; 5] = [
    Resource::Lumber,
    Resource::Brick,
    Resource::Wool,
    Resource::Grain,
    Resource::Ore,
];

/// Fixed-size array indexed by `Resource`. Used for hands, bank, costs.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Debug)]
pub struct ResourceArray(pub [u8; 5]);

impl Index<Resource> for ResourceArray {
    type Output = u8;
    fn index(&self, r: Resource) -> &u8 {
        &self.0[r as usize]
    }
}

impl IndexMut<Resource> for ResourceArray {
    fn index_mut(&mut self, r: Resource) -> &mut u8 {
        &mut self.0[r as usize]
    }
}

impl ResourceArray {
    pub const fn new(lumber: u8, brick: u8, wool: u8, grain: u8, ore: u8) -> Self {
        Self([lumber, brick, wool, grain, ore])
    }

    pub fn total(self) -> u8 {
        self.0.iter().sum()
    }

    /// Returns true if self has at least as many of every resource as `other`.
    pub fn contains(self, other: ResourceArray) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(a, b)| a >= b)
    }

    pub fn add(&mut self, other: ResourceArray) {
        for i in 0..5 {
            self.0[i] = self.0[i].saturating_add(other.0[i]);
        }
    }

    pub fn sub(&mut self, other: ResourceArray) {
        for i in 0..5 {
            if self.0[i] < other.0[i] {
                #[cfg(debug_assertions)]
                panic!(
                    "resource underflow: have {:?} sub {:?} (index {i})",
                    self.0, other.0,
                );
                #[cfg(not(debug_assertions))]
                {
                    self.0[i] = 0;
                    continue;
                }
            }
            self.0[i] -= other.0[i];
        }
    }
}

// Building costs
pub const ROAD_COST: ResourceArray = ResourceArray::new(1, 1, 0, 0, 0);
pub const SETTLEMENT_COST: ResourceArray = ResourceArray::new(1, 1, 1, 1, 0);
pub const CITY_COST: ResourceArray = ResourceArray::new(0, 0, 0, 2, 3);
pub const DEV_CARD_COST: ResourceArray = ResourceArray::new(0, 0, 1, 1, 1);
