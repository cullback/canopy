use canopy2::player::Player;

const INITIAL_DECK: [u8; 11] = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1];
const TOTAL_CARDS: u8 = 36;
const MIN_CARDS_RESHUFFLE: u8 = 13;
const RECENT_MEMORY: usize = 5;

#[derive(Clone, Copy, Debug)]
pub enum Dice {
    Random,
    Balanced(BalancedDice),
}

impl Default for Dice {
    fn default() -> Self {
        Dice::Random
    }
}

impl Dice {
    #[cfg(test)]
    pub fn roll(&mut self, rng: &mut fastrand::Rng, current_player: Player) -> u8 {
        match self {
            Dice::Random => rng.u8(1..=6) + rng.u8(1..=6),
            Dice::Balanced(b) => b.roll(rng, current_player),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BalancedDice {
    deck: [u8; 11],
    cards_left: u8,
    recent: [u8; RECENT_MEMORY],
    recent_count: [u8; 11],
    recent_len: u8,
    recent_start: u8,
    seven_streak_player: u8, // 0xFF = none
    seven_streak_count: u8,
    total_sevens: [u16; 2],
}

impl BalancedDice {
    pub fn new() -> Self {
        Self {
            deck: INITIAL_DECK,
            cards_left: TOTAL_CARDS,
            recent: [0; RECENT_MEMORY],
            recent_count: [0; 11],
            recent_len: 0,
            recent_start: 0,
            seven_streak_player: 0xFF,
            seven_streak_count: 0,
            total_sevens: [0; 2],
        }
    }

    pub fn cards_left(&self) -> u8 {
        self.cards_left
    }

    /// Unnormalized integer weights for each dice outcome (index 0..11 → sum 2..12).
    pub fn weights(&self, current_player: Player) -> [(usize, u32); 11] {
        let deck = if self.cards_left < MIN_CARDS_RESHUFFLE {
            &INITIAL_DECK
        } else {
            &self.deck
        };

        let mut weights = [0u32; 11];
        for i in 0..11 {
            let penalty = (100 - 34 * self.recent_count[i] as i32).max(0) as u32;
            weights[i] = deck[i] as u32 * penalty;
        }

        let seven_factor = self.seven_combined_100(current_player);
        for i in 0..11 {
            weights[i] *= if i == 5 { seven_factor } else { 100 };
        }

        let mut result = [(0usize, 0u32); 11];
        for i in 0..11 {
            result[i] = (i, weights[i]);
        }
        result
    }

    /// Sample a dice outcome using integer arithmetic only.
    /// Returns an action index 0..11 (sum 2..12).
    pub fn sample(&self, current_player: Player, rng: &mut fastrand::Rng) -> usize {
        let ws = self.weights(current_player);
        let total: u32 = ws.iter().map(|(_, w)| w).sum();
        if total == 0 {
            return rng.usize(0..11);
        }
        let mut r = rng.u32(0..total);
        for &(i, w) in &ws {
            if r < w {
                return i;
            }
            r -= w;
        }
        10
    }

    /// Seven combined adjustment factor in fixed-point scale 100.
    /// Returns 0..=200 (representing 0.0..=2.0).
    fn seven_combined_100(&self, current_player: Player) -> u32 {
        // Streak: ±0.4 * count → ±40 * count
        let streak: i32 = if self.seven_streak_count > 0 {
            let mag = 40 * self.seven_streak_count as i32;
            if self.seven_streak_player == current_player as u8 {
                -mag
            } else {
                mag
            }
        } else {
            0
        };

        // Imbalance: 2 - 2*my/total → 200 - 200*my/total
        let total_sevens = self.total_sevens[0] + self.total_sevens[1];
        let imbalance: i32 = if total_sevens >= 2 {
            let my = self.total_sevens[current_player as usize] as i32;
            200 - 200 * my / total_sevens as i32
        } else {
            100
        };

        (imbalance + streak).clamp(0, 200) as u32
    }

    #[cfg(test)]
    pub fn roll(&mut self, rng: &mut fastrand::Rng, current_player: Player) -> u8 {
        let chosen = self.sample(current_player, rng);
        let sum = (chosen + 2) as u8;
        self.draw(sum, current_player);
        sum
    }

    pub fn draw(&mut self, sum: u8, current_player: Player) {
        if self.cards_left < MIN_CARDS_RESHUFFLE {
            self.reshuffle();
        }
        let idx = (sum - 2) as usize;
        debug_assert!(self.deck[idx] > 0);
        self.deck[idx] -= 1;
        self.cards_left -= 1;

        // Update recent ring buffer
        if self.recent_len < RECENT_MEMORY as u8 {
            let pos = self.recent_len as usize;
            self.recent[pos] = sum;
            self.recent_len += 1;
        } else {
            // Evict oldest entry
            let oldest = self.recent[self.recent_start as usize];
            self.recent_count[(oldest - 2) as usize] -= 1;
            self.recent[self.recent_start as usize] = sum;
            self.recent_start = (self.recent_start + 1) % RECENT_MEMORY as u8;
        }
        self.recent_count[idx] += 1;

        // Update seven tracking (only on 7s — non-7 rolls leave streak untouched)
        if sum == 7 {
            let pid = current_player as usize;
            self.total_sevens[pid] += 1;
            if self.seven_streak_player == current_player as u8 {
                self.seven_streak_count += 1;
            } else {
                self.seven_streak_player = current_player as u8;
                self.seven_streak_count = 1;
            }
        }
    }

    fn reshuffle(&mut self) {
        self.deck = INITIAL_DECK;
        self.cards_left = TOTAL_CARDS;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_reshuffle() {
        let mut dice = BalancedDice::new();
        let mut rng = fastrand::Rng::with_seed(42);

        // Draw until below threshold, then next roll should reshuffle
        let mut draws = 0;
        while dice.cards_left >= MIN_CARDS_RESHUFFLE {
            dice.roll(&mut rng, Player::One);
            draws += 1;
            assert!(draws < 100, "should not take 100 draws to deplete deck");
        }

        // Next roll triggers reshuffle
        let before = dice.cards_left;
        assert!(before < MIN_CARDS_RESHUFFLE);
        dice.roll(&mut rng, Player::One);
        // After reshuffle + draw, cards_left should be TOTAL_CARDS - 1
        assert_eq!(
            dice.cards_left,
            TOTAL_CARDS - 1,
            "deck should reshuffle then draw one card"
        );
    }

    #[test]
    fn balanced_distribution() {
        let mut dice = BalancedDice::new();
        let mut rng = fastrand::Rng::with_seed(123);
        let mut counts = [0u32; 11];
        let n = 10_000;

        for i in 0..n {
            let player = if i % 2 == 0 { Player::One } else { Player::Two };
            let sum = dice.roll(&mut rng, player);
            assert!((2..=12).contains(&sum), "roll out of range: {sum}");
            counts[(sum - 2) as usize] += 1;
        }

        // Expected frequencies for 2d6 over 10k rolls
        let expected = [
            1.0 / 36.0,
            2.0 / 36.0,
            3.0 / 36.0,
            4.0 / 36.0,
            5.0 / 36.0,
            6.0 / 36.0,
            5.0 / 36.0,
            4.0 / 36.0,
            3.0 / 36.0,
            2.0 / 36.0,
            1.0 / 36.0,
        ];

        // Chi-squared test with generous threshold (balanced dice won't match
        // perfectly due to the streak/recent adjustments, but should be close)
        let mut chi2 = 0.0f64;
        for i in 0..11 {
            let exp = expected[i] * n as f64;
            let diff = counts[i] as f64 - exp;
            chi2 += diff * diff / exp;
        }

        // 10 degrees of freedom, p=0.001 critical value is ~29.6
        // We use a generous threshold since balanced dice intentionally skew
        assert!(
            chi2 < 100.0,
            "distribution too far from expected: chi2={chi2:.1}, counts={counts:?}"
        );
    }

    #[test]
    fn balanced_no_long_streaks() {
        // Run many games, check that no sum appears 4+ times in a row
        for seed in 0..20u64 {
            let mut dice = BalancedDice::new();
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut last_sum = 0u8;
            let mut streak = 0u32;

            for i in 0..500 {
                let player = if i % 2 == 0 { Player::One } else { Player::Two };
                let sum = dice.roll(&mut rng, player);
                if sum == last_sum {
                    streak += 1;
                } else {
                    streak = 1;
                    last_sum = sum;
                }
                assert!(
                    streak < 4,
                    "streak of {streak} for sum {sum} at roll {i} (seed {seed})"
                );
            }
        }
    }

    #[test]
    fn random_dice_passthrough() {
        let mut dice = Dice::Random;
        let mut rng = fastrand::Rng::with_seed(42);
        for _ in 0..100 {
            let sum = dice.roll(&mut rng, Player::One);
            assert!((2..=12).contains(&sum));
        }
    }

    #[test]
    fn balanced_dice_via_enum() {
        let mut dice = Dice::Balanced(BalancedDice::new());
        let mut rng = fastrand::Rng::with_seed(42);
        for _ in 0..100 {
            let sum = dice.roll(&mut rng, Player::One);
            assert!((2..=12).contains(&sum));
        }
    }
}
