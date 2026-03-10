use canopy2::player::Player;

const INITIAL_DECK: [u8; 11] = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1];
const TOTAL_CARDS: u8 = 36;
const MIN_CARDS_RESHUFFLE: u8 = 13;
const RECENT_MEMORY: usize = 5;
const PROB_REDUCTION_RECENT: f32 = 0.34;
const PROB_REDUCTION_SEVEN_STREAK: f32 = 0.4;

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

    pub fn probabilities(&self, current_player: Player) -> [(u32, f32); 11] {
        let mut weights = [0.0f32; 11];

        // Base probability from remaining deck counts
        // If deck is nearly empty, use initial deck weights (reshuffle will happen at draw time)
        if self.cards_left < MIN_CARDS_RESHUFFLE {
            for i in 0..11 {
                weights[i] = INITIAL_DECK[i] as f32;
            }
        } else {
            for i in 0..11 {
                weights[i] = self.deck[i] as f32;
            }
        }

        // Recent-roll penalty
        for i in 0..11 {
            let penalty = 1.0 - PROB_REDUCTION_RECENT * self.recent_count[i] as f32;
            weights[i] *= penalty.max(0.0);
        }

        // Seven-streak adjustment (index 5 = sum 7)
        self.apply_seven_adjustment(&mut weights, current_player);

        // Normalize
        let total_weight: f32 = weights.iter().sum();
        let mut result = [(0u32, 0.0f32); 11];
        if total_weight > 0.0 {
            for i in 0..11 {
                result[i] = (i as u32, weights[i] / total_weight);
            }
        } else {
            // Fallback: uniform over initial deck
            for i in 0..11 {
                result[i] = (i as u32, INITIAL_DECK[i] as f32 / TOTAL_CARDS as f32);
            }
        }
        result
    }

    #[cfg(test)]
    pub fn roll(&mut self, rng: &mut fastrand::Rng, current_player: Player) -> u8 {
        let mut weights = [0.0f32; 11];

        // Base probability from remaining deck counts
        // If deck is nearly empty, use initial deck weights (reshuffle will happen at draw time)
        if self.cards_left < MIN_CARDS_RESHUFFLE {
            for i in 0..11 {
                weights[i] = INITIAL_DECK[i] as f32;
            }
        } else {
            for i in 0..11 {
                weights[i] = self.deck[i] as f32;
            }
        }

        // Recent-roll penalty
        for i in 0..11 {
            let penalty = 1.0 - PROB_REDUCTION_RECENT * self.recent_count[i] as f32;
            weights[i] *= penalty.max(0.0);
        }

        // Seven-streak adjustment (index 5 = sum 7)
        self.apply_seven_adjustment(&mut weights, current_player);

        // Weighted draw
        let total_weight: f32 = weights.iter().sum();
        if total_weight <= 0.0 {
            // Fallback: pick uniformly from available deck slots
            return self.fallback_draw(rng, current_player);
        }

        let mut pick = rng.f32() * total_weight;
        let mut chosen = 10; // default to last
        for i in 0..11 {
            pick -= weights[i];
            if pick <= 0.0 {
                chosen = i;
                break;
            }
        }

        let sum = (chosen + 2) as u8;
        self.draw(sum, current_player);
        sum
    }

    fn apply_seven_adjustment(&self, weights: &mut [f32; 11], current_player: Player) {
        let seven_idx = 5; // sum 7 is at index 5

        // Streak adjustment: reduce if current player is on a 7-streak,
        // boost if the OTHER player is on a 7-streak
        let streak_adj = if self.seven_streak_count > 0 {
            let magnitude = PROB_REDUCTION_SEVEN_STREAK * self.seven_streak_count as f32;
            if self.seven_streak_player == current_player as u8 {
                -magnitude // current player streaking -> suppress
            } else {
                magnitude // other player streaking -> boost
            }
        } else {
            0.0
        };

        // Imbalance adjustment: push toward equal share of 7s
        // Activates once total sevens >= num_players (2 for 2p)
        let total_sevens = self.total_sevens[0] + self.total_sevens[1];
        let imbalance_adj = if total_sevens >= 2 {
            let total = total_sevens as f32;
            let ideal_share = 0.5; // 1 / num_players for 2p
            let actual_share = self.total_sevens[current_player as usize] as f32 / total;
            1.0 + (ideal_share - actual_share) / ideal_share
        } else {
            1.0
        };

        // Combined: additive, clamped to [0, 2]
        let combined = (imbalance_adj + streak_adj).clamp(0.0, 2.0);
        weights[seven_idx] *= combined;
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

    #[cfg(test)]
    fn fallback_draw(&mut self, rng: &mut fastrand::Rng, current_player: Player) -> u8 {
        // Pick uniformly from remaining cards in deck
        if self.cards_left == 0 {
            self.reshuffle();
        }
        let pick = rng.u8(0..self.cards_left);
        let mut count = 0u8;
        for i in 0..11 {
            count += self.deck[i];
            if count > pick {
                let sum = (i + 2) as u8;
                self.draw(sum, current_player);
                return sum;
            }
        }
        unreachable!()
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
