use canopy2::eval::{Evaluator, NnOutput};
use canopy2::game::Game;
use canopy2::player::Player;

use crate::game::action::{
    self, BUY_DEV_CARD, CITY_END, CITY_START, MONOPOLY_END, MONOPOLY_START, PLAY_KNIGHT,
    PLAY_ROAD_BUILDING, ROAD_END, ROAD_START, SETTLEMENT_END, SETTLEMENT_START, YOP_END, YOP_START,
};
use crate::game::state::GameState;

/// Static heuristic evaluator: no rollouts, scores leaves by VP difference.
pub struct HeuristicEvaluator;

impl Evaluator<GameState> for HeuristicEvaluator {
    fn evaluate(&self, state: &GameState, _rng: &mut fastrand::Rng) -> NnOutput {
        let vp1 = f32::from(state.total_vps(Player::One));
        let vp2 = f32::from(state.total_vps(Player::Two));
        NnOutput {
            policy_logits: vec![0.0; GameState::NUM_ACTIONS],
            value: (vp1 - vp2).tanh(),
        }
    }
}

/// Rollout evaluator with a hand-crafted policy prior.
///
/// Uses random rollouts for the value (like `RolloutEvaluator`) but returns
/// non-uniform policy logits that bias MCTS toward building actions. This
/// makes the MCTS tree search itself smarter without changing the rollouts.
///
/// Policy weights from Szita et al. (2009), converted to logits via ln(weight):
/// - Settlement/City: +10,000  (always good: +1 VP and resource income)
/// - Road: 10/10^R where R = roads / (settlements + cities)
/// - Knight: +100 if robber blocks own field, +1 otherwise
/// - Other dev card plays (YOP, Monopoly, Road Building): +10
/// - Everything else: +1 (base weight)
pub struct PolicyEvaluator {
    pub rollout: canopy2::eval::RolloutEvaluator,
}

impl Evaluator<GameState> for PolicyEvaluator {
    fn evaluate(&self, state: &GameState, rng: &mut fastrand::Rng) -> NnOutput {
        let mut out = self.rollout.evaluate(state, rng);
        out.policy_logits = policy_logits(state);
        out
    }
}

/// Compute state-dependent policy logits.
///
/// Weights are from the paper; logits = ln(weight) so that softmax(logits) ∝ weights.
fn policy_logits(state: &GameState) -> Vec<f32> {
    const LN_10: f32 = std::f32::consts::LN_10;
    const LN_100: f32 = LN_10 * 2.0;
    const LN_10000: f32 = LN_10 * 4.0;

    let mut logits = vec![0.0f32; action::ACTION_SPACE];

    let pid = state.current_player;
    let player = &state.players[pid];

    // Settlement/City: weight 10,000 → logit ln(10000)
    for i in SETTLEMENT_START..SETTLEMENT_END {
        logits[i as usize] = LN_10000;
    }
    for i in CITY_START..CITY_END {
        logits[i as usize] = LN_10000;
    }

    // Road: weight = 10 / 10^R where R = roads / (settlements + cities)
    // logit = ln(10) * (1 - R)
    let settlements = (5 - player.settlements_left) as f32;
    let cities = (4 - player.cities_left) as f32;
    let buildings = settlements + cities;
    let r = if buildings > 0.0 {
        player.roads_placed as f32 / buildings
    } else {
        0.0
    };
    let road_logit = LN_10 * (1.0 - r);
    for i in ROAD_START..ROAD_END {
        logits[i as usize] = road_logit;
    }

    // Knight: weight 100 if robber blocks own field, weight 1 otherwise
    let my_buildings = state.player_buildings(pid);
    let robber_mask = state.topology.adj.tile_nodes[state.robber.0 as usize];
    let robber_on_own = (robber_mask & my_buildings) != 0;
    logits[PLAY_KNIGHT as usize] = if robber_on_own { LN_100 } else { 0.0 };

    // Other dev card plays: weight 10 → logit ln(10)
    logits[PLAY_ROAD_BUILDING as usize] = LN_10;
    for i in YOP_START..YOP_END {
        logits[i as usize] = LN_10;
    }
    for i in MONOPOLY_START..MONOPOLY_END {
        logits[i as usize] = LN_10;
    }

    // Buy dev card: weight 10 → logit ln(10)
    logits[BUY_DEV_CARD as usize] = LN_10;

    // Maritime trade: base weight 1 → logit 0 (already default)

    // End turn: base weight 1 → logit 0 (already default)

    // Everything else (robber, discard, steal, roll) stays at 0.0

    logits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game;
    use crate::game::action::END_TURN;
    use crate::game::dice::Dice;
    use canopy2::eval::Evaluator;

    #[test]
    fn heuristic_returns_bounded_value() {
        let state = game::new_game(42, Dice::default());
        let eval = HeuristicEvaluator;
        let mut rng = fastrand::Rng::with_seed(0);
        let out = eval.evaluate(&state, &mut rng);
        assert!(out.value >= -1.0 && out.value <= 1.0);
        assert_eq!(out.policy_logits.len(), GameState::NUM_ACTIONS);
    }

    #[test]
    fn policy_returns_bounded_value() {
        let state = game::new_game(42, Dice::default());
        let eval = PolicyEvaluator {
            rollout: canopy2::eval::RolloutEvaluator { num_rollouts: 1 },
        };
        let mut rng = fastrand::Rng::with_seed(0);
        let out = eval.evaluate(&state, &mut rng);
        assert!(out.value >= -1.0 && out.value <= 1.0);
        assert_eq!(out.policy_logits.len(), GameState::NUM_ACTIONS);
    }

    #[test]
    fn policy_prefers_building_over_end_turn() {
        let state = game::new_game(42, Dice::default());
        let logits = policy_logits(&state);
        let settle = logits[SETTLEMENT_START as usize];
        let end = logits[END_TURN as usize];
        assert!(
            settle > end,
            "settlement ({settle}) should have higher prior than end_turn ({end})"
        );
    }

    #[test]
    fn road_logit_decreases_with_more_roads() {
        let mut state = game::new_game(42, Dice::default());
        // Simulate some buildings placed
        state.players[Player::One].settlements_left = 3; // 2 settlements
        state.players[Player::One].roads_placed = 2;
        state.current_player = Player::One;
        let logits_few = policy_logits(&state);

        state.players[Player::One].roads_placed = 6;
        let logits_many = policy_logits(&state);

        assert!(
            logits_few[ROAD_START as usize] > logits_many[ROAD_START as usize],
            "road logit should decrease as road-to-building ratio grows"
        );
    }

    #[test]
    fn knight_boosted_when_robber_on_own_field() {
        let mut state = game::new_game(42, Dice::default());
        state.current_player = Player::One;

        // Place a settlement for P1 and put robber on a tile touching it
        let tile = &state.topology.tiles[0];
        let node = tile.nodes[0];
        state.boards[Player::One].settlements |= 1u64 << node.0;
        state.players[Player::One].settlements_left -= 1;
        state.robber = tile.id;

        let logits = policy_logits(&state);
        assert!(
            logits[PLAY_KNIGHT as usize] > 4.0,
            "knight logit should be ln(100) ≈ 4.6 when robber is on own field"
        );

        // Move robber elsewhere — no own buildings there
        let other_tile = state
            .topology
            .tiles
            .iter()
            .find(|t| {
                let mask = state.topology.adj.tile_nodes[t.id.0 as usize];
                mask & state.player_buildings(Player::One) == 0
            })
            .unwrap();
        state.robber = other_tile.id;

        let logits = policy_logits(&state);
        assert!(
            logits[PLAY_KNIGHT as usize] < 0.01,
            "knight logit should be 0 when robber is not on own field"
        );
    }
}
