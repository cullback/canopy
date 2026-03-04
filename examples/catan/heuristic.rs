use canopy2::eval::{Evaluation, Evaluator};

use crate::game::action::{
    self, BUY_DEV_CARD, CITY_END, CITY_START, END_TURN, MARITIME_END, MARITIME_START, MONOPOLY_END,
    MONOPOLY_START, PLAY_KNIGHT, PLAY_ROAD_BUILDING, ROAD_END, ROAD_START, SETTLEMENT_END,
    SETTLEMENT_START, YOP_END, YOP_START,
};
use crate::game::state::GameState;

/// Rollout evaluator with a hand-crafted policy prior.
///
/// Uses random rollouts for the value (like `RolloutEvaluator`) but returns
/// non-uniform policy logits that bias MCTS toward building actions. This
/// makes the MCTS tree search itself smarter without changing the rollouts.
///
/// Policy inspired by Szita et al. (2009), with moderate logit magnitudes
/// so the MCTS search can still improve the policy:
/// - Settlement/City: strong preference (+1 VP and resource income)
/// - Road: moderate, decays with road-to-building ratio (floored at 0)
/// - Knight: boosted if robber blocks own field
/// - Other dev card plays: moderate
/// - Everything else: baseline
pub struct HeuristicEvaluator {
    pub rollout: canopy2::eval::RolloutEvaluator,
}

impl Evaluator<GameState> for HeuristicEvaluator {
    fn evaluate(&self, state: &GameState, rng: &mut fastrand::Rng) -> Evaluation {
        let mut out = self.rollout.evaluate(state, rng);
        out.policy_logits = policy_logits(state);
        out
    }
}

/// Compute state-dependent policy logits.
///
/// Relative ordering from Szita et al. (2009), scaled to moderate logit range
/// so the MCTS search can still improve the policy. The paper's raw weights
/// (10,000 for settlements) are designed for rollout sampling; as MCTS priors,
/// ln(10000) ≈ 9.2 creates a near-deterministic distribution that the search
/// can't overcome. We preserve the state-dependent structure (road ratio,
/// robber check) with practical magnitudes.
fn policy_logits(state: &GameState) -> Vec<f32> {
    let mut logits = vec![0.0f32; action::ACTION_SPACE];

    let pid = state.current_player;
    let player = &state.players[pid];

    // Settlement/City: strongest preference
    for i in SETTLEMENT_START..SETTLEMENT_END {
        logits[i as usize] = 3.0;
    }
    for i in CITY_START..CITY_END {
        logits[i as usize] = 3.0;
    }

    // Road: decays as road-to-building ratio R grows, floored at baseline.
    // R=0 → 2.0, R=1 → 1.0, R≥2 → 0.0 (never penalized below base weight)
    let settlements = (5 - player.settlements_left) as f32;
    let cities = (4 - player.cities_left) as f32;
    let buildings = settlements + cities;
    let r = if buildings > 0.0 {
        player.roads_placed as f32 / buildings
    } else {
        0.0
    };
    let road_logit = (2.0 - r).max(0.0);
    for i in ROAD_START..ROAD_END {
        logits[i as usize] = road_logit;
    }

    // Knight: strong if robber blocks own field, baseline otherwise
    let my_buildings = state.player_buildings(pid);
    let robber_mask = state.topology.adj.tile_nodes[state.robber.0 as usize];
    let robber_on_own = (robber_mask & my_buildings) != 0;
    logits[PLAY_KNIGHT as usize] = if robber_on_own { 2.5 } else { 0.0 };

    // Other dev card plays: moderate preference
    logits[PLAY_ROAD_BUILDING as usize] = 1.5;
    for i in YOP_START..YOP_END {
        logits[i as usize] = 1.5;
    }
    for i in MONOPOLY_START..MONOPOLY_END {
        logits[i as usize] = 1.5;
    }

    // Buy dev card: moderate
    logits[BUY_DEV_CARD as usize] = 1.5;

    // Maritime trade: slight preference over doing nothing
    for i in MARITIME_START..MARITIME_END {
        logits[i as usize] = 0.5;
    }

    // End turn: low priority (explore building options first)
    logits[END_TURN as usize] = -1.0;

    // Everything else (robber, discard, steal, roll) stays at 0.0

    logits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game;
    use crate::game::dice::Dice;
    use canopy2::eval::Evaluator;
    use canopy2::game::Game;
    use canopy2::player::Player;

    #[test]
    fn heuristic_returns_bounded_value() {
        let state = game::new_game(42, Dice::default());
        let eval = HeuristicEvaluator {
            rollout: canopy2::eval::RolloutEvaluator { num_rollouts: 1 },
        };
        let mut rng = fastrand::Rng::with_seed(0);
        let out = eval.evaluate(&state, &mut rng);
        assert!(out.value >= -1.0 && out.value <= 1.0);
        assert_eq!(out.policy_logits.len(), GameState::NUM_ACTIONS);
    }

    #[test]
    fn road_logit_decays_but_never_negative() {
        let mut state = game::new_game(42, Dice::default());
        state.players[Player::One].settlements_left = 3; // 2 settlements
        state.current_player = Player::One;

        state.players[Player::One].roads_placed = 2;
        let logits_few = policy_logits(&state);

        state.players[Player::One].roads_placed = 6;
        let logits_many = policy_logits(&state);

        assert!(
            logits_few[ROAD_START as usize] > logits_many[ROAD_START as usize],
            "road logit should decrease as road-to-building ratio grows"
        );
        assert!(
            logits_many[ROAD_START as usize] >= 0.0,
            "road logit should never go negative"
        );
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
            logits[PLAY_KNIGHT as usize] > 2.0,
            "knight logit should be boosted when robber is on own field"
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
