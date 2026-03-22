# Canopy

MCTS framework for two player games.

- a general MCTS framework for one and two player games
- perfect-information, optionally stochastic
- best-in-class framework for training games on single machine systems (not distributed, not networked)
- autoregressive actions

## Compared to gumbel paper

1. Gumbel-Top-k sampling (Algorithm 1) — init_gumbel samples Gumbel(0) per action, selects top-m by g + logit
2. Sequential Halving (Algorithm 2) — round-robin allocation, halving by g + logit + σ(q), budget tracking
3. σ transformation (Eq. 8) — (c_visit + max_visits) * c_scale * q_norm
4. Improved policy target (Eq. 11) — softmax(logits + σ(completedQ)) over all edges
5. Non-root deterministic selection (Eq. 14, "Full Gumbel MuZero") — argmax(π'(a) - N(a)/(1+ΣN))
6. completedQ (Eq. 10) — child Q if expanded, v_mix otherwise
7. v_mix (Eq. 33) — prior-weighted Q interpolated with NN value
8. Policy loss — cross-entropy -Σ π'(a) log π(a) is equivalent to KL(π', π) for optimization (differ by constant H(π'))
9. Default hyperparameters — c_visit=50, c_scale=1.0 match the paper

Differences

1. Transposition table — state deduplication via state_key() / DAG structure. Paper doesn't mention transpositions.
2. Tree reuse between turns — step_to + compact preserves and reroots subtrees. Standard optimization not in the paper.
3. Arena-based tree — contiguous node/edge storage with index-based navigation. Better cache locality.
4. Value target z/q blending — target = (1-α)z + αq with linear ramp. Paper only uses z (game outcome). Critical for
   stochastic games with high variance outcomes.
5. Native chance node support — extends the algorithm to stochastic games (dice, cards). Paper only addresses
   perfect-information games.
6. Path-only Q-bound tracking — creates a natural warmup: trust policy early (tight bounds → small σ), trust Q later
   (wide bounds → large σ). Paper uses full-tree bounds.
7. Forced move skipping — single legal action bypasses MCTS entirely.

## References

- <https://suragnair.github.io/posts/alphazero.html>
- <https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf>
- <https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191>
- <https://freedium-mirror.cfd/https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191>
- <https://brantondemoss.com/writing/kata/>
- <https://gwern.net/doc/reinforcement-learning/model/alphago/2017-silver.pdf#page=5>
- <https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md>
- [Enhancements for Real-Time Monte-Carlo Tree Search in General Video Game Playing (2024)](https://arxiv.org/html/2407.03049v1)
  - progressive simulation
- [cosine learning rate](https://medium.com/@utkrisht14/cosine-learning-rate-schedulers-in-pytorch-486d8717d541)

repos

- <https://github.com/google-deepmind/mctx>
- <https://github.com/gorisanson/quoridor-ai>
- <https://github.com/Aenteas/MCTS>
- <https://github.com/hzyhhzy/KataGomo>
