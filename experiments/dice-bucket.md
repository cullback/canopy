# Dice Bucketing

## Problem

Dice chance nodes are the main bottleneck for search depth. Each roll fans out up to 11 ways. With ~3 actions per decision, ~3 decisions per turn, and weighted sampling, 3200 sims reaches ~2 turns of meaningful depth. Each full round (my turn + dice + opp turn + dice) costs ~3000x budget.

More sims has diminishing returns: 1200->3200 (2.7x) produced no tournament strength gain.

## Approach: bucket by resource distribution

Group dice outcomes that produce identical resource effects into a single tree edge. Each bucket represents "what happened to the board" rather than "which number rolled."

Early game example — P1 settled on 6-wheat and 8-ore, P2 on 5-wood and 9-brick:

| Bucket | Rolls               | Effect    |
| ------ | ------------------- | --------- |
| A      | 2, 3, 4, 10, 11, 12 | nothing   |
| B      | 5                   | P2 +wood  |
| C      | 6                   | P1 +wheat |
| D      | 7                   | robber    |
| E      | 8                   | P1 +ore   |
| F      | 9                   | P2 +brick |

6 buckets instead of 11. Bucket A (nothing) dominates probability mass — search goes deep there for free.

Late game with buildings on most numbers: fewer merges, but depth matters less (games are mostly decided).

## Preserving balanced dice accuracy

Each simulation still samples a real dice outcome from the balanced deck, then maps to the correct bucket in the tree. Same pattern as SO-ISMCTS for hidden cards — the tree is coarser but each sim carries exact game state including deck.

The approximation: within a bucket, different outcomes leave different deck states. The tree's value estimate averages across them. The total predictive edge of balanced dice tracking is 0.26 bits/roll (see law-of-small-numbers post). Within-bucket deck divergence is a fraction of that — two rolls that produce the same resources differ only in which deck cards were drawn.

Seven always gets its own bucket (triggers robber, distinct game effect).

## Expected depth gain — not enough

After setup, both players have settlements touching 6-8 unique numbers. Only 2-3 numbers produce nothing, so bucketing reduces 11 branches to ~8-9. A reduction of 2-3, not transformative.

By mid game, nearly every number is active with different resource distributions. Bucketing saves almost nothing.

**Verdict: not worth implementing.** The practical branching reduction is too small to meaningfully increase search depth.

## Implementation

1. When expanding a dice chance node, compute resource delta for each outcome
2. Group outcomes with identical (p1_resources, p2_resources) into one edge
3. Edge weight = sum of constituent outcome weights
4. During simulation: sample real outcome from balanced deck, map to bucket edge
5. If the sampled outcome's bucket has no tree edge yet (SO-ISMCTS divergence), abort sim as we do today

No changes needed to training, colonist replay (uses real dice), or the neural network. The encoder still sees exact deck state per-sim.

## Evaluation

Tournament: bucketed search vs unbucketed at equal sims. If bucketing helps, equivalent strength at fewer sims (faster training iterations).
