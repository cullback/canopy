# Lazy Chance Node Expansion

## Problem

Chance nodes (dice rolls, card draws) are the main depth bottleneck in MCTS. Each dice roll fans out 11 ways. Deep in the tree, a chance node with 20 visits expands into 11 children that each get ~2 visits — not real depth, just 11 network evaluations that approximate what a single pre-chance evaluation already knows.

## Core idea

Don't expand chance nodes into full subtrees until they have enough visits to justify it. Below the threshold, use a cheap baseline value. Above it, search selectively.

Two levels of sophistication:

### Level 1: single cached eval (simplest)

Cache the network's value at the pre-chance state on the first visit. The network already sees deck state / probability information, so its value is an implicit average over all outcomes. On subsequent visits below the threshold, back up the cached value without re-evaluating.

**Threshold**: expand when `visits > num_outcomes * K` (K = 2-4). A dice node (11 outcomes) needs ~22-44 visits. A coin flip (2 outcomes) needs 4-8. Adapts to branching factor and tree depth automatically.

### Level 2: exact baseline + selective deepening (better)

At a chance node with known probabilities p(o) for each outcome o:

1. Enumerate all outcomes, apply transitions to get successor states.
2. Batch-evaluate all successors with the network (one forward pass).
3. Compute exact expected baseline: `mu = sum(p(o) * v_net(s_o))`.
4. Only search deeply into a small subset of outcomes E.
5. Back up: `V = mu + sum over E of p(o) * (V_search(s_o) - v_net(s_o))`.

This keeps balanced-dice accuracy (all outcomes included with correct probabilities) while concentrating search budget on outcomes where search corrections matter.

**Which outcomes to refine**: prioritize by `p(o) * |v_net(s_o) - mu|` — high probability AND high deviation from the mean. For Catan, always refine 7 (qualitatively different), and rolls that swing resource affordability thresholds.

**Why this beats bucketing**: not approximate grouping of outcomes. All 11 outcomes are included exactly. The search budget is just allocated unevenly.

## Visit count management

When a lazy chance node hasn't expanded:

- First visit: evaluate, cache value, visit_count = 1
- Subsequent visits: back up cached value, increment expansion counter but NOT MCTS visit count

Keeping the expansion counter separate from the MCTS visit count avoids dilution: once expanded, new search-backed values immediately dominate Q rather than competing with N stale cached visits.

Simpler alternative: don't separate counters. The cached value is the network's genuine estimate, so it's a reasonable prior that search values gradually correct. Set K low (2) to minimize dilution.

## Expected depth gain

At 3200 sims with 11-way dice:

| Depth | Visits | Without lazy                | With lazy                       |
| ----- | ------ | --------------------------- | ------------------------------- |
| 1     | ~300   | Expand all 11               | Expand all 11 (above threshold) |
| 2     | ~30    | 11 children, ~3 visits each | Lazy — use baseline             |
| 3     | ~3     | 11 children, <1 visit each  | Lazy — use baseline             |

Each lazy chance node eliminates an 11x branching penalty, letting the search go deeper on lines that matter. Depth-2+ chance nodes are almost always better served by the network estimate than by fragmenting into 11 barely-visited branches.

## Generic implementation

This belongs in the MCTS framework (`src/mcts/`), not game-specific code. The `Game` trait already distinguishes chance nodes from decision nodes and provides outcome probabilities. The only new config is K (minimum visits per outcome before expanding).

Works for any game with chance nodes: dice rolls, card draws, random events. The branching factor adapts the threshold automatically.

## Literature

- Stochastic MuZero (Antonoglou et al., 2022): formalizes "afterstates" — post-decision, pre-chance states whose values are used directly.
- The exact-baseline + selective correction approach resembles RAVE/AMAF-style baseline corrections applied to chance nodes rather than action nodes.

## Related ideas (not in scope here)

- **Dual value heads (pre-roll / post-roll)**: train the network to explicitly predict pre-chance expected values with a consistency loss against the post-roll values. Makes level 2 more accurate.
- **Turn-plan search**: branch over unique end-of-turn states instead of primitive actions. Collapses multiple intra-turn plies into one search ply. Complementary to lazy chance nodes.
- **Adaptive root width**: use policy entropy to choose Gumbel-Top-K dynamically (4-8 for obvious positions, 16 for complex ones). Saves budget for depth.

## Open questions

- **Level 1 vs level 2**: is the batch eval of all 11 successors worth the cost? Level 1 is one eval, level 2 is 11 evals but gives exact baseline. For CPU inference, 11 evals might be too expensive at every chance node. For GPU with batching, it's one forward pass.
- **Interaction with Gumbel**: sequential halving operates at the root. Lazy chance nodes are deeper, so interaction should be minimal. Verify.
- **Training impact**: if the network sees fewer deep evaluations but more breadth at decision points, does value accuracy improve or degrade?
- **Selective deepening criteria**: beyond probability * deviation, should we always refine 7s? Rolls that cross affordability thresholds?
