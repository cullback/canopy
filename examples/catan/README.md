# Catan

Two-player Catan with MCTS search, a web analysis board, and a live
[colonist.io](https://colonist.io) integration via CDP.

## Search optimizations

### Lexicographic discard ordering

When a 7 rolls and a player holds more than 9 cards, they must discard
half. Discarding N cards from 5 resource types produces a large number
of equivalent orderings — discarding lumber then brick reaches the same
state as brick then lumber. Without pruning, the branching factor
grows combinatorially (e.g. discarding 4 from a hand of 10 can exceed
100 leaf states after accounting for all orderings).

The fix enforces a lexicographic order: after discarding resource R,
only resources with index >= R are offered on the next discard step.
A `min_resource` field on the `Discard` phase tracks the lower bound.
A suffix-sum check also prunes resources when the remaining cards from
that index onward cannot reach the discard target, cutting dead-end
branches early.

```
Phase::Discard { player, remaining, roller, min_resource }
```

### Canonical build ordering

See [experiments/commutativity.md](/experiments/commutativity.md) — 8-step FSM
that enforces a canonical action ordering within each turn, eliminating
transpositions without losing any reachable end-of-turn state.

### Dominated action pruning

Some legal actions are strictly dominated and pruned from
`legal_actions`. Like the canonical ordering, disabled during colonist
replay (`canonical_build_order=false`) since real games may include
suboptimal plays and tracked state can diverge.

- **Monopoly/YoP/Road Building in PreRoll**: rolling first gives strictly
  more information. Resources and roads from these cards can't help until
  Main phase, and rolling a 7 after YoP increases discard risk. Only
  Knight remains in PreRoll (blocks opponent production on this roll).
- **Monopoly on zero-count resource**: if the opponent holds 0 of a
  resource, Monopoly gains nothing and wastes the one dev card play per
  turn.
- **Robber on empty tiles**: if any legal tile touches opponent
  buildings, tiles without opponent buildings are suppressed (no steal,
  no opponent production blocked). When friendly robber forces all
  opponent tiles off-limits, all remaining tiles stay available.
- **Road Building with 0 roads left**: wastes the dev card play.
- **Setup settlement below 8 pips**: dominated in 1v1 with 50+ spots.

## Colonist replay

```
https://colonist.io/api/replay/data-from-game-id?gameId=178911848&playerColor=1
```

## Episode 3

- go back to 3 layer GNN, all MLPs 4 layers
- global resource features
- remove all road features
- fix road policy - concatenate node features. helps preserve directionality of roads
- drop gumbel_m to 8. a top catan player is probably only considering ~4 moves, but 8 leaves headroom for the prior to be wrong
- add soft policy target from katago
- way more extensive action commutativity
