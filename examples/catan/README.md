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

During the main phase a player can build settlements, cities, roads,
buy dev cards, play dev cards, and trade — in any order. Many of these
action sequences lead to identical end-of-turn states. Building City A
then City B produces the same state as B then A. Buying a dev card
then building a city produces the same state as the reverse.

The optimization classifies main-phase actions as **ordered** or
**unordered** and enforces a canonical priority among ordered actions.
Resource spending is commutative: if you can afford action A then B,
you can always afford B then A (same total cost). Actions that only
spend resources and don't change the board commute unconditionally.

**Ordered** (must respect priority + lexicographic within type):

| Priority | Action                          | Rationale                       |
| -------- | ------------------------------- | ------------------------------- |
| 0        | Dev card buy                    | No board change                 |
| 1        | City on pre-existing settlement | No connectivity or ratio change |
| 2        | Non-port settlement             | No connectivity or ratio change |

**Unordered** (can happen at any time):

| Action                   | Why unordered                                       |
| ------------------------ | --------------------------------------------------- |
| Road                     | Changes connectivity, can unlock new locations      |
| Port settlement          | Changes trade ratios, can enable new trades         |
| City on same-turn settle | Causal dependency — settlement must exist first     |
| Dev card play            | Changes resources (YoP, Monopoly) or triggers phase |
| Maritime trade           | Changes resources                                   |

Three `u8` fields on `GameState` track the ordering state:
`min_build_type` (which priorities remain available), `min_city_node`
and `min_settle_node` (lexicographic bounds within each type). A
`settlements_at_turn_start` bitmask distinguishes pre-existing
settlements (ordered cities) from same-turn settlements (unordered
cities), preserving the settle-then-upgrade-same-turn line of play.

The ordering activates only inside MCTS simulations (via `determinize`).
The UI and human play always see the full set of legal actions.

### Dominated action pruning

Some legal actions are strictly dominated and pruned from `legal_actions`
unconditionally (not just during search):

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

## Colonist replay

```
https://colonist.io/api/replay/data-from-game-id?gameId=178911848&playerColor=1
```
