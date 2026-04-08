# Canonical Action Ordering

Within a turn, many action sequences lead to the same end-of-turn state. A mid-game turn with 2 trades + buy dev + city + 2 roads = 6 actions has 720 permutations, almost all transpositions.

Canonical ordering eliminates these transpositions, giving search more depth per sim. It also makes the policy head's job easier — fewer choices at each decision point means the model can more clearly distinguish relative quality.

## Ordering

Force a strict ordering on action types within a turn. Once you take an action in category N, categories < N are locked out. Within each category, actions are lexicographically ordered — once you take action with key X, only actions with key >= X remain (allows repeating the same action, e.g., same maritime trade twice).

### 1. Play dev card

Dev cards are the only main-phase action that can expand your resources or action space (Knight steals, YOP picks resources, Monopoly takes all of one type, Road Building gives 2 free roads). Playing a dev card never restricts future actions — it only adds options. So it should always be first: you want the extra resources/roads available before deciding how to spend.

One per turn (game rule), so no within-type ordering needed.

### 2. Maritime trade

After dev card play, the player's resource pool is finalized from external sources. Trading rearranges the hand without changing the board. Since all subsequent steps only spend resources, trading first ensures you have the exact hand needed for all planned builds. The end-of-turn state is the same regardless of when trades happen relative to builds — total resources spent are identical.

Ordered by canonical trade index (give-resource, receive-resource pair) to eliminate within-type transpositions.

### 3. Buy dev card

Pure resource expenditure that doesn't change the board, trade ratios, or piece availability. Commutes with every other spending action.

**Known limitation**: buying a dev card is followed by a chance node (card draw). Placing it after trades means the search can't condition trade decisions on the drawn card. In theory, buying before trading gives the search more information (e.g., drawing a VP at 14 VP wins immediately without needing to trade). In practice, this would require a second buy-dev slot (pre-trade and post-trade) introducing a 2x transposition and extra complexity. We accept the minor search quality loss for a cleaner ordering.

### 4. City on pre-existing settlement

Cities are split into pre-existing vs same-turn because they have different causal properties. A city on a pre-existing settlement is a pure upgrade — it doesn't depend on anything built this turn. Importantly, upgrading a city **reclaims a settlement piece**, which may be needed for steps 6–8. If a player has all 5 settlements on the board, they must city first to free a piece before building a new settlement. Ordering cities before settlements (step 4 < step 6) preserves this dependency.

By node ID ascending to eliminate within-type transpositions. Uses `settlements_at_turn_start` bitmask to distinguish pre-existing from same-turn.

### 5. Build road

Roads extend the network frontier, potentially enabling settlement spots at step 6. Ordering roads before settlements preserves this causal dependency. Settlements don't extend road connectivity (`reachable_edges` is computed from roads only), so there's no reverse dependency.

Ordered by (distance to network, edge ID) ascending. Distance = hops from current frontier, computed via BFS. Encode as `distance * 72 + edge_id`. This handles the case where road A extends the frontier to make road B reachable — A has lower distance and sorts first, preventing the ordering from blocking a valid chain. Frontier edges (distance 0) are independently buildable and commute; edge ID breaks ties.

`road_distances` must be recomputed when returning to Main from Road Building phase (free roads extend the frontier, invalidating distances computed at turn start).

The Road Building phase itself should also apply `(distance, edge_id)` ordering to the 2 free roads. If both are frontier edges they commute, so ordering eliminates the 2x transposition.

### 6. Build settlement on non-port

By this point, all roads are built and the network is fully extended. All legal settlement spots are determined by the road network, so settlements at different nodes are independently placeable and commute. Non-port settlements don't change trade ratios or enable any new action types.

By node ID ascending to eliminate within-type transpositions.

### 7. City on same-turn settlement

Building a settlement then upgrading it to a city in the same turn is a real competitive play worth 3 VP (costs B+L+W+2G+3O total). This has a hard causal dependency on step 6 — you can't upgrade a settlement that doesn't exist yet. So same-turn cities are exempt from the strict type ordering and always available after their settlement is built.

**Known limitation**: upgrading a same-turn settlement reclaims a piece, which could theoretically enable another settlement. But this requires 12+ resources in one turn and upgrading a brand-new settlement instead of one of 4 higher-quality existing ones — virtually never optimal. The common reclaim path (city on pre-existing at step 4 → settle at step 6) is handled correctly. We do not reset `min_action_type` or `min_settle_node` at step 7.

### 8. Build settlement on port

Port settlements change maritime trade ratios, which means subsequent trades produce different results. This is the one case where a build action changes the available action space for a non-build action. After building on a port, the player should be able to trade at the new rate and then continue building.

Resets `min_action_type` to step 2, `min_trade_idx` to 0, and `min_port_settle_node` to 0. Port settlements don't commute with each other when trades happen between them — building port X first enables trades at X's rate before building port Y, producing a different end state than the reverse. Other within-type constraints (road, settle, city) persist since they're unaffected by trade ratios.

### End turn

Always available.

## Implementation

No reachable end-of-turn state is lost — the pruned orderings all reach identical states. The ordering is active by default (search, UI, bot play) and disabled during colonist replay (`canonical_build_order = false`) where the event log may use non-canonical orderings.

State fields on `GameState` (reset each turn in `apply_end_turn`):

- `min_step: u8` — lowest allowed category (1–8)
- `min_trade_idx: u8` — within trades, only pairs >= this
- `min_city_node: u8` — within pre-existing cities, only nodes >= this
- `min_road_key: u16` — within roads, only keys >= this (`distance * 72 + edge_id`)
- `min_settle_node: u8` — within non-port settlements, only nodes >= this
- `min_port_settle_node: u8` — within port settlements, only nodes >= this
- `road_distances: [u8; 72]` — BFS from frontier, computed at turn start and after Road Building
- `settlements_at_turn_start: u64` — bitmask for pre-existing vs same-turn
