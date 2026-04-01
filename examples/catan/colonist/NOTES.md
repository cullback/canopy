# Colonist integration notes

Live-play integration between the canopy MCTS engine and colonist.io via CDP.

## Architecture: single action stream

Every state change must flow through the game engine via action IDs. The engine
handles all side effects (costs, road networks, longest road, phase transitions,
`on_opponent_build`). Direct mutation of `GameState` fields produces subtle bugs
because it bypasses side effects the engine would apply.

### Data sources

Each poll queries two things from colonist via CDP:

1. **DOM board**: settlements/cities/roads read from the rendered board. Always
   has coordinates. Authoritative for _what_ was built and _where_.

2. **Event log**: the colonist action log. Carries event types, player colors,
   and sometimes coordinates. Authoritative for _ordering_, _dice outcomes_,
   _dev card plays_, _resource transfers_, and _steal results_.

### Deriving actions

The poll must produce a `Vec<usize>` of engine action IDs from these sources.
Almost every game event maps to an engine action:

| Event                     | Engine action                                                |
| ------------------------- | ------------------------------------------------------------ |
| Roll                      | `ROLL` + dice chance outcome                                 |
| BuildRoad                 | `road_id(eid)` from coords                                   |
| BuildSettlement           | `settlement_id(nid)` from coords                             |
| BuildCity                 | `city_id(nid)` from coords                                   |
| PlaceRoad (Road Building) | `road_id(eid)` — resolve from DOM diff if event lacks coords |
| PlayedKnight              | `PLAY_KNIGHT` + `MoveRobber` + steal chance                  |
| PlayedRoadBuilding        | `PLAY_ROAD_BUILDING`                                         |
| PlayedYearOfPlenty        | `yop_id(r1, r2)`                                             |
| PlayedMonopoly            | `monopoly_id(r)`                                             |
| MoveRobber                | `move_robber_id(tile)`                                       |
| Stole                     | steal chance outcome (resource index 0–4)                    |
| Discard                   | `discard_id(resource)`                                       |
| BankTrade                 | `maritime_id(give, recv)`                                    |
| EndTurn                   | `END_TURN` (inferred from turn change)                       |

**One exception**: opponent `BuyDevCard`. Card identity is hidden. Use
`apply_hidden_dev_card_buy` (direct mutation, increments `hidden_dev_cards`).

### Coordinate resolution

Event log coordinates may be missing (especially type 4 "place" events during
Road Building). The DOM board always has coordinates. To resolve:

1. Diff DOM board against `committed_state` bitmasks to find new buildings
2. Convert positions to action IDs via `road_id`/`settlement_id`/`city_id`
3. Match against events by type

When multiple buildings appear in one poll, find a valid ordering by scanning
for currently-legal actions and applying them one at a time. The final state may
be a transposition of the actual play order — this is fine.

### What NOT to do

Never mutate `GameState` fields directly for actions the engine can handle.
Past bugs from direct mutation:

- **Missing `on_opponent_build`**: `place_settlement` didn't break opponent's
  road continuity. Opponent's `longest_len` cache went stale.
- **Missing `settlements_left += 1`**: `place_city` didn't return the settlement
  piece.
- **Missing road network update**: silent fallback on unmapped coords just
  subtracted resources. Road not added to network, no walk action pushed, tree
  edges stale, new settlement spots invisible to search.
- **Double-placement**: DOM sync and event processing both placed the same
  building, double-counting `roads_placed`/`roads_left`.

## Search tree vs game state

The search tree and committed game state are separate objects. Tree nodes store
edges (legal actions) computed at creation time. Updating the game state does
not update the tree's edges.

### `set_final_state` vs `reset_to_state`

`set_final_state(state)` overwrites `search.root_state` but preserves the tree.
The root node keeps its original edges. Use when `walk_tree` succeeded.

`reset_to_state(state)` clears the tree and sets new state. Next search rebuilds
the root with fresh `legal_actions()`. Use when:

- `walk_tree` partially failed (walked < total actions)
- State changed but produced 0 walk actions

**Bug**: calling `set_final_state` after a road build left the tree with old
edges — `Settle N34` was legal in the state but absent from the tree. Search
never explored it.

### `walk_tree` and unexpanded edges

`child_for_action` returns `None` both when the edge doesn't exist and when it
exists but was never visited (no child node). With few sims across many edges,
some stay unexpanded. Tree resets correctly — no subtree to preserve.

## Turn/phase transitions

### Event-driven, not DOM-driven

`current_player` and `phase` must change through the engine, not from DOM reads.

**Bug**: `apply_live_state` mutated `current_player` from colonist's `turnState`
before events arrived. `process_new_events` then skipped `END_TURN` generation
(player already matched). Walk actions came out as `[ROLL, dice]` instead of
`[END_TURN, ROLL, dice]`. Tree couldn't reroot.

**Fix**: `apply_live_state` only logs DOM state. Turn/phase changes flow
exclusively through event processing.

### Proactive END_TURN

Colonist's DOM shows turn changes (`turnState=1`) before the event log catches
up. When `turnState=1` and `committed_state.current_player` doesn't match,
proactively apply `END_TURN` through the engine so search starts exploring the
next player's roll immediately. Fires for both directions.

### Phase::Roll mapping

`Phase::Roll` is an internal chance node for auto-resolving dice. In colonist
mode dice come from events, so `apply_live_state` maps `Phase::Roll` back to
`Phase::PreRoll` or `Phase::Main` based on `state.pre_roll`.

## Colonist React state

Extracted via CDP JavaScript on React fiber tree.

- `turnState`: 1 = pre-roll, 2 = post-roll/main
- `actionState`: 0 = default, 4 = ?, 6 = build settlement, 24 = place robber
- `localColor`: browser session owner's color
- Dev cards: `{cards: [11,...], bought_this_turn: [14,...]}`.
  Enums: 11=Knight, 12=VP, 13=Monopoly, 14=RoadBuilding, 15=YearOfPlenty.

### Dev card inventory

`apply_dev_cards` sets the local player's `dev_cards` from React (authoritative).
`has_played_dev_card_this_turn` derived from event log via
`played_dev_card_this_turn()`. Opponent cards tracked as `hidden_dev_cards`.

## Search-based replay (`try_replay`)

Initial connection and live polling both use `try_replay` — a single recursive
function that replays events through engine actions with backtracking.

### Gotchas

- **`ensure_player`**: pre-roll actions (Knight, dev cards) from the next player
  arrive before their Roll event. Must inject `END_TURN` when event player
  doesn't match `current_player`. Call before every player-specific action.

- **`reveal_hidden_card`**: `apply_hidden_dev_card_buy` puts cards in
  `hidden_dev_cards`. The engine's `apply_play_knight` etc. check `dev_cards`
  which is 0 — `saturating_sub(1)` silently no-ops. Must move the card from
  hidden to revealed before playing.

- **Robber: last move uses DOM**: the last `MoveRobber` in the event slice uses
  the DOM robber position directly (authoritative). Only intermediate moves need
  inference via TileBlocked + steal outcome + distribution filters + backtracking.

- **Robber: empty candidates → backtrack**: if all robber candidates are filtered
  out, return `None` to force the search to reconsider an earlier placement.
  Previously silently continued with the wrong robber.

- **Bank trades: multi-resource decomposition**: colonist can combine trades
  (e.g. `L L L L B B → G G`). Decompose by iterating received resources and
  matching against given using `trade_ratios`.

- **Friendly robber uses `building_vps`**: not `public_vps`. Largest army and VP
  dev cards don't count. Threshold is `>= 3`.

- **Setup events during polling**: `sync_buildings` handles placement from DOM.
  `replay_events` must NOT re-process setup events. Apply `StartingResources`
  directly from the log.

- **State from `try_replay`**: returns final state explicitly. Don't extract from
  last timeline entry — events like MoveRobber+StoleNothing produce 0 entries.

- **Retry on failure**: when `replay_events` fails, don't advance
  `committed_event_count`. Next poll retries with more events (gives robber
  inference validation data from future rolls).

- **Type 139**: colonist steal event (robber's perspective), same structure as
  type 15. Parse with `15 | 139 =>`.

## Budget model

`sims_budget` refills when `state_changed` (new timeline entries or walk
actions). Not based on visit count comparison — that failed when a build event
produced a timeline entry but 0 walk actions.

## Gumbel logits guard

After tree reroot, `gumbel.root_logits` length may differ from current edges.
`snapshot()` must filter on `gs.root_logits.len() == edges.len()` before
indexing.
