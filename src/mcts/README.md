# MCTS

Gumbel AlphaZero MCTS with DAG-based search, leaf parallelism, and tree reuse. Implements the search algorithm from [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO) (Danihelka et al., ICLR 2022).

## Public API

### `Search<G: Game>` — state-machine driver

```rust
Search::new(root_state: G, config: Config) -> Self
search.state() -> &G
search.set_num_simulations(n: u32)
search.apply_action(action: usize)
search.supply(evals: &[Evaluation], rng: &mut Rng) -> Step<'_, G>
```

The search tree persists across moves. Call `apply_action` to mirror game actions — if the tree has an expanded child for that action, the root pointer follows it in O(1). The next search compacts the tree to the surviving subtree. If the child wasn't expanded, the tree is discarded and a fresh search starts.

### `Step` — state-machine output

```rust
enum Step<'a, G: Game> {
    NeedsEval(&'a [G]), // leaf states that need evaluation
    Done(SearchResult),  // search complete
}
```

### `SearchResult` — search output

```rust
struct SearchResult {
    pub policy: Vec<f32>,              // improved policy over [0, NUM_ACTIONS) — training target
    pub value: f32,                    // root value estimate (P1 perspective)
    pub selected_action: usize,        // action chosen by Sequential Halving
    pub network_value: f32,            // raw network value before search corrections
    pub children_q: Vec<(usize, f32)>, // (action, Q) for visited root children
    pub prior_top1_action: usize,      // network policy argmax
}
```

### `Config` — search parameters

```rust
struct Config {
    pub num_simulations: u32,       // simulation budget (default 800)
    pub num_sampled_actions: u32,   // Gumbel-Top-k action count at root (default 16)
    pub c_visit: f32,              // σ scaling for Q influence (default 50.0)
    pub c_scale: f32,              // σ scaling (default 1.0)
    pub leaf_batch_size: u32,      // leaves per eval batch (default 1)
}
```

## Usage

`supply` is the only stepping function. Pass an empty slice to start a search; pass evaluations on subsequent calls. `NeedsEval` borrows the leaf states directly.

```rust
let mut search = Search::new(state, config);
let mut evals = vec![];
loop {
    match search.supply(&evals, &mut rng) {
        Step::NeedsEval(states) => {
            evals = evaluate(states);
        }
        Step::Done(result) => break result,
    }
}
search.apply_action(result.selected_action);
// next search: just loop supply again (tree is reused)
```

The caller controls batching across multiple concurrent searches — the neural evaluator can combine leaf requests from parallel self-play workers into a single forward pass. The MCTS module has no dependency on `Evaluator`; evaluation is entirely the caller's responsibility.

## Key optimizations

> **Gumbel AlphaZero (replacing PUCT entirely)**

Replaces PUCT selection, Dirichlet noise, temperature-based action sampling, and visit-count policy targets with a unified Gumbel-based framework. Root uses Sequential Halving with Gumbel noise; non-root uses deterministic improved-policy selection. Guarantees policy improvement even at 2 simulations. Touches the MCTS selection loop, root action selection, policy target computation, and the training loop — essentially every module in the search-and-train pipeline.

> **DAG-based graph search instead of a tree**

Replaces the tree with a directed acyclic graph, enabling transposition tables where different move sequences reaching the same board state share a single node. Used by KataGo since v1.12. Degrades gracefully to normal MCTS if no transpositions exist. Games opt in by implementing `state_key() -> Option<u64>`. Backpropagation runs up the current path only (standard approach — full DAG-aware backprop is expensive).

> **Resumable search with batched NN evaluation**

Splits the MCTS loop so that search pauses at leaf nodes, batches multiple pending evaluations into a single GPU forward pass, then resumes. The `supply`/`pending_states` state machine makes this the natural API rather than a retrofit. `leaf_batch_size` controls how many leaves to collect before yielding.

> **Efficient arena-based tree data structure with tree reuse**

All nodes live in a flat `Vec` arena, edges in a packed contiguous `Vec`, with integer `NodeId` indices instead of pointers. Tree reuse works by rerooting the DAG after each move via `apply_action` + `compact` rather than discarding and rebuilding. Compaction is O(reachable nodes), typically much smaller than the full tree.

> **Leaf parallelism via virtual loss**

When `leaf_batch_size > 1`, virtual losses prevent multiple in-flight simulations from selecting the same path. Each in-flight edge gets a pessimistic Q bias (`-player.sign()`) that is removed once the real evaluation returns. Round-robin assignment at the root distributes simulations evenly across Sequential Halving candidates.

> **Value target mixing**

Mixes the actual game outcome `z` with the MCTS root Q-value `q` as the training target: `target = α·z + (1-α)·q`, where α linearly falls from 1.0 to 0.0 over ~20 generations. Critical for Catan where dice variance makes pure `z` noisy. Requires storing `(state, π, z, q)` tuples in training samples from day one.

## Module layout

| File      | Contents                                                                                                           |
| --------- | ------------------------------------------------------------------------------------------------------------------ |
| `mod.rs`  | `Search`, `Config`, `Step`, `SearchResult`, Gumbel Sequential Halving, simulation loop, improved policy extraction |
| `tree.rs` | `Tree` arena, `NodeData`, `Edge`, transposition table, `compact`, `backprop`, `recompute_q`                        |
