# Search Architecture

## Core types

### Game trait

```rust
enum Status {
    Decision(f32),   // sign: +1 maximizer, -1 minimizer
    Chance,
    Terminal(f32),   // reward from P1's perspective
}

trait Game: Clone + Send + Sync {
    fn status(&self) -> Status;
    fn legal_actions(&self, buf: &mut Vec<usize>);
    fn apply_action(&mut self, action: usize);
    fn state_key(&self) -> Option<u64> { None }       // transposition key
    fn chance_outcomes(&self, buf: &mut Vec<(usize, u32)>) {}  // (outcome, weight)
    fn sample_chance(&self, rng) -> Option<usize> { .. }
    fn determinize(&mut self, rng) -> bool { false }    // resample hidden info; true = filter legal
}
```

`Status` directly encodes the three node types — no inferring chance from `chance_outcomes()` being non-empty. All actions are `usize` in `[0, NUM_ACTIONS)` — chance outcomes reuse the same action space.

`determinize` is called once per simulation on a clone of the root state before tree descent. It resamples hidden information the observer can't see (e.g., opponent dev cards in Catan), consistent with public observations. Returns `true` if resampling occurred — search then filters tree edges against `legal_actions()` during descent, since the tree may contain edges from previous sims with different hidden state.

Self-play training uses full info (`determinize` returns `false`). The network input features must not include hidden state (e.g., opponent card identities). Since the network can't see hidden info, it learns the Bayesian-average policy naturally across many games — equivalent to determinized search but cheaper. Tournament/live play determinizes each sim to handle real uncertainty.

### Evaluation

```rust
struct Wdl { w: f32, d: f32, l: f32 }  // P1 perspective

struct Evaluation { policy_logits: Vec<f32>, wdl: Wdl }

trait Evaluator<G: Game>: Send {
    fn evaluate(&self, state: &G, rng) -> Evaluation;
    fn evaluate_batch(&self, states: &[&G], rng) -> Vec<Evaluation> { .. }
}
```

Decoupled from search — the caller sits between `Search` and `Evaluator`.

### Search API

Tree persists across actions for reuse. `select`/`backup` split lets the caller own the eval loop — single-threaded or multi-threaded, batched or not, with no API change.

```rust
Search<G: Game>::new(state: G, config: Config) -> Self
  .select(rng) -> Select<G>                   // descend tree, apply virtual loss
  .backup(LeafId, Evaluation)                  // propagate eval, remove virtual loss
  .backup_terminal(LeafId, Wdl)                // propagate terminal value, remove virtual loss
  .result() -> SearchResult                    // extract result when done
  .apply_action(usize)                         // reroot tree, clear virtual losses
  .state() -> &G
  .tree() -> &Tree                             // read-only tree access
  .reset(G)                                    // new game, reuse allocs

enum Select<G> {
    Eval(LeafId, G),           // leaf needs network evaluation
    Terminal(LeafId, Wdl),     // terminal node, value known
    Done,                      // budget exhausted
}

struct SearchResult {
    policy: Vec<f32>,          // improved policy (training target)
    wdl: Wdl,             // root WDL, P1 perspective
    selected_action: usize,
    network_value: f32,        // raw net value before search
    children_q: Vec<(usize, f32)>,
    prior_top1_action: usize,
    pv_depth: u32,
    max_depth: u32,
}

// Tree is read-only to callers. Arena-based: flat Vec<Node>, Vec<Edge>,
// DAG transpositions via state_key(). Callers walk it directly for UI.
Tree
  .root() -> NodeId                            // always valid after first select()
  .edges(NodeId) -> &[Edge]
  .q(NodeId) -> f32                            // W - L
  .wdl(NodeId) -> Wdl
  .kind(NodeId) -> &NodeKind                   // Decision(sign) | Chance | Terminal
  .max_edge_visits(NodeId) -> u32

Edge { action, child: Option<NodeId>, prior, logit, visits }

struct Config {
    num_simulations: u32,       // budget per search (default 800)
    c_puct: f32,                // PUCT exploration constant (default 2.5)
    c_visit: f32,               // σ scaling for Q influence (default 50.0)
    c_scale: f32,               // σ scaling (default 1.0)
    dirichlet_alpha: f32,       // root noise (~10/num_actions, ~0.05 for Catan)
    dirichlet_epsilon: f32,     // noise mixing fraction (default 0.25)
}
```

Caller patterns:

```rust
// single-threaded (training, self-play)
loop {
    match search.select(&mut rng) {
        Select::Eval(id, state) => search.backup(id, evaluator.evaluate(&state)),
        Select::Terminal(id, wdl) => search.backup_terminal(id, wdl),
        Select::Done => break,
    }
}
let result = search.result();

// single-threaded, GPU batched (self-play with GPU)
let mut batch = Vec::new();
loop {
    match search.select(&mut rng) {
        Select::Eval(id, state) => batch.push((id, state)),
        Select::Terminal(id, wdl) => search.backup_terminal(id, wdl),
        Select::Done if batch.is_empty() => break,
        Select::Done => {}
    }
    if batch.len() >= batch_size || matches!(/* done */) {
        let evals = evaluator.evaluate_batch(&batch);
        for ((id, _), eval) in batch.drain(..).zip(evals) {
            search.backup(id, eval);
        }
    }
}

// multi-threaded (tournament, CPU inference)
// Search is single-threaded; caller wraps in Mutex.
// Lock held only during select/backup (microseconds).
// Eval runs per-thread with no lock (milliseconds).
scope(|s| {
    for _ in 0..num_threads {
        s.spawn(|| loop {
            let sel = { lock.lock(); search.select(&mut rng) };
            match sel {
                Select::Eval(id, state) => {
                    let eval = evaluator.evaluate(&state);
                    lock.lock(); search.backup(id, eval);
                }
                Select::Terminal(id, wdl) => {
                    lock.lock(); search.backup_terminal(id, wdl);
                }
                Select::Done => break,
            }
        });
    }
});
let result = search.result();
```

No snapshot types in the search module. The server/presenter layer walks `tree()` directly to build whatever JSON the UI needs, computing improved policy from edge logits and Q values.

PUCT at root: `Q + c_puct * P * sqrt(N_parent) / (1 + N)` with Dirichlet noise on root priors. Virtual loss inflates visit count for in-flight selections, naturally deflecting concurrent threads to different leaves. Budget is a simple counter; callers can change `config.num_simulations` between searches freely.

Improved-policy interior selection (`softmax(logit + σ(completedQ))`) at non-root nodes, shared across all node types. Result extraction uses improved policy (same formula). Selected action is argmax of improved policy.

### Internals

**Root expansion**: the first `select()` finds no root node, returns `Select::Eval(id, root_state)`. The subsequent `backup(id, eval)` creates the root with edges for all legal actions (priors from softmax of policy logits). After that, `select` descends normally. No special case in the API — the root is just the first leaf.

**Chance nodes**: `select` handles chance nodes internally during descent — samples an outcome (weighted random from stored probabilities, or `state.sample_chance(rng)` when determinized) and continues descending. No evaluation needed. Chance nodes are transparent to the caller.

**Tree compaction**: `apply_action` reroots the tree to the child matching the played action. On the next `select`, unreachable nodes are compacted away (BFS from new root, remap IDs, drop the rest). Tree size is bounded by what's reachable from the current root, not cumulative history. Old branches are freed every move.

**v_mix (FPU)**: value estimate for unvisited children. Interpolates the node's network value (v̂) with the policy-weighted average Q of visited children: `v_mix = (v̂ + N_total * E[q|visited]) / (1 + N_total)`. Early in search, v_mix ≈ v̂ (pure network prior). As visits accumulate, it blends toward the empirical Q. Unvisited edges use v_mix as their Q in the PUCT and improved-policy formulas.

**Virtual losses**: when `select` descends through edges, it increments an in-flight counter on each edge along the path. This inflates visit counts, deflecting concurrent selects to different leaves. `backup` decrements the counters. `apply_action` clears any outstanding virtual losses before rerooting (from abandoned selects that never got a backup).

### Parallel search gotchas

Node expansion races: two threads reach the same unexpanded leaf. Need CAS on node state or check-and-back-off to prevent duplicate expansion. Losing thread abandons its playout.

Terminal node flooding: terminals evaluate instantly (no inference), so a thread can rack up many terminal visits while others wait on inference. Can skew statistics. KataGo throttles terminal-visiting threads to match inference latency.

Virtual loss weakens play at high thread counts — stale node statistics cause suboptimal exploration. KataGo recommends batching across positions (multiple games) over many threads on one tree. For tournament with ~8 threads this is manageable.

Transpositions + parallelism: if using DAG (state_key), edge visits must be tracked per parent-child pointer, not on the shared child node. Also need cycle detection (thread keeps visited-set during descent).
