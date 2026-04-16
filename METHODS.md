# Methods

Architecture and optimization decisions behind Canopy, with a section
on future ideas at the end. Complements the high-level framing in the
[README](README.md) and the Catan-specific tricks in
[examples/catan/OPTIMIZATIONS.md](examples/catan/OPTIMIZATIONS.md).

## Search architecture

### Types

```rust
enum Status {
    Decision(f32),   // sign: +1 maximizer, −1 minimizer
    Chance,
    Terminal(f32),   // reward from P1's perspective
}

trait Game: Clone + Send + Sync {
    fn status(&self) -> Status;
    fn legal_actions(&self, buf: &mut Vec<usize>);
    fn apply_action(&mut self, action: usize);
    fn state_key(&self) -> Option<u64> { None }              // transposition key
    fn chance_outcomes(&self, buf: &mut Vec<(usize, u32)>) {} // (outcome, weight)
    fn sample_chance(&self, rng) -> Option<usize> { .. }
    fn determinize(&mut self, rng) -> bool { false }         // resample hidden info
}
```

`Status` directly encodes the three node types — no inferring chance
from `chance_outcomes()` being non-empty. All actions are `usize` in
`[0, NUM_ACTIONS)`; chance outcomes reuse the same action space.

`determinize` is called once per simulation on a clone of the root
state before tree descent. It resamples hidden information the
searching player can't see (e.g., opponent dev cards in Catan),
consistent with public observations. Returns `true` if resampling
occurred — search then filters tree edges against `legal_actions()`
during descent, since the tree may contain edges from earlier sims
with different hidden state (Single-Observer ISMCTS).

Self-play training uses full information (`determinize` returns
`false`). The network input features must not include hidden state,
so the network naturally learns the Bayesian-average policy across
many games — equivalent to determinized search but cheaper.
Tournament / live play determinizes each sim to handle real
uncertainty.

### Evaluation

```rust
struct Wdl { w: f32, d: f32, l: f32 }   // P1 perspective
struct Evaluation { policy_logits: Vec<f32>, wdl: Wdl }

trait Evaluator<G: Game>: Send {
    fn evaluate(&self, state: &G, rng) -> Evaluation;
    fn evaluate_batch(&self, states: &[&G], rng) -> Vec<Evaluation> { .. }
}
```

Decoupled from search — the caller sits between `Search` and
`Evaluator`. The search module has no dependency on neural-network
code.

### Search API

```rust
Search<G: Game>::new(state: G, config: Config) -> Self
    .select(rng) -> Select<G>                   // descend tree, apply virtual loss
    .backup(LeafId, Evaluation)                 // propagate eval, remove virtual loss
    .backup_terminal(LeafId, Wdl)               // propagate terminal value
    .result() -> SearchResult                   // extract when done
    .apply_action(usize)                        // reroot tree, clear virtual losses
    .state() -> &G
    .tree() -> &Tree                            // read-only tree access
    .reset(G)                                   // new game, reuse allocs

enum Select<G> {
    Eval(LeafId, G),          // leaf needs network evaluation
    Terminal(LeafId, Wdl),    // terminal node, value known
    Done,                     // budget exhausted
}

struct Config {
    num_simulations: u32,     // budget per search (default 800)
    c_puct: f32,              // PUCT exploration constant (default 2.5)
    c_visit: f32,             // σ scaling for Q influence (default 50.0)
    c_scale: f32,              // σ scaling (default 1.0)
    dirichlet_alpha: f32,     // root noise concentration (~10/num_actions)
    dirichlet_epsilon: f32,   // noise mixing fraction (default 0.25)
}
```

Tree persists across actions for reuse. The `select` / `backup` split
lets the caller own the eval loop — single-threaded or multi-threaded,
batched or not, with no API change.

Caller patterns:

```rust
// single-threaded
loop {
    match search.select(&mut rng) {
        Select::Eval(id, state) => search.backup(id, evaluator.evaluate(&state)),
        Select::Terminal(id, wdl) => search.backup_terminal(id, wdl),
        Select::Done => break,
    }
}

// GPU batched
let mut batch = Vec::new();
loop {
    match search.select(&mut rng) {
        Select::Eval(id, state) => batch.push((id, state)),
        Select::Terminal(id, wdl) => search.backup_terminal(id, wdl),
        Select::Done if batch.is_empty() => break,
        Select::Done => {}
    }
    if batch.len() >= batch_size {
        for ((id, _), eval) in batch.drain(..).zip(evaluator.evaluate_batch(&batch)) {
            search.backup(id, eval);
        }
    }
}

// multi-threaded (Search is single-threaded; wrap in Mutex)
scope(|s| for _ in 0..num_threads {
    s.spawn(|| loop {
        let sel = { lock.lock(); search.select(&mut rng) };
        match sel {
            Select::Eval(id, state) => {
                let eval = evaluator.evaluate(&state);  // no lock held
                lock.lock(); search.backup(id, eval);
            }
            Select::Terminal(id, wdl) => { lock.lock(); search.backup_terminal(id, wdl); }
            Select::Done => break,
        }
    });
});
```

### Selection formulas

**Root** uses PUCT with Dirichlet noise mixed into priors:

```
score = Q + c_puct · P · √N_parent / (1 + N)
P ← (1 − ε) · P_net + ε · Dir(α)
```

**Interior** nodes use the improved-policy formula from Gumbel
AlphaZero:

```
score ∝ softmax(logit + σ(completedQ))
σ(q) = (c_visit + max_visits) · c_scale · q_norm
completedQ(edge) = Q(child) if visited else v_mix
```

Selected action at the end of search is the argmax of the improved
policy at the root. Improved policy is also the training target.

### Internals

**Root expansion.** The first `select()` finds no root, returns
`Select::Eval(id, root_state)`. The subsequent `backup(id, eval)`
creates the root with edges for all legal actions (priors from
softmax of policy logits). After that, `select` descends normally. No
special case in the API — the root is just the first leaf.

**Chance nodes.** `select` handles chance nodes internally during
descent — samples an outcome (weighted random from stored
probabilities, or `state.sample_chance(rng)` when determinized) and
continues descending. No evaluation needed. Chance nodes are
transparent to the caller.

**Tree compaction.** `apply_action` reroots the tree to the child
matching the played action. On the next `select`, unreachable nodes
are compacted away (BFS from new root, remap IDs, drop the rest).
Tree size is bounded by what's reachable from the current root.

**v_mix (FPU).** Value estimate for unvisited children. Interpolates
the node's network value v̂ with the policy-weighted average Q of
visited children:

```
v_mix = (v̂ + N_total · E[q | visited]) / (1 + N_total)
```

Early in search, v_mix ≈ v̂ (pure network prior). As visits
accumulate, it blends toward the empirical Q. Unvisited edges use
v_mix as their Q in both the PUCT score and the improved-policy
formula.

**Virtual loss.** When `select` descends through edges, it increments
an in-flight counter on each edge along the path. This inflates
visit counts, deflecting concurrent selects to different leaves.
`backup` decrements the counters. `apply_action` clears any
outstanding virtual losses before rerooting.

### Parallel search gotchas

- **Node expansion races**: two threads reach the same unexpanded
  leaf. Need CAS on node state or check-and-back-off to prevent
  duplicate expansion. Losing thread abandons its playout.
- **Terminal node flooding**: terminals evaluate instantly (no
  inference), so a thread can rack up many terminal visits while
  others wait on inference. Can skew statistics. KataGo throttles
  terminal-visiting threads to match inference latency.
- **Virtual loss at high thread counts**: stale node statistics cause
  suboptimal exploration. KataGo recommends batching across positions
  (multiple games) over many threads on one tree. For tournament with
  ~8 threads this is manageable.
- **Transpositions + parallelism**: if using the DAG, edge visits must
  be tracked per parent-child pointer, not on the shared child. Also
  need cycle detection (thread keeps a visited-set during descent).

## Design decisions

Each entry notes expected impact on playing strength per unit of
implementation effort.

### Search — PUCT root with improved-policy interior — _high impact_

Root: standard PUCT with Dirichlet noise on priors. Interior nodes:
the improved-policy formula from Gumbel AlphaZero
(`softmax(logit + σ(completedQ))`). Earlier versions of this
framework used Sequential Halving + Gumbel-Top-k at the root; that
was replaced with PUCT because exploration quality at small budgets
is easier to tune with `c_puct` + Dirichlet than with Gumbel noise
scale. Interior improved-policy selection is retained — it keeps the
policy-improvement guarantee and provides a well-defined training
target.

Reference: [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO) (Danihelka et al., ICLR 2022).

### Search — DAG transposition table — _medium impact_

Replaces the search tree with a directed acyclic graph. Different
action sequences reaching the same state share a single node. Games
opt in via `state_key() -> Option<u64>`. Degrades gracefully to a
normal tree when no transpositions exist. Impact depends on game
structure — large for 2048, Catan (with canonical action ordering);
negligible for Go.

### Search — Tree reuse across moves — _medium impact_

After each action, `apply_action` reroots the DAG at the new state
and the next `select` compacts to the surviving subtree. Subsequent
searches start with existing visit counts and Q-values rather than
from scratch. Free speedup proportional to how much of the previous
tree survives — substantial when the opponent plays a move the search
considered heavily.

### Search — Batched leaf evaluation — _high impact_

The state-machine split (`select` pauses at leaves, caller batches,
`backup` resumes) lets multiple concurrent self-play workers share
a single GPU forward pass. Without this, inference is
one-leaf-at-a-time and GPU utilization is near zero. The MCTS module
itself has no evaluator dependency — the caller owns batching
strategy.

### Search — Arena-based tree storage — _low–medium impact_

All nodes in a flat `Vec<Node>`, edges packed contiguously, integer
`NodeId` indices instead of pointers. Better cache locality, zero
per-node allocation, and tree reuse / compaction become simple index
remapping.

### Training — Value target z/q blending — _high impact_

Mixes game outcome `z` with MCTS root Q-value `q`:

```
target = (1 − α) · z + α · q
```

The mixing weight α ramps linearly from 0 to `q_weight_max` over
`q_weight_ramp_iters`. Early in training the value head is weak so
Q is garbage, and game outcome z provides the only real signal
despite its variance. As the network improves, Q becomes a better
per-position target than z because it averages over many simulations
rather than one noisy game result. By the end of the ramp, Q
dominates. Critical for stochastic games like Catan where dice
variance makes pure z extremely noisy throughout training.

### Training — Auxiliary short-term value heads — _medium impact_

Additional value heads trained on exponential moving averages of
future Q-values, providing intermediate-horizon value signals
alongside the main outcome head. For each horizon h:

```
ema = α · Q[t] + (1 − α) · ema,   α = 1 − exp(−1 / h)
```

This gives the network credit for predicting what search will think
a few actions ahead, not just the final outcome. Shares a hidden
layer across horizons with a single multi-output projection.
Controlled by `aux_value_horizons` (e.g. `[4, 10, 30]` for Catan's
~90-move games; empty disables) and `aux_value_weight` (default 0.5
per head).

Reference: [KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
— "Auxiliary Short-Term Value Targets".

### Training — Playout cap randomization — _high impact_

75% of actions use a fast search (small budget); 25% use the full
search. Only full-search positions contribute policy targets; all
positions contribute value targets. Yielded 1.37× throughput
improvement in KataGo — effectively quadruples the number of value
training samples per unit of search compute. A per-sample
`full_search` flag masks policy loss during training.

Reference: [KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
— "Playout Cap Randomization".

### Training — Replay buffer — _low impact_

Sample-capped replay buffer retains the most recent
`replay_buffer_samples` samples. Training triggers on fresh sample
count (`train_samples_per_iter`) rather than game count. Oldest games
evicted when the buffer exceeds capacity.

### Representation — Canonical per-game state — _medium impact (per-game)_

For games with symmetry, the game model maps to a single canonical
state — most commonly by encoding from the current player's
perspective (so state features don't need a player-ID channel).
Eliminates the need for data augmentation and makes transposition
tables more effective. Must be designed into each game's encoder;
retrofitting changes state hashing and invalidates stored training
data. Impact is proportional to the symmetry group size. Implemented
for Catan — see
[examples/catan/OPTIMIZATIONS.md](examples/catan/OPTIMIZATIONS.md).

## Future ideas

### Reanalyze

Keep all self-play positions forever. Periodically re-run search on
old positions with the current network to refresh policy/value
targets. Decouples data generation from target quality.

Current flow:

```
self-play → samples → replay buffer (capped) → train
```

With reanalyze:

```
self-play → position store (permanent, outcomes only)
                 │
                 ▼
           reanalyze worker (re-searches with latest net)
                 │
                 ▼
             sample pool → train
```

Each training batch mixes fresh self-play samples with reanalyzed
samples (Will used 50/50). Reanalyze searches can be cheaper than
self-play searches since we only need the root policy/value, not a
full trajectory.

Decisions to make:

- **Reanalyze budget**: sims per reanalyzed position (can be less
  than self-play).
- **Mix ratio**: fraction of each batch from reanalyze vs fresh.
- **Staleness**: how often to re-search the same position
  (diminishing returns as net improves less between iterations).
- **Storage**: position store is just features + outcome z, no search
  targets.

Playout cap randomization is a natural complement — the cheap
fast-searches are good reanalyze candidates since they're fast to
re-search.

Reference: MuZero paper Appendix H. Will's TakZero: 50M positions,
50/50 mix, 4× target reuse.

### Lazy chance node expansion

Chance nodes are the main depth bottleneck. Each dice roll fans out
11 ways. Deep in the tree, a chance node with 20 visits expands into
11 children that each get ~2 visits — not real depth, just 11
network evaluations that approximate what a single pre-chance
evaluation already knows.

The idea: don't expand chance nodes into full subtrees until they
have enough visits to justify it. Below the threshold, use a cheap
baseline value. Above it, search selectively.

**Level 1 — single cached eval.** Cache the network's value at the
pre-chance state on the first visit. The network already sees deck
state / probability information, so its value is an implicit average
over outcomes. Below threshold, back up the cached value without
re-evaluating. Threshold: expand when `visits > num_outcomes · K`
(K = 2–4).

**Level 2 — exact baseline + selective deepening.** At a chance node
with outcome probabilities p(o):

1. Enumerate outcomes, apply transitions to get successor states.
2. Batch-evaluate all successors (one forward pass).
3. Compute exact baseline `μ = Σ p(o) · v_net(s_o)`.
4. Search deeply only into a small subset E.
5. Back up: `V = μ + Σ_{o ∈ E} p(o) · (V_search(s_o) − v_net(s_o))`.

Keeps balanced-dice accuracy (all outcomes included exactly) while
concentrating search budget where search corrections matter.
Prioritize outcomes by `p(o) · |v_net(s_o) − μ|`.

This belongs in the MCTS framework, not game-specific code. The
`Game` trait already distinguishes chance from decision and provides
outcome probabilities.

Open questions:

- Level 1 vs Level 2 cost tradeoff on CPU inference.
- Training impact of fewer deep evaluations but more breadth at
  decision points.
- Selective deepening criteria beyond p · deviation.

Reference: Stochastic MuZero (Antonoglou et al., 2022) formalizes
"afterstates" — post-decision, pre-chance states whose values are
used directly.

### Dice bucketing (shelved)

Group dice outcomes that produce identical resource effects into a
single tree edge — "what happened to the board" rather than "which
number rolled". The approximation: different outcomes within a
bucket leave different deck states, so the tree's value estimate
averages across them.

Investigated for Catan and shelved: after setup, both players touch
6–8 unique numbers, so bucketing only merges the 2–3 numbers that
produce nothing. Reduction is too small to meaningfully deepen
search. Lazy chance expansion is the more promising direction.

## Related reading

- [Training configuration guide](notes/training-config.md) —
  parameter-by-parameter reference for tuning a training run.
- [1v1 Catan strategy notes](notes/strategy-doc.md) — human-strategy
  guide distilled from colonist.io material, useful context for
  model/encoder design decisions.
- [examples/catan/OPTIMIZATIONS.md](examples/catan/OPTIMIZATIONS.md)
  — Catan-specific action-space and state-space optimizations.
