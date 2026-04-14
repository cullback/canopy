# Search & Training Architecture Ideas

## Core types

### Game trait

```rust
enum Status { Ongoing, Terminal(f32) }  // reward from P1's perspective

trait Game: Clone + Send + Sync {
    const NUM_ACTIONS: usize;
    fn status(&self) -> Status;
    fn legal_actions(&self, buf: &mut Vec<usize>);
    fn apply_action(&mut self, action: usize);
    fn current_sign(&self) -> f32 { 1.0 }            // +1 maximizer, -1 minimizer
    fn state_key(&self) -> Option<u64> { None }       // transposition key
    fn chance_outcomes(&self, buf: &mut Vec<(usize, u32)>) {}  // (outcome, weight)
    fn sample_chance(&self, rng) -> Option<usize> { .. }
    fn determinize(&mut self, rng) {}                  // resample hidden info per sim
}
```

Three node types emerge from the trait: **decision** (legal_actions non-empty, chance_outcomes empty), **chance** (chance_outcomes non-empty), **terminal** (Status::Terminal). All actions are `usize` in `[0, NUM_ACTIONS)` — chance outcomes reuse the same action space.

### Evaluation

```rust
struct Evaluation { policy_logits: Vec<f32>, wdl: [f32; 3] }

trait Evaluator<G: Game>: Send {
    fn evaluate(&self, state: &G, rng) -> Evaluation;
    fn evaluate_batch(&self, states: &[&G], rng) -> Vec<Evaluation> { .. }
}
```

Decoupled from search — the caller sits between `Search` and `Evaluator`, batching leaf states from `NeedsEval` into `evaluate_batch`.

### Search API

State-machine search driven by the caller. Tree persists across actions for reuse.

```rust
Search<G: Game>::new(state: G, config: Config) -> Self
  .step(&[Evaluation], &mut Rng) -> Step<G>   // only stepping fn
  .apply_action(usize)                         // mirror game actions
  .state() -> &G
  .snapshot() -> Option<SearchSnapshot>        // live UI data
  .snapshot_subtree(max_depth) -> Option<TreeNodeSnapshot>
  .snapshot_at_path(&[usize], max_depth) -> Option<TreeNodeSnapshot>
  .set_num_simulations(u32)
  .set_filter_legal(bool)                      // SO-ISMCTS toggle
  .cancel_search()                             // clean up virtual losses
  .reset(G)                                    // new game, reuse allocs
  .walk_tree(&[usize]) -> usize                // advance tree without state
  .update_state(FnOnce(&mut G))                // patch state without clearing tree

enum Step<'a, G> { NeedsEval(&'a [G]), Done(SearchResult) }

struct SearchResult {
    policy: Vec<f32>,          // improved policy (training target)
    wdl: [f32; 3],             // root WDL, P1 perspective
    selected_action: usize,
    network_value: f32,        // raw net value before search
    children_q: Vec<(usize, f32)>,
    prior_top1_action: usize,
    pv_depth: u32,
    max_depth: u32,
}

struct Config {
    num_simulations: u32,       // budget per search (default 800)
    num_sampled_actions: u32,   // Gumbel top-k at root (default 6)
    c_visit: f32,               // σ scaling for Q influence (default 50.0)
    c_scale: f32,               // σ scaling (default 1.0)
    leaf_batch_size: u32,       // leaves per eval batch (default 1)
    gumbel_scale: f32,          // noise scale (default 1.0; 0.0 for perfect-info)
    filter_legal: bool,         // SO-ISMCTS interior filtering
}
```

Caller loop: `step(&[]) → NeedsEval → evaluate → step(&evals) → ... → Done`.

Internals: arena-based tree (flat `Vec<Node>`, `Vec<Edge>`), DAG transpositions via `state_key()`, virtual losses for leaf parallelism, improved-policy interior selection (`softmax(logit + σ(completedQ))`), v_mix for FPU.

## Root policy trait

Extract root selection strategy from `Search<G>` into `Search<G, P: RootPolicy = Gumbel>`.

```rust
pub trait RootPolicy: Default + Sized {
    /// root_sign: Some(sign) for decision roots, None for chance.
    fn init(
        &mut self, tree: &Tree, root: NodeId,
        root_sign: Option<f32>, root_value: f32,
        config: &Config, rng: &mut fastrand::Rng,
        legal_edges: Option<Vec<usize>>,
    );
    fn is_done(&self) -> bool;
    fn select_root_edge(&self, tree: &Tree, root: NodeId) -> Option<usize>;
    fn q_bounds(&self) -> (f32, f32);
    fn after_sim(&mut self, tree: &Tree, path: &[(NodeId, usize)], root: NodeId, config: &Config);
    fn extract_result<G: Game>(
        &self, tree: &Tree, root: NodeId, config: &Config,
        network_value: f32, pv_depth: u32, max_depth: u32,
    ) -> SearchResult;
    fn improved_policy(&self, tree: &Tree, root: NodeId, config: &Config) -> Option<Vec<f32>>;
}
```

`Gumbel` implements this with an internal enum for Decision (full SH) vs Chance (budget countdown) vs Empty (pre-init). Policy owns all root state — no `Option<P>` or `vanilla_*` fields on Search. `begin_search` always calls `init`; `run_simulations` and `run_vanilla_sims` merge into one loop dispatching through the trait.

Interior selection (`interior_select`, formerly `gumbel_interior_select`) stays shared — it's the same improved-policy formula for all policies.

## PUCT alternative

Second `RootPolicy` impl. `select_root_edge` computes PUCT scores with Dirichlet noise and returns argmax. `extract_result` uses improved policy (same `logit + σ(completedQ)` formula). `after_sim` just widens Q bounds and decrements budget.

```rust
pub struct Puct { /* root_sign, dirichlet_noise, q_bounds, budget, legal_edges */ }
```

New Config fields: `c_puct` (~2.5), `dirichlet_alpha` (~0.15 for 249 actions), `dirichlet_epsilon` (~0.25).

Current Gumbel-Top-k with `num_sampled_actions=6` ignores ~97% of legal moves at root. PUCT naturally visits 30-60 distinct actions with 800 sims — better coverage, especially early in training when the policy prior is weak.

Quick experiment first: bump `num_sampled_actions` to 32-64 under Gumbel and compare.

## Reanalyze

Keep all self-play positions forever. Periodically re-run search on old positions with the current network to refresh policy/value targets. Decouples data generation from target quality.

Current flow:

```
self-play -> samples -> replay buffer (capped) -> train
```

With reanalyze:

```
self-play -> position store (permanent, outcomes only)
                  |
                  v
            reanalyze worker (re-searches with latest net)
                  |
                  v
              sample pool -> train
```

Each training batch mixes fresh self-play samples with reanalyzed samples (Will used 50/50). Reanalyze searches can be cheaper than self-play searches since we only need the root policy/value, not a full game trajectory.

Key decisions:

- **Reanalyze budget**: how many sims per reanalyzed position (can be less than self-play)
- **Mix ratio**: fraction of each batch from reanalyze vs fresh self-play
- **Staleness**: how often to re-search the same position (diminishing returns as net improves less between iterations)
- **Storage**: position store is just features + game outcome Z, no search targets

Reference: MuZero paper Appendix H. Will's TakZero: 50M positions, 50/50 mix, 4x target reuse.

## Playout cap randomization (from methods.md, not yet implemented)

Prerequisite for reanalyze to be cost-effective. 75% of moves use small budget (e.g. 50 sims), 25% use full budget. Only full-search positions produce policy targets. All positions produce value targets. ~4x more value samples per unit of search compute. The cheap searches are also good reanalyze candidates since they're fast to re-search.
