# Training configuration guide

How to size, tune, and reason about `TrainConfig` parameters. Each section
follows the same format: what the parameter controls, how to initialize it from
game properties, and how to diagnose misconfiguration from training metrics.

## Core loop

Each iteration: self-play generates at least `train_samples_per_iter` fresh
samples, then training runs `epochs` passes over the full replay buffer. Games
play to completion, so actual sample count may slightly exceed the target. The
buffer holds at most `replay_buffer_samples` samples, evicting oldest games
first.

```
retention        = replay_buffer_samples / train_samples_per_iter   (iters before eviction)
steps_per_epoch  = replay_buffer_samples / train_batch_size
gradient_steps   = epochs * steps_per_epoch
freshness_ratio  = train_samples_per_iter / replay_buffer_samples   (fraction of buffer that is new)
```

---

## `mcts_sims`

Search budget per action. More sims produce better policy targets (the MCTS
visit distribution the network trains to match) and more accurate value backups,
but each game takes proportionally longer. For a fixed compute budget, more sims
means fewer games — you're trading sample quantity for sample quality.

**Initialize:** estimate the effective branching factor B_eff — how many actions
the network puts non-trivial probability on once past random initialization (not
the legal action count). Start at 20–40 × B_eff, floor at 100.

If B_eff is unknown: `~2 × sqrt(action_space_size) × avg_game_length^0.25`
(order-of-magnitude estimate — sweep if uncertain).

| Game type                          | Typical B_eff | Starting sims |
| ---------------------------------- | ------------- | ------------- |
| Simple (Pig, Tic-tac-toe)          | 3-5           | 50-200        |
| Medium (Connect4, Othello)         | 5-15          | 200-800       |
| Deep tactics (Chess, Go)           | 8-30          | 800-1200      |
| Wide but shallow (Catan, Splendor) | 5-15          | 200-400       |

The last two rows both have high action counts but need sims for different
reasons. Chess/Go need deep search for tactical resolution — the value of a
move may not be apparent until 15+ actions later. Games like Catan have wide
branching but shallow tactical horizons; the strategy is positional, and the
network prior handles most of the assessment with search mainly comparing a few
candidates.

Stochastic games typically hit diminishing returns on sims sooner than
deterministic games of similar decision complexity — additional sims past a
point are just resampling the same chance distribution against a variance floor.
Size sims for tactical depth, and expect the useful range to be lower for
stochastic games.

**Diagnose:** check `policy_surprise_avg` in metrics.csv (KL of MCTS improved
policy vs network prior, averaged over full-search samples).

- Near zero (< 0.1 nats) throughout training → search is decorative, increase
  sims
- Very large (> 2 nats) late in training → network isn't learning from search
  targets, check policy loss and training pipeline
- Value loss stuck high while policy loss declines → tree isn't deep enough for
  reliable value backups, increase sims
- Metrics improve well per iteration but wall-clock is painful → sims fine
  algorithmically, too much compute per sample; reduce sims and compensate with
  more games (see [interactions](#how-sims-samples-and-buffer-interact))

## `mcts_sims_start`

Starting sims for the warmup ramp. Sims ramp linearly from `mcts_sims_start` to
`mcts_sims` over `warmup_iters`. Early in training both heads are random —
extra sims just average noise without producing better targets.

**Initialize:** `2–4 × gumbel_m`. This is the floor for Gumbel Sequential
Halving to complete one full halving pass. Below it the schedule is degenerate
and can't rank actions. Model size doesn't matter — a large model with random
weights is still random.

**Diagnose:**

- Early games are near-random even by early-training standards → sims_start
  below 2 × gumbel_m, Sequential Halving can't complete a pass
- First iterations are very slow with no quality benefit → sims_start too high,
  wasting compute on random evaluations

## `gumbel_m`

Top-k actions considered at root via Gumbel sampling. Sequential Halving
allocates sims across these candidates: `n / (ceil(log2(m)) × m)` visits per
candidate in the first round. Doubling `m` roughly halves per-action depth.
Clamped to the legal action count, so overshooting is safe but wastes sim
budget.

**Initialize:** set to roughly the number of plausible actions in a typical
position.

| Position type              | m     |
| -------------------------- | ----- |
| 5-10 good actions typical  | 16    |
| High-branching (Go, Chess) | 32-64 |

Verify: `mcts_sims / (ceil(log2(m)) × m)` should be ≥ 2. If not, either
increase sims or decrease m.

**Diagnose:**

- Policy improvement stalls, search never finds the right action → m too low,
  strong actions excluded from candidates
- MCTS policy target looks like the prior (no improvement from search) → m too
  high relative to sims, each candidate gets too few visits for Sequential
  Halving to distinguish them

## `train_samples_per_iter`

How much new data per iteration. Controls how much the network's opponents
change between training steps — the fundamental freshness/stability tradeoff.
Not "number of games" — the system collects games until it has enough samples.
Longer games produce more samples per game, so game count varies.

**Initialize:** think in games first, then convert.

```
num_games = max(256, 2 * action_space_size)
train_samples_per_iter = num_games * avg_game_length
```

The `2 * action_space_size` ensures diversity in the first action alone. The 256
floor provides statistical stability. Example: 100 legal actions, avg 60 actions
→ 200 games × 60 = 12,000 samples/iter.

Stochastic games should bias toward more games (and thus higher samples_per_iter
relative to a deterministic game of similar complexity). Noisier value targets
from chance events need more averaging across trajectories to produce stable
training signal.

Check the new-data-to-gradient-steps ratio after computing other params:

```
grad_steps = epochs × replay_buffer_samples / train_batch_size
ratio = grad_steps / train_samples_per_iter
```

- Ratio < 1: generating data faster than training on it, wasteful
- Ratio 2–10: sweet spot
- Ratio > 20: grinding on limited new data, overfitting risk

**Diagnose:**

- Loss oscillates between iterations (drops during training, jumps on fresh
  data) → overfitting within iteration, increase samples or reduce epochs
- Strength improves per iteration but wall-clock is painful → samples_per_iter
  too high, try halving it and running more frequent iterations
- Elo plateaus despite loss decreasing → possible diversity starvation, too few
  games exploring the game tree
- Self-play time dominates → samples_per_iter too high relative to training
  capacity. Training time dominates → too low

## `replay_buffer_samples`

Total buffer capacity in samples, sized as a multiple of
`train_samples_per_iter`. Controls how much historical data the network trains
on. At 5×, each sample survives 5 iterations; with 2 epochs/iter, each sample
gets trained on ~10 times across its lifetime.

**Initialize:** start at 5× `train_samples_per_iter`.

| Ratio | Retention  | Use when                                   |
| ----- | ---------- | ------------------------------------------ |
| 3-5×  | 3-5 iters  | Small/fast games, rapid policy improvement |
| 5-10× | 5-10 iters | Most games (default range)                 |

Adjust based on learning speed: rapid improvement (50+ Elo/iter) → use 3–5×
(staleness matters more). Slow, steady gains (5–15 Elo/iter) → 7–10× is fine
(old data still approximately valid). Below 3× the network oscillates (forgets
patterns, rediscovers, forgets). Above 10× old value targets actively mislead.

Stochastic games tolerate larger buffers than deterministic games at the same
learning speed. Staleness matters less when outcome variance from chance events
already dominates the variance from network improvement between iterations —
the old data isn't much noisier than new data. Bias toward 7–10× for stochastic
games.

**Diagnose:** track value prediction error stratified by data age (bucket by
iterations old).

- Old data has dramatically higher value loss (2×+) than recent data → buffer
  too large, old targets misleading the network. Shrink buffer.
- Value loss roughly uniform across ages → buffer could be larger without harm
- Policy loss decreasing but value loss stuck/increasing → buffer staleness,
  old value targets from weaker networks confuse value head. Shrink buffer.
- Network forgets openings it previously handled → buffer too small, those
  positions evicted. Increase buffer.
- Training loss suspiciously low but playing strength doesn't match →
  overfitting to a small buffer. Increase size.

## `epochs`

Training passes over the replay buffer per iteration. Each epoch reshuffles the
buffer. `gradient_steps = epochs × ceil(0.8 × replay_buffer_samples / train_batch_size)`
(the 0.8 accounts for the 80/20 train/validation split).

**Initialize:** start at 2. Typical range 2–4. Epochs interact with buffer
ratio: high epochs + small buffer = heavy overfitting; high epochs + large
buffer is more forgiving but slow.

**Diagnose:** compare train loss and validation loss at the end of each epoch
within an iteration.

- Validation loss rises while training loss drops → overfitting, reduce epochs
- Epoch 3+ shows minimal improvement over epoch 2 → diminishing returns, stay
  at 2
- Each sample seen roughly once (1 epoch) → may not extract full signal from
  high-quality full-search samples

## `playout_cap_full_prob`

Fraction of actions that get full search and contribute to policy training.
Policy loss trains only on full-search samples (fast-search samples are masked
to zero). Value loss trains on all samples. This directly controls policy
training signal density.

**Initialize:** 0.25. Compute savings:
`avg_sims = p × full_sims + (1 - p) × fast_sims`. With p=0.25, full=800,
fast=64: 248 avg sims vs 800 = 3.2× speedup. Policy samples per game ≈
`p × non_forced_actions`.

**Diagnose:**

- Policy loss stagnates while value loss improves → policy head starving for
  signal, increase p
- Self-play nearly as slow as without playout cap → p too high, minimal compute
  savings

## `playout_cap_fast_sims`

Sims for fast-search actions. Fast-search actions form the game trajectory that
determines Z (game outcome). Bad fast search produces noisy Z targets.

**Initialize:** at least `2 × gumbel_m` (same Sequential Halving floor as
`mcts_sims_start`). Below that, fast-search picks near-random actions. Keep
below `full_sims / 4` for meaningful compute savings. Default: 64.

**Diagnose:**

- Games are much longer or more random than expected → fast_sims too low,
  trajectory quality degraded, Z targets noisy
- fast_sims > full_sims / 4 → speedup from playout cap randomization is modest,
  consider lowering

## `train_batch_size`

Samples per gradient step. Determines how many gradient steps per iteration
and the noise/smoothness tradeoff in optimization.

**Initialize:** start at 1024. Check the resulting step count:
`gradient_steps = epochs × ceil(0.8 × replay_buffer_samples / train_batch_size)`.
Target 100–500 steps/iter. Adjust batch size or buffer to stay in range.

If you change batch size, co-adjust learning rate. The linear scaling rule:
doubling batch size → double LR (each step averages twice as many gradients, so
a larger step is safe). This isn't exact — very large batches may need
sublinear scaling — but it's the right default.

**Diagnose:**

- < 50 steps/iter → batch too large, each epoch does very little work, loss
  barely moves within an iteration
- \> 1000 steps/iter → iterations slow from optimizer overhead
- Loss spikes during training → batch too small, gradient noise too high
- Changed batch size without adjusting LR → effective learning rate scales
  inversely; doubling batch size halves effective step size

## `inference_batch_size`

Max evaluations per GPU forward pass during self-play. The batcher blocks until
the first inference request arrives, then drains all queued requests up to this
limit and fires one forward pass.

**Initialize:** 1024 for most GPUs. Increase for large GPUs with spare
throughput. The actual batch fill depends on `concurrent_games` and
`leaf_batch_size`.

**Diagnose:**

- `avg batch` in progress bar < 50% of `inference_batch_size` → GPU underfull,
  increase `concurrent_games` or `leaf_batch_size`
- `evals/s` dropping → inference bottleneck, check GPU utilization

## `leaf_batch_size`

MCTS leaves collected per search before pausing to request a GPU eval. During
search, each simulation reaching an unexpanded leaf pushes it into a pending
buffer (with virtual loss). Once pending count reaches `leaf_batch_size`, MCTS
yields and sends a batched inference request. Higher values improve GPU batching
but stall individual searches. Must not exceed `mcts_sims`.

**Initialize:**

| mcts_sims | leaf_batch_size |
| --------- | --------------- |
| 50-100    | 1-4             |
| 200-800   | 4-16            |
| 800+      | 16-32           |

**Diagnose:**

- `avg batch` low despite enough concurrent games → leaf_batch_size too small,
  each game contributes too few samples per request
- Search quality degrades (policy targets look noisy) → leaf_batch_size too
  high relative to sims, most sims consumed by batch collection

## `concurrent_games`

Async game tasks running in parallel during self-play. Each game contributes
`leaf_batch_size` samples per inference request. The batcher needs enough
concurrent requests to fill `inference_batch_size`.

**Initialize:** start at 256. Minimum to fill one batch:
`concurrent_games × leaf_batch_size ≥ inference_batch_size`. Overshoot by 2–4×
since not all games submit requests simultaneously. Each concurrent game holds
an MCTS tree and game state in memory — for complex games with large trees this
cost adds up.

**Diagnose:**

- `avg batch` < 50% of `inference_batch_size` → increase concurrent_games
- Memory pressure / OOM → too many concurrent games with large trees, reduce

## `max_actions`

Safety cap on game length (total actions including chance nodes). Games hitting
the cap terminate as a draw (reward=0). Set to 0 to disable.

**Initialize:** 2–3× the expected game length with competent play. Estimate avg
actions per game (decisions + chance nodes). Pig to 100: ~130 actions with good
play → cap at 500. Catan: ~1000 actions → cap at 2000+.

**Diagnose:**

- High draw rate with avg game length near max_actions → cap too low, games
  hitting the limit and getting reward=0, destroying value signal
- Early iterations very slow → random play produces very long games, set a cap
  even for complex games

## `warmup_iters`

Ramp period for sims and Q-weight. `warmup_frac = min(iter / warmup_iters, 1)`.
Two things ramp together:

- Sims: `effective = mcts_sims_start + frac × (mcts_sims - mcts_sims_start)`
- Q-weight: `q_weight = frac × q_weight_max`

Early in training the value head is random, so Q targets (derived from search
using that value head) are noise. Z (actual game outcome) provides the only real
signal. As the network improves, Q becomes a better per-position target than the
coarse game-level Z.

**Initialize:** warmup should last until the value head starts discriminating
positions. No fixed formula — depends on model capacity and game complexity.
Start at 20–30 iterations and adjust based on diagnostics.

**Diagnose:** watch `value_error_late` and `value_network_stddev` in the
training CSV.

- `value_error_late` declining and `value_network_stddev` rising above near-zero
  → value head is producing meaningful evaluations, warmup should be ending
- These signals arrive well before warmup ends → warmup too long, wasting
  iterations on coarse Z targets
- Warmup ends while value head is still flat → warmup too short, Q targets
  introduced before they're useful; value loss may jump when Q-weight kicks in

## `q_weight_max`

Maximum blend toward Q (search value) after warmup completes. The value target
is `(1 - q_weight) × Z + q_weight × Q`. Values below 1.0 retain a Z anchor
that prevents value drift from self-referential Q targets.

**Initialize:** 0.85. This keeps 15% Z weight as a grounding signal. Use 1.0
only with high sims where Q is reliably better than Z.

**Diagnose:**

- Value loss slowly drifts upward after warmup, playing strength plateaus →
  q_weight_max too high with insufficient sims, value head chasing its own tail.
  Reduce q_weight_max or increase sims.
- Search produces excellent Q targets but value head isn't learning from them →
  q_weight_max too low, search refinement wasted

## `explore_actions`

Number of initial actions where the agent samples from the MCTS-improved policy
distribution instead of taking the deterministic Sequential Halving best action.
Compared against the total action count (including chance nodes and forced
single-legal-action actions), not just decision points.

**Initialize:** start at 20–30% of avg_game_length for deterministic games,
10–15% for stochastic games.

- Deterministic games: need more explore_actions since there's no natural
  trajectory diversity. Set to cover the opening phase where many positions look
  similar to the network.
- Stochastic games: chance nodes provide natural diversity, so fewer are needed.
  Effective explored decisions = explore_actions minus expected chance actions in
  that window.

**Diagnose:** measure trajectory diversity — count unique game states at action
N across self-play games in an iteration.

- Diversity plateaus at action 1 (deterministic games) → explore_actions too
  low, self-play producing near-identical games, network overfits to a narrow
  slice of the game tree
- Diversity already high by action 5 (stochastic games) → explore_actions
  beyond that adds noise without benefit
- Network produces unrealistic positions in early game → explore_actions too
  high, noisy MCTS policy sampling creates positions the network wastes capacity
  learning

---

## How sims, samples, and buffer interact

These three parameters form a triangle:

- **mcts_sims** → quality per sample (how good the search targets are)
- **train_samples_per_iter** → quantity of fresh data per learning step
- **replay_buffer_samples** → memory (how much history the network trains on)

For a fixed compute budget, increasing sims means decreasing samples_per_iter
(fewer but better games). The KataGo finding: early in training, the trade
favors more games with fewer sims. As the network strengthens, search becomes
more valuable and you can tolerate fewer games because each game's targets are
higher quality.

Practical approach: start with moderate sims (~200), generous samples_per_iter,
and 5× buffer. Increase sims once the network is past warmup and losses are
declining.

Stochasticity pushes the entire triad: fewer sims (variance floor), more games
(noisier targets need more averaging), larger buffer (staleness matters less
when outcome variance dominates network-improvement variance).

## `aux_value_horizons`

Auxiliary value heads that predict short-horizon exponential moving averages of
future Q-values. Each horizon `h` produces a target via backwards EMA:
`alpha = 1 - exp(-1/h)`, so horizon 10 averages roughly the next 10 actions'
search evaluations. Shares hidden layers with the main value head (single
multi-output projection).

Essential for stochastic games: the correlation between position quality and
final outcome can be genuinely weak when dozens of future chance events
intervene. Short-horizon targets let the value head learn positional concepts
(e.g., "building here improves my position") without that signal being washed
out by outcome noise. For deterministic games, useful but less critical since Z
is already a strong signal.

**Initialize:** use 2–4 horizons in a geometric spread covering short to
medium-long range. Keep all horizons well below avg_game_length — longer
horizons approach Z and share its noise problem.

| Avg game length | Example horizons |
| --------------- | ---------------- |
| ~50 actions     | [4, 10, 25]      |
| ~200 actions    | [8, 25, 75]      |
| ~1000 actions   | [10, 30, 100]    |

The shortest horizon should be long enough to capture a meaningful decision
sequence (not just 1–2 actions). The longest should be short enough to still
have substantially lower variance than Z.

**Diagnose:** check `loss_aux_value_*_train` / `loss_aux_value_*_val` per
horizon in metrics.csv.

- Shortest horizon loss improves quickly while main value loss is stuck →
  aux targets are working, providing signal that Z can't
- Longest horizon loss tracks main value loss closely → that horizon isn't
  adding much beyond Z, consider shortening it
- All aux losses stuck high → horizons may be too long (approaching Z noise
  level) or the network lacks capacity
- Aux loss very low but main value loss still high → the value head learns
  short-term patterns but can't extrapolate to game outcome. Expected early in
  training; if persistent, may need longer horizons or more capacity

## Quick-start recipe

1. Estimate avg game length (actions) with random play
2. `max_actions` = 2–3× that estimate
3. `gumbel_m` = 16, `mcts_sims` = 200, `mcts_sims_start` = 64
   - Verify: `mcts_sims / (ceil(log2(m)) × m) ≥ 2`. With m=16, sims=200:
     `200 / (4 × 16) = 3.1` ✓. If you drop sims below 128 with m=16, this
     fails — reduce m or increase sims.
4. `train_samples_per_iter` = `max(256, 2 × action_space_size) × avg_game_length`
   - Stochastic games: bias toward the higher end (more games)
5. `replay_buffer_samples` = 5× `train_samples_per_iter` (7–10× for stochastic)
6. `epochs` = 2, `train_batch_size` = 1024
7. `playout_cap_full_prob` = 0.25, `playout_cap_fast_sims` = 64
8. `concurrent_games` = 256, `inference_batch_size` = 1024
9. `warmup_iters` = 25, `q_weight_max` = 0.85
10. `explore_actions` = 20–30% of avg_game_length (10–15% for stochastic)
11. `aux_value_horizons`: 2–4 horizons in geometric spread, well below
    avg_game_length. Essential for stochastic games.
12. Run one iteration. Check:
    - `avg batch` > 50% of `inference_batch_size`? If not, increase
      `concurrent_games`.
    - Self-play vs train time roughly balanced? If not, adjust
      `train_samples_per_iter`.
    - `gradient_steps` in 100–500 range? If not, adjust `train_batch_size`
      (and co-adjust LR: double batch → double LR).
13. After warmup, increase `mcts_sims` — better targets justify slower
    iterations once the network is strong.

## Hardware utilization signals

| Metric                      | Healthy                        | Action                                                 |
| --------------------------- | ------------------------------ | ------------------------------------------------------ |
| `avg batch`                 | >50% of `inference_batch_size` | Low → increase `concurrent_games` or `leaf_batch_size` |
| `evals/s`                   | Stable, proportional to GPU    | Dropping → inference bottleneck                        |
| `self-play` vs `train` time | Roughly balanced               | One dominates → adjust `train_samples_per_iter`        |
| Draw rate                   | Decreasing over iterations     | Constant high → `max_actions` too low                  |
