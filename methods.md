# Methods

Training and search techniques used in Canopy, organized by category. Impact ratings reflect expected contribution to playing strength per unit of implementation effort.

## Search - Gumbel AlphaZero — _high impact_

Replaces PUCT selection, Dirichlet noise, temperature-based action sampling, and visit-count policy targets with a unified Gumbel-based framework. Root uses Sequential Halving with Gumbel noise; non-root uses deterministic improved-policy selection. Guarantees policy improvement even at 2 simulations. Eliminates ~10 hyperparameters (Dirichlet alpha/weight, exploration temperature schedule, PUCT constants, FPU).

Reference: [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO) (Danihelka et al., ICLR 2022)

## Search - DAG-based transposition table — _medium impact_

Replaces the search tree with a directed acyclic graph. Different action sequences reaching the same board state share a single node. Games opt in via `state_key() -> Option<u64>`. Degrades gracefully to a normal tree when no transpositions exist. Impact depends heavily on game structure — large for games with many transpositions (2048, chess), negligible for games without (Go).

## Search - Tree reuse — _medium impact_

After each action, reroots the DAG at the new state and compacts to the surviving subtree. Subsequent searches start with existing visit counts and Q-values rather than from scratch. Free speed boost proportional to how much of the previous tree survives.

## Search - Batched leaf evaluation — _high impact_

Splits the MCTS loop into a state machine that pauses at leaf nodes, collects a batch, and yields for external evaluation. The caller controls batching across concurrent searches — multiple self-play workers share a single GPU forward pass. Without this, inference is one-leaf-at-a-time and GPU utilization is near zero.

## Search - Arena-based tree storage — _low-medium impact_

All nodes in a flat `Vec`, edges packed contiguously, integer indices instead of pointers. Better cache locality, zero per-node allocation. Makes tree reuse and compaction simple index remapping.

## Training targets - Value target mixing — _high impact_

Mixes game outcome `z` with MCTS root Q-value `q`: `target = (1-α)·z + α·q`. The mixing weight `q_weight` (α) ramps linearly from 0 to `q_weight_max` over `q_weight_ramp_iters` — early in training the value head is weak so Q is garbage, and pure game outcome Z provides the only real signal despite its variance. As the network improves, Q becomes a better per-position target than Z because it averages over many simulations rather than one game result. By the end of the ramp, Q dominates. Critical for stochastic games like Catan where dice variance makes pure Z extremely noisy throughout training.

## Training targets - Soft policy target — _medium impact_

A second policy head trained against a softened MCTS target: `soft_target = normalize(policy_target^(1/T))` with temperature T (default 4.0). Acts as a regularizer on the shared trunk — the soft head sees a smoother target that preserves probability mass on runner-up actions, preventing the network from becoming overly sharp. The soft head shares all intermediate features with the hard policy head; only the final linear projections are separate (~10K extra parameters). Controlled by `soft_policy_temperature` (0.0 disables) and `soft_policy_weight` (default 8.0).

Reference: [KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) — "Soft Policy"

## Training targets - Auxiliary short-term value heads — _medium impact_

Additional value heads trained on exponential moving averages of future Q-values, providing intermediate-horizon value signals alongside the main game-outcome value head. For each horizon h, targets are computed backwards through the game: `ema = α·Q[t] + (1-α)·ema` with `α = 1 - exp(-1/h)`. This gives the network credit for predicting what search will think a few actions ahead, not just the final outcome. Shares a hidden layer across all horizons with a single multi-output projection. Controlled by `aux_value_horizons` (e.g. `[4, 10, 30]` for Catan's ~90-move games; empty disables) and `aux_value_weight` (default 0.5 per head).

Reference: [KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) — "Auxiliary Short-Term Value Targets"

## Training efficiency - Playout cap randomization — _high impact_

75% of actions use a fast search (small budget); 25% use the full search. Only full-search positions contribute policy targets; all positions contribute value targets. Yielded 1.37x throughput improvement in KataGo — effectively quadruples the number of value training samples per unit of search compute. The boolean per-sample `full_search` flag masks policy loss during training.

Reference: [KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) — "Playout Cap Randomization"

## Training efficiency - Replay buffer — _low impact_

A sample-capped replay buffer (`replay_buffer_samples`) retains the most recent games. Training triggers on fresh sample count (`train_samples_per_iter`) rather than game count. Oldest games are evicted when the buffer exceeds capacity.

## Game representation - Canonical board representation — _medium impact (per-game strategy)_

For games with symmetry, the game model maps to a single canonical state (e.g., normalizing board orientation or player identity). Eliminates the need for data augmentation and makes transposition tables more effective. Must be designed into each game's model — retrofitting changes state hashing and invalidates stored training data. Impact is proportional to the symmetry group size. Implemented for Catan.

## Considering - Root policy softmax temperature — _medium impact_

Applies a temperature T=1.1–1.25 to the policy logits before Gumbel sampling at the root during self-play, decaying toward 1.0 over the course of each game. Slightly flattens the prior so search explores more broadly in the opening/midgame, acting as a restoring force against the policy becoming too peaked. Complementary to `explore_actions` (which forces visit-count proportional play) and `gumbel_m` (which controls how many actions enter Sequential Halving) — this is softer than either, nudging exploration without overriding search. Cheap to implement: scale logits by 1/T before adding Gumbel noise.

Reference: [KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) — "Root Policy Softmax Temperature"
