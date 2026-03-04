# Canopy

MCTS framework for two player games.

- two-player, stochastic games

## Game ideas

- Pig
- 2048
- Connect four
- 1v1 Catan
- Checkers
- Othello
- Risk

## Optimizations

---

> **DAG-based graph search instead of a tree**

Replaces the tree with a directed acyclic graph, enabling transposition tables where different move sequences reaching the same board state share a single node. Used by KataGo since v1.12. Degrades gracefully to normal MCTS if no transpositions exist. Retrofit cost is extreme: backpropagation must handle multiple parents, node lifetime management changes completely, and every traversal routine needs rewriting.

> **Gumbel AlphaZero (replacing PUCT entirely)**

Replaces PUCT selection, Dirichlet noise, temperature-based action sampling, and visit-count policy targets with a unified Gumbel-based framework. Root uses Sequential Halving with Gumbel noise; non-root uses deterministic improved-policy selection. Guarantees policy improvement even at 2 simulations. Touches the MCTS selection loop, root action selection, policy target computation, and the training loop — essentially every module in the search-and-train pipeline.

> **Resumable search with batched NN evaluation**

Splits the MCTS loop so that search pauses at leaf nodes, batches multiple pending evaluations into a single GPU forward pass, then resumes. Without this design from the start, the search is synchronous one-eval-at-a-time, and retrofitting requires restructuring the entire search control flow into a state machine with suspend/resume semantics.

> **Efficient arena-based tree data structure with tree reuse**

Stores all nodes in a pre-allocated arena (no per-node heap allocation), using integer indices instead of pointers. Supports tree reuse by rerooting the DAG after each move rather than discarding it. If you start with standard heap-allocated nodes, switching to arena indexing later means rewriting every node access pattern across the codebase.

> **Canonical board representation**

For games with symmetry, the game model always maps to a single canonical state (e.g., always placing the first settlement in a canonical position, or normalizing player identity). Eliminates the need for data augmentation and makes transposition tables far more effective. Must be designed into the game model interface from the start — retrofitting changes state hashing, transposition keys, and invalidates all previously stored training data.

> **Value target mixing**

Mixes the actual game outcome `z` with the MCTS root Q-value `q` as the training target: `target = α·z + (1-α)·q`, where α linearly falls from 1.0 to 0.0 over ~20 generations. Critical for Catan where dice variance makes pure `z` noisy. The key architectural requirement is storing `(state, π, z, q)` tuples in training samples from day one — if you only store `(state, π, z)`, adding `q` later means discarding all historical training data.

> **Progressive simulation**

Starts training with very few MCTS simulations (e.g., n=2) and gradually ramps up to full budget (e.g., n=200) over the course of training, keeping total compute constant. Early iterations churn through more games cheaply while the network is weak; later iterations invest in deeper search once the evaluator is worth planning with. Implementation is just a schedule over the simulation budget config parameter. No architectural changes.

> **Playout cap randomization**

On 75% of moves, run a fast search (small budget, e.g., n=32); on 25%, run a full search (large budget, e.g., n=200). Only full-search positions contribute policy targets; all positions contribute value targets. Yielded 1.37× throughput improvement in KataGo. Implementation is a single boolean flag per move deciding search depth — trivial to bolt on later.

> **Learning rate scheduling**

Cosine or step-decay schedule for the learning rate during training. Pure training-config change, no architectural impact.

> **Replay buffer strategy**

Controls how training data is sampled and aged out — e.g., sampling each position ~8× before discarding, or prioritized replay. Pure training-config change, no architectural impact.

## References

- https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf
- https://gwern.net/doc/reinforcement-learning/model/alphago/2017-silver.pdf#page=5
- https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
- [Enhancements for Real-Time Monte-Carlo Tree Search in General Video Game Playing (2024)](https://arxiv.org/html/2407.03049v1)
  - progressive simulation
