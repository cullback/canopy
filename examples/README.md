# Examples

Each example is a self-contained directory under `examples/`.

- **`game.rs`** — Standalone game logic. No canopy imports; portable
  code you could drop into another project.
- **`main.rs`** — Implements the `Game` trait for the game type and
  wires up the CLI. Where Canopy meets the game.
- **`encoder.rs`, `model.rs`** — Tensor encoding and network
  architecture, used when the `nn` feature is enabled for training
  and inference.
- **`strategy.rs`** — Optional scripted opponents (random, heuristic)
  for tournament comparison.

## Games

- **[`pig`](pig/)** — Pig dice game. Small, stochastic, a good smoke
  test for the framework.
- **[`twenty48`](twenty48/)** — 2048. Solo stochastic game; shows how
  chance nodes work in a non-adversarial setting.
- **[`tenure`](tenure/)** — Spencer's Attacker-Defender Game. The
  attacker partitions pieces, the defender destroys a partition.
  Uses micro-actions to keep the action space linear.
- **[`catan`](catan/)** — 1v1 Catan. The largest example: web
  analysis board, live colonist.io integration, search-space
  optimizations (canonical ordering, dominated-action pruning,
  lexicographic discards). See its [README](catan/README.md).
