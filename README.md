# Canopy

MCTS framework for two player games.

## Optimizations

- graph based mcts, allows for transposition tables
- MCTS is a state machine that yields NeedsEval(state) when it hits a leaf and pauses until the caller
  provides NnOutput via supply(). The caller (an orchestrator) decides how to fulfill evals — sync NN call, batched GPU,
  rollout, cache lookup
- tree reuse / rerooting
