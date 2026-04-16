# Catan

Two-player Catan with MCTS search, a web analysis board, and a live
[colonist.io](https://colonist.io) integration via CDP.

## Running against a live colonist.io game

Download the `nexus-v3` model checkpoint from the
[catan-nexus-v3 release](https://github.com/cullback/canopy/releases/tag/catan-nexus-v3),
then:

```
cargo run --release --example catan --features server,nn -- \
    colonist --port 9223 --serve 3000 --eval nexus-v3:path/to/model_iter_315.mpk
```

`--port 9223` is the CDP endpoint (see [CDP.md](CDP.md) for the
Chrome and SSH tunnel setup) and `--serve 3000` exposes the web
analysis board on `http://localhost:3000`.

## Search and state-space optimizations

Catan-specific techniques (canonical action ordering, lexicographic
discard, dominated-action pruning, SO-ISMCTS determinization,
canonical state encoding) are documented in
[OPTIMIZATIONS.md](OPTIMIZATIONS.md).

## Colonist replay

```
https://colonist.io/api/replay/data-from-game-id?gameId=178911848&playerColor=1
```
