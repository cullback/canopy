# Examples

Each example lives in its own folder with at least two files:

- **`game.rs`** — Standalone game implementation. No canopy2 imports. This should be portable code you could drop into any project.
- **`main.rs`** — Implements `Game` trait for the game type, wires up CLI and tournament. Kept minimal to showcase what integration looks like.

## Game ideas

popular

- [ ] Checkers
- [ ] Othello
- [ ] 1v1 Catan
- [ ] Chess
- [ ] 2048
- [ ] Connect four
- [ ] Risk
- [ ] Go

less popular

- [x] Pig
- Connect 6
- Quoridor
- [ ] Hex
