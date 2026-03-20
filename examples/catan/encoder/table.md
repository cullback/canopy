# NexusEncoder — 1266 features

## Global (107 = 7 + 44×2 + 12)

| Feature          | Count | Norm    | Notes              |
| ---------------- | ----- | ------- | ------------------ |
| Phase            | 7     | one-hot | scalar for Discard |
| Per-player       | 88    |         | cur player first   |
| Dice probability | 11    | prob    | per-value weights  |
| Dice deck frac   | 1     | /36     | 1.0 if random      |

## Per-player (44 × 2 = 88)

| Feature             | Count | Norm      | Notes              |
| ------------------- | ----- | --------- | ------------------ |
| Resources           | 5     | /19       | per resource       |
| Trade ratios        | 5     | /4        | (4−ratio)/4        |
| Number production   | 11    | /10       | bldg_wt per number |
| Victory points      | 1     | /15       | total / public     |
| Settlements left    | 1     | /5        |                    |
| Cities left         | 1     | /4        |                    |
| Roads left          | 1     | /15       |                    |
| Longest road award  | 1     | binary    |                    |
| Longest road length | 1     | /15       |                    |
| Largest army award  | 1     | binary    |                    |
| Dev playable        | 5     | /deck_max | exact / hypergeo   |
| Dev played          | 5     | /deck_max | visible both       |
| Dev bought          | 5     | /deck_max | exact / 0          |
| Played dev turn     | 1     | binary    |                    |

## Tiles (19 × 7 = 133)

| Feature  | Count | Norm    |
| -------- | ----- | ------- |
| Resource | 5     | one-hot |
| Pips     | 1     | /5      |
| Robber   | 1     | binary  |

## Nodes (54 × 19 = 1026)

| Feature            | Count | Norm  | Notes                |
| ------------------ | ----- | ----- | -------------------- |
| Building cur       | 1     | 0/½/1 |                      |
| Building opp       | 1     | 0/½/1 |                      |
| Port ratios        | 5     |       | .5 specific, .25 gen |
| Production         | 5     | /13   | adj tile pips/res    |
| Blocked production | 5     | /5    | robber tiles only    |
| Road count cur     | 1     | /3    |                      |
| Road count opp     | 1     | /3    |                      |
