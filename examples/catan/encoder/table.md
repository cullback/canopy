# NexusEncoder — 1441 features

## Global (117 = 7 + 49×2 + 12)

| Feature          | Count | Norm    | Notes              |
| ---------------- | ----- | ------- | ------------------ |
| Phase            | 7     | one-hot | scalar for Discard |
| Per-player       | 98    |         | cur player first   |
| Dice probability | 11    | prob    | per-value weights  |
| Dice deck frac   | 1     | /36     | 1.0 if random      |

## Per-player (49 × 2 = 98)

### Economy (26)

| Feature        | Count | Norm | Notes                |
| -------------- | ----- | ---- | -------------------- |
| resource_count | 5     | /19  | per resource         |
| trade_ratio    | 5     | /4   | (4−ratio)/4          |
| resource_prod  | 5     | /35  | bldg_wt × pips / res |
| number_prod    | 11    | /10  | bldg_wt per number   |

### Board (7)

| Feature             | Count | Norm   | Notes          |
| ------------------- | ----- | ------ | -------------- |
| settlement_count    | 1     | /5     | remaining      |
| city_count          | 1     | /4     | remaining      |
| road_count          | 1     | /15    | remaining      |
| longest_road_award  | 1     | binary |                |
| longest_road_length | 1     | /15    |                |
| largest_army_award  | 1     | binary |                |
| victory_points      | 1     | /15    | total / public |

### Dev cards (16)

| Feature         | Count | Norm      | Notes            |
| --------------- | ----- | --------- | ---------------- |
| dev_playable    | 5     | /deck_max | exact / hypergeo |
| dev_played      | 5     | /deck_max | visible both     |
| dev_bought_turn | 5     | /deck_max | exact / 0        |
| dev_played_turn | 1     | binary    |                  |

## Tiles (19 × 10 = 190)

| Feature             | Count | Norm    | Notes                        |
| ------------------- | ----- | ------- | ---------------------------- |
| resource            | 5     | one-hot |                              |
| pips                | 1     | /5      | long-run structural strength |
| roll_prob           | 1     | raw     | current balanced-dice prob   |
| robber              | 1     | binary  |                              |
| cur_building_weight | 1     | /6      | sum of bldg_wt, 6 nodes      |
| opp_building_weight | 1     | /6      | sum of bldg_wt, 6 nodes      |

## Nodes (54 × 21 = 1134)

| Feature          | Count | Norm  | Notes                |
| ---------------- | ----- | ----- | -------------------- |
| cur_building     | 1     | 0/½/1 |                      |
| opp_building     | 1     | 0/½/1 |                      |
| port_ratio       | 5     |       | .5 specific, .25 gen |
| resource_prod    | 5     | /13   | adj tile pips / res  |
| blocked_prod     | 5     | /5    | robber tiles only    |
| cur_road_count   | 1     | /3    | incident roads       |
| opp_road_count   | 1     | /3    | incident roads       |
| cur_network_dist | 1     | /6    | BFS, capped at 6     |
| opp_network_dist | 1     | /6    | BFS, capped at 6     |

## Normalization reference

| Divisor | Meaning                           |
| ------- | --------------------------------- |
| 19      | max resource cards of one type    |
| 14/5/2  | original deck count per dev type  |
| 5       | max settlements / max tile pips   |
| 4       | max cities                        |
| 15      | max roads / win threshold (VP)    |
| 13      | max pips at a single node (5+4+4) |
|         | roll_prob is raw probability      |
| 35      | max per-resource production       |
| 10      | max per-number production         |
| 3       | max incident roads at a node      |
| 6       | max tile corner nodes / BFS cap   |
| 36      | balanced dice deck size           |
