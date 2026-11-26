# Take It Easy AI

Play **Take It Easy** against a neural network, right in your browser — no install, no server.

**Play:** https://filiphalys02.github.io/Take-It-Easy-AI/

- You and the network get the **same sequence of 19 tiles** and each fill your own hexagonal board.
- You always move first; after your move the network places the same tile on its board.
- The higher final score wins. Works on desktop and on phones (tap a cell to preview the tile, tap again to place it).
- Interface languages: English (default) and Polish. The **RULES** button links to the game rules on Wikipedia
  ([EN](https://en.wikipedia.org/wiki/Take_It_Easy_(game)), [PL](https://pl.wikipedia.org/wiki/Take_it_Easy)).

## How the AI plays

| | |
|---|---|
| Model | Value network (~590k parameters) that estimates the expected future score of a board position |
| Search | Expectimax, depth 2, on top of the network (draw a tile → best cell, twice) |
| Endgame | Exact solution to the end of the game when 5 or fewer cells are empty |
| Runtime | Entirely client-side: ONNX Runtime Web (WebAssembly) in a Web Worker; model file is ~2.1 MB |
| Move time | ~1.5 s for the slowest (first) move in headless Chromium on a desktop PC; faster later in the game |

The network was trained purely by self-play value iteration — there is no human game data and no hand-written strategy.

## Strength

Scoring: each of the 15 lines (5 vertical, 5 "/", 5 "\\") scores *(common number) × (line length)* when all its tiles carry the same number
in that direction. The theoretical maximum is **307** points.

Results on **1000 games** with a fixed set of tile sequences (seed 12345), network used greedily (depth 1):

| Metric | Value |
|---|---|
| Average score | **166.6** (±0.9 standard error) |
| Standard deviation | 27.6 |
| 10th / 50th / 90th percentile | 131 / 170 / 200 |
| P(score ≥ 100) | 99% |
| P(score ≥ 150) | 75% |
| P(score ≥ 200) | 11% |

For reference, an earlier smaller test (100 games, seed 999) compared the search variants:

| Policy | Average score | Standard error |
|---|---|---|
| Network, greedy (depth 1) | 164.8 | ±2.7 |
| Network + expectimax, depth 2 (full) | 165.2 | ±2.8 |
| **Browser version** (depth 2, top-5 moves, no rotation averaging) | 165.2 | ±2.8 |

Notes:
- With a standard deviation of ~28 points per game, 100 games give a standard error of ~2.8 and 1000 games ~0.9.
- On that small sample, depth-2 search added no measurable gain over the greedy network (the difference is well inside the noise).
- The browser version prunes depth-2 search to the 5 best moves; in the 100-game test this gave the same average as the full search
  (paired difference 0.09 ± 0.09), at ~7× lower cost.
- The in-browser engine chose the same cell as the reference Python implementation in 75 of 78 test positions;
  the 3 differences were opening moves where the reference itself samples tiles randomly.
- The score does not depend on the other player, so these numbers describe single-board play.

## Training code

The code used to train the network (and the training logs) is **not publicly available**.
If you are interested in it, or have questions about the model, feel free to get in touch via my GitHub profile
([@filiphalys02](https://github.com/filiphalys02)).

## Run locally

The app is a static site (ES modules + Web Worker), so it needs to be served over HTTP (it will not work from `file://`):

```bash
python -m http.server 8000 -d app
# open http://127.0.0.1:8000/
```

## Credits

- Game: *Take It Easy* by Peter Burley (published by many companies).
- Inference: [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT license), bundled in `app/vendor/`.
