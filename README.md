# Score-four

Score-four is a two-player strategy game played on a 4x4x4 three-dimensional board. This program is a C implementation of Score-four.

## Features
- Human vs Human
- Human vs Computer
- Computer vs Computer
- AI based on the Minimax algorithm (search depth configurable)
- AlphaZero (PV-MCTS) training and play in Python (PyTorch) (`score_four_az/`)

## Build and Run
### Compile
```sh
gcc -o score_four src/main.c -fopenmp -O3 -march=native -lm
```
This program performs heavy computation, so it uses OpenMP and `__builtin_popcountl`. We recommend building with optimization options such as `-O3`, `-fopenmp`, and `-march=native`.

### Options
- `-1`, `--player1` `[h|m|c|r]`: Set the type of player 1.
    - `h`: Human (default)
    - `m`: Minimax AI
    - `c`: MCTS (root-parallel UCT)
    - `r`: Random AI
- `-2`, `--player2` `[h|m|c|r]`: Set the type of player 2.
- `-d`, `--player1-depth` `[number]`: Set the Minimax search depth for player 1.
- `-D`, `--player2-depth` `[number]`: Set the Minimax search depth for player 2.
- `--no-board`: Do not display the board
- `--no-result`: Do not display the final result (winner/draw)
- MCTS (global / per-player overrides):
    - `--mcts-iterations N` / `--player1-mcts-iterations N` / `--player2-mcts-iterations N`
    - `--mcts-time-ms MS` / `--player1-mcts-time-ms MS` / `--player2-mcts-time-ms MS`
    - `--mcts-threads T`
    - `--mcts-c C`
    - `--mcts-rollout-depth D`
    - `--mcts-max-nodes N`
    - `--mcts-verbose V`
    - `--mcts-seed SEED`

### Examples
Human (first) vs Minimax AI (second, search depth 3):
```sh
./score_four -1 h -2 m -D 3
```
Human (first) vs MCTS AI (second, 4 threads, 300ms per move):
```sh
./score_four -1 h -2 c --mcts-threads 4 --mcts-time-ms 300
```
Minimax vs Minimax (both depth 4), omitting board and result output:
```sh
./score_four -1 m -2 m -d 4 -D 4 --no-board --no-result
```

## Rules
Players take turns placing pieces. The first player to line up four of their pieces in a straight line—vertically, horizontally, in depth, or diagonally—wins.

## License
This project is provided under the MIT License.
