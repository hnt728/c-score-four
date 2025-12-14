# Players (Types) and Usage

The executable in this project (e.g., `./score_four`) lets you run matches by selecting the “type” of player 1 / 2.

## Player Types (`-1/--player1`, `-2/--player2`)

There are four player types: `h` / `r` / `m` / `c`.

- `h`: Human
- `r`: Random
- `m`: Minimax (alpha-beta search)
- `c`: MCTS (root-parallel UCT)

If not specified, the defaults are `player1=h`, `player2=h`.

## Basic Examples

Human vs Human (default):
```sh
./score_four
```

Human (first) vs Minimax (second, search depth 3):
```sh
./score_four -1 h -2 m -D 3
```

Human (first) vs MCTS (second, 4 threads, 300ms per move):
```sh
./score_four -1 h -2 c --mcts-threads 4 --player2-mcts-time-ms 300
```

Suppress board and final result output (for logging/benchmarks):
```sh
./score_four -1 m -2 c -d 3 --no-board --no-result
```
Note: `--no-board` / `--no-result` only suppress the board display and the final outcome. Logs like `turn:` and `black put ...` are still printed.

## `h`: Human

- You choose moves via interactive input.
- Each turn, the program prints `Enter index:`; enter the legal move **index (an integer 0–63)**.
- If you enter an invalid index, it will keep prompting until you enter a legal move.
- If you enter non-numeric input, reading may fail and the program may not proceed, so please enter an integer.

### What the index means (64 cells)

The board is “4x4 for 4 layers” (64 cells total). In the display, the 4x4 board for each layer is printed in order.

- Layer 0: `0..15`
- Layer 1: `16..31`
- Layer 2: `32..47`
- Layer 3: `48..63`

Within each layer, the 4x4 is row-major:

```
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

Because this game has “gravity”, for each (row, column) the same position stacks upward by `+16` (e.g., `0 -> 16 -> 32 -> 48`). From an empty position, only layer 0 (`0..15`) is legal at first; once the lower cell is filled, the cell above it (`+16`) becomes the next legal move.

## `r`: Random

- Chooses one move uniformly at random from the currently legal moves.
- Because the random seed changes each run, the move sequence may differ even under the same conditions.

## `m`: Minimax (alpha-beta search)

- Chooses moves using alpha-beta search (search depth is configurable).
- **Required**: If you use Minimax, you must set the search depth.
  - If player 1 is `m`: `-d N` or `--player1-depth N`
  - If player 2 is `m`: `-D N` or `--player2-depth N`
- Increasing depth makes it stronger, but the computation grows rapidly and it becomes slower.
- The search is parallelized with OpenMP (the number of threads depends on your environment).

## `c`: MCTS (root-parallel UCT)

- Chooses moves using Monte Carlo Tree Search (UCT).
- Default settings (at startup):
  - `iterations=20000`
  - `time_ms=0` (no time limit)
  - `threads=0` (uses `omp_get_max_threads()`)
  - `C=1.41421356237`
  - `rollout_max_depth=64`
  - `max_nodes=0` (auto)
  - `verbose=1`
  - `seed=0` (auto)

### MCTS Parameters (Global)

These apply to **both players (who use MCTS)** at the same time.

- `--mcts-iterations N`: Number of simulations per move (total)
  - If `N <= 0`, it becomes “no iteration limit” and keeps running until the time limit from `--mcts-time-ms` (time must be specified).
- `--mcts-time-ms MS`: Time limit per move (milliseconds)
- `--mcts-threads T`: Number of search threads (`<= 0` uses max threads)
- `--mcts-c C`: UCT exploration constant
- `--mcts-rollout-depth D`: Maximum rollout length
- `--mcts-max-nodes N`: Per-thread node limit (`<= 0` is auto)
- `--mcts-verbose V`: Log verbosity (suppress with `0`, show per-move stats with `>= 1`)
- `--mcts-seed SEED`: Random seed (set to make runs deterministic)

If you specify both `--mcts-iterations > 0` and `--mcts-time-ms > 0`, the search stops when it reaches whichever limit comes first (“iterations” or “time”).
If `--mcts-iterations <= 0`, there is no iteration limit and it searches up to the time limit from `--mcts-time-ms`.

Example: time-limited search (300ms per move):
```sh
./score_four -1 h -2 c --mcts-iterations 0 --mcts-time-ms 300
```

### MCTS Parameters (Per-player Overrides)

These can be overridden **per player** (only the specified side changes).

- `--player1-mcts-iterations N`
- `--player2-mcts-iterations N`
- `--player1-mcts-time-ms MS`
- `--player2-mcts-time-ms MS`

Note:
- Options are applied in the order they appear, so if you specify the same setting multiple times, the **last one wins** (e.g., specify `--mcts-time-ms` first, then `--player2-mcts-time-ms` to override only player 2).

## Output Controls

- `--no-board`: Do not display the board
- `--no-result`: Do not display the final result (winner/draw)
  - Neither of these stops “other log output”.

## Options Summary

```text
-1, --player1 [h|m|c|r]
-2, --player2 [h|m|c|r]
-d, --player1-depth N
-D, --player2-depth N
    --no-board
    --no-result
    --mcts-iterations N
    --mcts-time-ms MS
    --mcts-threads T
    --mcts-c C
    --mcts-rollout-depth D
    --mcts-max-nodes N
    --mcts-verbose V
    --mcts-seed SEED
    --player1-mcts-iterations N
    --player2-mcts-iterations N
    --player1-mcts-time-ms MS
    --player2-mcts-time-ms MS
```

