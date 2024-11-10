# Score-four
Score-four is a two-player strategy game played on a 4x4x4 three-dimensional board. This program is an implementation of Score-four in C.

## Features
- Human vs. Human gameplay
- Human vs. Computer gameplay
- Computer vs. Computer gameplay
- AI using Minimax algorithm (customizable search depth)

## Installation and Execution
### Compilation
```
gcc -o score_four main.c -fopenmp -O3 -march=native
```
This program performs computationally intensive processing, so it uses OpenMP and the __builtin_popcountl function. It is recommended to use optimization options such as -O3, -fopenmp, and -march=native.

### Options
- -1, --player1 [h|m|r]: Set the type of player 1.
    - h: Human (default)
    - m: Minimax AI
    - r: Random AI
- -2, --player2 [h|m|r]: Set the type of player 2.
- -d, --depth1 [number]: Set the AI search depth for player 1.
- -D, --depth2 [number]: Set the AI search depth for player 2.

### Example Gameplay
To play as a human (first player) against the Minimax AI (second player with search depth 3):
```
./score_four -1 h -2 m -D 3
```
To have two Minimax AIs (both with search depth 4) play against each other without displaying the board:
```
./score_four -1 m -2 m -d 4 -D 4
```
## Game Rules
Players take turns placing pieces on the board. The first player to align four of their pieces in any direction—vertically, horizontally, depth-wise, or diagonally—wins the game.

## License
This project is licensed under the MIT License.
