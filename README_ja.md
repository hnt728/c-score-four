# Score-four
Score-four は 4x4x4 の三次元ボードで遊ぶ、2 人用の戦略ゲームです。このプログラムは Score-four を C で実装したものです。

## 特徴
- 人間 vs 人間
- 人間 vs コンピュータ
- コンピュータ vs コンピュータ
- Minimax アルゴリズムによる AI（探索深さを調整可能）
- Python（PyTorch）で AlphaZero（PV-MCTS）学習・対戦（`score_four_az/`）

## インストールと実行
### コンパイル
```
gcc -o score_four src/main.c -fopenmp -O3 -march=native -lm
```
本プログラムは計算量の多い処理を行うため、OpenMP と `__builtin_popcountl` を利用します。`-O3`、`-fopenmp`、`-march=native` などの最適化オプションを付けてのビルドを推奨します。

### オプション
- `-1`, `--player1` `[h|m|c|r]`: プレイヤー 1 の種類を指定します。
    - `h`: 人間（デフォルト）
    - `m`: Minimax AI
    - `c`: MCTS（root-parallel UCT）
    - `r`: ランダム AI
- `-2`, `--player2` `[h|m|c|r]`: プレイヤー 2 の種類を指定します。
- `-d`, `--player1-depth` `[number]`: プレイヤー 1 の Minimax 探索深さを指定します。
- `-D`, `--player2-depth` `[number]`: プレイヤー 2 の Minimax 探索深さを指定します。
- `--no-board`: 盤面表示をしない
- `--no-result`: 結果（勝者/引き分け）を表示しない
- MCTS（グローバル / プレイヤー別上書き）:
    - `--mcts-iterations N` / `--player1-mcts-iterations N` / `--player2-mcts-iterations N`
    - `--mcts-time-ms MS` / `--player1-mcts-time-ms MS` / `--player2-mcts-time-ms MS`
    - `--mcts-threads T`
    - `--mcts-c C`
    - `--mcts-rollout-depth D`
    - `--mcts-max-nodes N`
    - `--mcts-verbose V`
    - `--mcts-seed SEED`

### 実行例
人間（先手）が Minimax AI（後手、探索深さ 3）と対戦する:
```
./score_four -1 h -2 m -D 3
```
人間（先手）が MCTS AI（後手、スレッド 4、1 手あたり 300ms）と対戦する:
```
./score_four -1 h -2 c --mcts-threads 4 --mcts-time-ms 300
```
Minimax AI 同士（両者とも探索深さ 4）で対戦し、盤面と結果表示を省略する:
```
./score_four -1 m -2 m -d 4 -D 4 --no-board --no-result
```

## ルール
プレイヤーは交互に駒を置きます。縦・横・奥行き・斜めのいずれかの方向で、自分の駒を 4 つ一直線に揃えたプレイヤーが勝ちです。

## ライセンス
このプロジェクトは MIT License の下で提供されています。
