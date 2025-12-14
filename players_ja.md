# Players（プレイヤー種別）と使い方

このプロジェクトの実行ファイル（例: `./score_four`）は、プレイヤー 1 / 2 の「種類」を指定して対戦させられます。

## プレイヤー種別（`-1/--player1`, `-2/--player2`）

プレイヤーの種類は `h` / `r` / `m` / `c` の 4 つです。

- `h`: Human（人間）
- `r`: Random（ランダム）
- `m`: Minimax（αβ探索）
- `c`: MCTS（root-parallel UCT）

指定しない場合のデフォルトは `player1=h`, `player2=h` です。

## 基本的な指定方法（例）

人間 vs 人間（デフォルト）:
```sh
./score_four
```

人間（先手） vs Minimax（後手、探索深さ 3）:
```sh
./score_four -1 h -2 m -D 3
```

人間（先手） vs MCTS（後手、4 スレッド、1 手あたり 300ms）:
```sh
./score_four -1 h -2 c --mcts-threads 4 --player2-mcts-time-ms 300
```

盤面表示と結果表示を抑制（ログ収集やベンチ向け）:
```sh
./score_four -1 m -2 c -d 3 --no-board --no-result
```
※ `--no-board` / `--no-result` は盤面と最終結果の表示のみを抑制します。`turn:` や `black put ...` などのログは引き続き出力されます。

## `h`: Human（人間）

- 対話入力で手を選びます。
- 1 手ごとに `Enter index:` と表示されるので、合法手の **index（0〜63 の整数）** を入力します。
- 不正な index を入れると、合法手が入力されるまで再入力になります。
- 数値以外を入力すると読み取りに失敗して進まないことがあるため、整数で入力してください。

### index の意味（64 マスの並び）

盤面は「4x4 が 4 層（合計 64 マス）」です。表示上は 4 層ぶんの 4x4 が順に出ます。

- 0 層: `0..15`
- 1 層: `16..31`
- 2 層: `32..47`
- 3 層: `48..63`

各層の 4x4 は行優先（row-major）です:

```
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

このゲームは「重力あり」のため、各（行,列）ごとに同じ位置の index が `+16` ずつ積み上がります（例: `0 -> 16 -> 32 -> 48`）。
空の状態で最初に置けるのは 0 層（`0..15`）のみで、下が埋まるとその上（`+16`）が次に合法手になります。

## `r`: Random（ランダム）

- その時点の合法手からランダムに 1 手を選びます。
- 実行ごとに乱数シードが変わるため、同じ条件でも手順が変わることがあります。

## `m`: Minimax（αβ探索）

- αβ探索で手を選びます（探索深さを指定可能）。
- **必須**: Minimax を使う側は探索深さを指定してください。
  - プレイヤー 1 が `m` の場合: `-d N` または `--player1-depth N`
  - プレイヤー 2 が `m` の場合: `-D N` または `--player2-depth N`
- 深さを大きくすると強くなりますが、計算量が急増して遅くなります。
- 探索は OpenMP で並列化されます（環境によってスレッド数が変わります）。

## `c`: MCTS（root-parallel UCT）

- モンテカルロ木探索（UCT）で手を選びます。
- デフォルト設定（起動時点）:
  - `iterations=20000`
  - `time_ms=0`（時間制限なし）
  - `threads=0`（`omp_get_max_threads()` を使用）
  - `C=1.41421356237`
  - `rollout_max_depth=64`
  - `max_nodes=0`（自動）
  - `verbose=1`
  - `seed=0`（自動）

### MCTS パラメータ（グローバル）

以下は **両プレイヤー（MCTS を使う側）に同時に適用** されます。

- `--mcts-iterations N`: 1 手あたりのシミュレーション回数（合計）
  - `N <= 0` の場合は「回数制限なし」になり、`--mcts-time-ms` による時間制限まで回します（時間指定は必須）。
- `--mcts-time-ms MS`: 1 手あたりの時間制限（ミリ秒）
- `--mcts-threads T`: 探索スレッド数（`<=0` なら最大スレッド）
- `--mcts-c C`: UCT の探索定数
- `--mcts-rollout-depth D`: ロールアウトの最大手数
- `--mcts-max-nodes N`: スレッドごとのノード上限（`<=0` なら自動）
- `--mcts-verbose V`: ログ詳細（`0` で抑制、`1` 以上で手ごとに統計表示）
- `--mcts-seed SEED`: 乱数シード（固定化したい場合に指定）

`--mcts-iterations > 0` かつ `--mcts-time-ms > 0` を両方指定した場合、探索は「回数」または「時間」のどちらか先に到達した方で止まります。
`--mcts-iterations <= 0` の場合は、回数制限はかからず `--mcts-time-ms` の時間まで探索します。

時間制限ベースで回したい例（300ms/手）:
```sh
./score_four -1 h -2 c --mcts-iterations 0 --mcts-time-ms 300
```

### MCTS パラメータ（プレイヤー別の上書き）

以下は **プレイヤー別に上書き** できます（指定した側のみ変更）。

- `--player1-mcts-iterations N`
- `--player2-mcts-iterations N`
- `--player1-mcts-time-ms MS`
- `--player2-mcts-time-ms MS`

注意:
- オプションは与えた順に反映されるため、同じ項目を複数回指定した場合は **後勝ち** になります（例: 先に `--mcts-time-ms`、後から `--player2-mcts-time-ms` を指定すると、プレイヤー 2 のみ後者が有効）。

## 出力制御

- `--no-board`: 盤面表示をしない
- `--no-result`: 結果（勝者/引き分け）を表示しない
  - どちらも「それ以外のログ出力」を止めるものではありません。

## オプション一覧（まとめ）

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
