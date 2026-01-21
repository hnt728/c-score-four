# Score-Four AlphaZero（最小構成・コード準拠）

このフォルダは **4x4x4 立体四目並べ**を AlphaZero 方式で学習・対戦する最小構成です。
ルール部分は C の共有ライブラリ（`src-c/engine.c`）に切り出し、Python で MCTS（PUCT）と学習・CLI 対戦を行います。

## 概要
- C 側: 合法手生成・勝敗判定・着手適用（ビットボード）
- Python 側: MCTS（NN評価のみ）、自己対戦、学習、CLI

## 必要なもの
- Python 3.10+ 推奨
- PyTorch
- GCC（共有ライブラリのビルド）

`uv` 用の `pyproject.toml` / `uv.lock` が `score_four_az/` にあります。

## ファイルと役割（実コード準拠）
- `env.py`
  - C エンジンの ctypes バインディング
  - `GameState(black, white, turn)`
  - 合法手・結果・着手適用
  - NN入力テンソル化（`2x4x4x4`）
- `mcts.py`
  - PUCT 型 MCTS
  - NN から `policy logits` と `value` を受け取る
- `model.py`
  - 3D CNN（policy 64、value 1）
- `main.py`
  - `selfplay` / `train` / `play` の CLI

## C エンジンのビルド
リポジトリルートから：
```
bash src-c/build.sh
```
生成物：
- `src-c/libscorefour.so`

別パスに置く場合は環境変数で指定：
```
SCORE_FOUR_LIB=/path/to/libscorefour.so
```

## 実行方法（パッケージ不要）
`__init__.py` は使っていないので、単純にスクリプト実行します。

### 自己対戦データ生成
```
python score_four_az/main.py selfplay --games 2 --sims 200
```

### 学習（自己対戦→学習ループ）
```
python score_four_az/main.py train --iters 3 --games-per-iter 4 --sims 200
```

### 対戦（人間 vs AI）
```
python score_four_az/main.py play --human b --sims 200
```
- `--human b` 黒 / `--human w` 白
- `--human n` で AI vs AI

## コマンド詳細
### selfplay
```
python score_four_az/main.py selfplay \
  --model score_four_az/models/latest.pt \
  --games 8 \
  --sims 300 \
  --temp-moves 8 \
  --out score_four_az/data/selfplay.npz
```
- `--model`: 省略可（ランダム初期値）
- `--temp-moves`: 序盤のみ温度 > 0 で多様化
- `--out`: 生成データを `.npz` 保存

### train
```
python score_four_az/main.py train \
  --model score_four_az/models/latest.pt \
  --out score_four_az/models/latest.pt \
  --iters 5 \
  --games-per-iter 8 \
  --sims 300 \
  --batch-size 64 \
  --epochs 2 \
  --lr 1e-3 \
  --bench-interval 2 \
  --bench-games 6
```
- 各イテレーションで「自己対戦 → そのデータで学習」
- **リプレイバッファなし**（毎回そのイテレーションのデータのみ学習）

### `train` の主要パラメータ（何を変えているか / 増減の影響）
この実装は **「自己対戦で集めたデータ（その iter 分）だけで学習」**するため、特に `games-per-iter` と `epochs` のバランスで挙動が変わります。

- `--iters`（外側ループ回数）
  - 意味: 「自己対戦 → 学習」を何回繰り返すか。
  - 増やす: データ分布が繰り返し更新され、強くなりやすい（ただし総計算時間はほぼ比例して増える）。
  - 減らす: すぐ終わるが、学習が進む前に止まりやすい。
- `--games-per-iter`（1 iter あたりの自己対戦局数）
  - 意味: その iter の学習に使う教師データ量（局面数）を決める。
  - 増やす: 1 iter の学習が安定しやすく、過学習しにくい傾向（ただし自己対戦が重くなる）。
  - 減らす: 1 iter が軽くなるが、データが薄くなり `epochs` を回すと過学習気味になりやすい。
- `--sims`（MCTS のシミュレーション回数）
  - 意味: 1手あたり何回 `_search` を回して policy（訪問回数）を作るか。
  - 増やす: policy が鋭くなりやすく教師の質が上がる一方、自己対戦が大幅に遅くなる。
  - 減らす: 自己対戦が速くなるが、探索が浅くなり policy がノイジーになりやすい（学習のブレが増えることがある）。
- `--batch-size`（学習のミニバッチ）
  - 意味: 1回の更新で何サンプルまとめて勾配計算するか。
  - 増やす: 更新が安定しやすい／GPU なら効率が上がることがある（ただしメモリ使用量が増える）。
  - 減らす: メモリは軽くなるが、勾配ノイズが増えて不安定になりやすい（同じ性能までに時間がかかることも）。
- `--epochs`（1 iter のデータを何周学習するか）
  - 意味: その iter の自己対戦データを DataLoader で何周回すか。
  - 増やす: その iter のデータへの当てはまりが強くなる（**リプレイバッファ無し**なので上げすぎると過学習・方策の崩れが起きやすい）。
  - 減らす: 1 iter は軽くなるが、学習が進みにくい／更新が小さくなりやすい。
- `--lr`（学習率）
  - 意味: Adam の更新幅。
  - 増やす: 収束が速くなる場合があるが、発散・不安定化（value が振れる、policy が極端になる等）しやすい。
  - 減らす: 安定しやすいが、学習が遅くなりやすい（iters を多めに回す必要が出やすい）。
- `--bench-interval`（ベンチマークを実行する iter 間隔）
  - 意味: 指定した iter ごとに簡易ベンチマークを挟む（0 で無効）。
  - 増やす: ベンチマーク頻度が下がり、学習は速いが進捗確認の粒度が粗くなる。
  - 減らす: 進捗は追いやすいが、その分計算が増える。
- `--bench-games`（ベンチマークの対戦回数）
  - 意味: 1相手あたりの対戦回数（半分は黒番・半分は白番）。
  - 増やす: 勝率が安定するが、ベンチマークが重くなる。
  - 減らす: 速いが、勝率のばらつきが大きくなる。

#### ベンチマーク内容
`train` 中に以下の相手と対戦して勝率を出します（指定回数、黒白交互）。
- `random`：合法手からランダムに選択
- `heuristic`：即勝ちがあればそこ、相手の即勝ちがあればブロック、それ以外はランダム

### play
```
python score_four_az/main.py play --model score_four_az/models/latest.pt --sims 400
```

## 盤面・行動空間（C 実装と一致）
- 盤面は 64bit ビットボード 2枚（black/white）
- 行動は **index 0–63**
  - `bit = 1 << (63 - idx)` の対応
  - `Engine.move_bit(index)` / `Engine.move_index(bit)` で相互変換
- 合法手は最大 16（重力付きの柱ごと 1 手）

## テンソル化（`encode_state`）
`env.encode_state(state)` は **2x4x4x4** のテンソルを返します：
- ch0: 手番プレイヤーの石
- ch1: 相手の石

インデックス→座標の対応（Python実装そのまま）
- `idx = z*16 + y*4 + x`
- `bit = 1 << (63 - idx)`

## MCTS（実装どおり）
- PUCT で行動選択：
  - `Q + c_puct * P * sqrt(N) / (1 + n)`
- NN は `(policy logits, value)` を返す
- value は **「手番プレイヤー視点」**として扱う
  - 探索で手番が変わるので値を符号反転
- ルートで Dirichlet ノイズを混ぜる
- ロールアウトは行わない（NN評価のみ）

## 学習ロジック（実装どおり）
- self-play で `(state, policy, z)` を作成
  - `policy` は訪問回数から作る
  - `z` は結果（勝ち +1 / 負け -1 / 引分 0）
  - **`z` はその局面の手番プレイヤー視点**
- 損失:
  - policy: クロスエントロピー
  - value: MSE

## 生成データ形式（`selfplay --out`）
`.npz` に以下を保存：
- `states`: shape `(N, 2, 4, 4, 4)`（**すでにテンソル化済み**）
- `policies`: shape `(N, 64)`
- `values`: shape `(N,)`

## 各ファイルの詳細解説（関数ごと）

### `src-c/engine.h`
**C 共有ライブラリの公開 API** を定義します。
- `az_init()`
  - 内部テーブル初期化（`engine.c` の `init_cell_lines()`）を一度だけ実行。
- `az_legal_moves(black, white, out_moves)`
  - 盤面の合法手を bit 形式で最大 16 個 `out_moves` に格納。
  - 戻り値は合法手数。
- `az_result(black, white)`
  - `'b'` / `'w'` / `'d'` / `'n'` を返す。
  - `n` はゲーム継続。
- `az_apply_move(black, white, turn, move, out_black, out_white)`
  - `move` を `turn`（'b' or 'w'）に適用した新盤面を返す。
  - `out_black` / `out_white` に結果を書き込む。
- `az_move_index(move)`
  - bit 形式の着手を index(0–63) に変換。
- `az_move_bit(index)`
  - index(0–63) を bit 形式に変換。

### `src-c/engine.c`
**C ルールエンジン本体**です。
- **ビットボード表現**
  - `black` と `white` の 64bit で盤面を表現。
  - `index 0` は最上位ビット（MSB）に対応。
- **勝利ライン**
  - `conditions[76]` に 76 本の勝利ラインをビットマスクで保持。
- **合法手生成**
  - `get_possible_pos_board()` が重力付きの合法手を生成。
  - `get_possible_poses_binary()` は合法手を最大 16 個配列へ。
- **勝敗判定**
  - `which_is_win()` は `conditions` に対して包含判定。
  - 合法手が 0 の場合は引き分け `'d'`。
- **ビット変換**
  - `bit_to_index()` / `index_to_bit()` で index と bit を変換。
- **高速化用テーブル**
  - `init_cell_lines()` は各セルが属する勝利ラインを事前計算。
  - 現状の公開 API では未使用だが、将来の高速化用に残している。

### `src-c/build.sh`
- `libscorefour.so` を生成する最小ビルドスクリプト。
- 実行内容:
  - `gcc -shared -fPIC -O3 -o libscorefour.so engine.c`

### `score_four_az/env.py`
**Python 側の C バインディング＋状態表現**です。
- `_load_lib()`
  - `SCORE_FOUR_LIB` があればそれを利用。
  - 無ければ `src-c/libscorefour.so` を探す。
  - ctypes の `argtypes/restype` を設定。
  - `az_init()` を必ず呼ぶ。
- `GameState`
  - `black`, `white`, `turn` の不変データ。
- `Engine.legal_moves_bits(state)`
  - `az_legal_moves()` を呼び出し、bit の合法手配列を返す。
- `Engine.legal_moves_indices(state)`
  - 上記の bit を index(0–63) に変換。
- `Engine.apply_move(state, move)`
  - `move` が index(0–63) なら `az_move_bit()` で bit 化。
  - C 側の `az_apply_move()` で新盤面を得る。
  - 手番を自動で反転して次状態を返す。
- `Engine.result(state)`
  - `az_result()` を呼び出し `'b'/'w'/'d'/'n'` を返す。
- `Engine.move_bit()` / `Engine.move_index()`
  - index ↔ bit 変換の薄いラッパ。
- `encode_state(state)`
  - 2ch のテンソル `2x4x4x4` を作成。
  - ch0: 手番プレイヤー、ch1: 相手。
- `_fill_plane(plane, board_bits)`
  - `bit = 1 << (63 - idx)` でビットを走査。
  - `idx -> (z,y,x)` は `z=idx//16`, `y=(idx%16)//4`, `x=idx%4`。
- `render_board(state)`
  - 4 層を `layer 0..3` として表示。
  - `X`=黒, `O`=白, `.`=空。

### `score_four_az/mcts.py`
**PUCT 型 MCTS（NN評価のみ）**です。
- `Node`
  - `N`（訪問回数）, `W`（累積価値）, `P`（事前確率）, `children` を保持。
  - `Q = W / N` をプロパティで計算。
- `MCTS.run(state, temperature)`
  - `num_simulations` 回 `_search()` を実行。
  - 探索回数を policy 分布（訪問回数）に変換。
  - `temperature<=0` は argmax に固定。
- `MCTS._search(state, node, add_root_noise)`
  - 終局なら value を返す（draw=0、勝ち=+1、負け=-1）。
  - 未展開なら `_expand()` で NN 評価。
  - 展開済みなら PUCT で最良行動を選び子に再帰。
  - 子ノードに入るとき value を符号反転。
- `MCTS._expand(state, node, add_root_noise)`
  - `legal_moves_indices()` で合法手を取得。
  - NN で `(logits, value)` を評価。
  - 非合法手を `-1e9` でマスクして softmax。
  - ルートノイズは **最初の展開時のみ**混ざる。
- `select_action(policy)`
  - policy 分布に従って確率的にサンプリング。

### `score_four_az/model.py`
**最小構成の policy/value ネットワーク**です。
- 入力: `2x4x4x4`
- Backbone: `Conv3d(2→32)` → `Conv3d(32→64)`
- Policy head: Flatten → Linear(64*4*4*4→256→64)
- Value head: Flatten → Linear(64*4*4*4→128→1) → `tanh`

### `score_four_az/main.py`
**CLI エントリポイント**です。
- `load_model(path, device)`
  - path が存在すればロード、無ければ初期化のまま。
- `save_model(model, path)`
  - 親ディレクトリを作成して保存。
- `self_play_game(engine, mcts, temperature_moves)`
  - 1局分の自己対戦を実行。
  - `temperature_moves` まで温度あり、以降は argmax。
  - 返り値は `(state, policy, z)` の配列。
- `train_model(model, data, ...)`
  - policy クロスエントロピー + value MSE。
  - optimizer は Adam。
- `cmd_selfplay()`
  - 指定回数の self-play を実行し `.npz` 保存。
- `cmd_train()`
  - self-play → 学習を `iters` 回繰り返す。
  - **リプレイバッファ無し**。
- `cmd_play()`
  - 人間 vs AI / AI vs AI の簡易対戦。
  - `--human n` の場合、内部で `human = "x"` にして AI vs AI。

## 注意点・制限
- 最小構成のため最適化は最小限
- MCTS は単スレッド・評価のバッチ化なし
- 評価マッチ（新旧モデル比較）は未実装

## トラブルシュート
- `shared library not found` → `bash src-c/build.sh` を実行
- PyTorch が無い → 環境にインストール（uv/pip/conda 等）
