import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from env import Engine, GameState, encode_state, render_board
from mcts import MCTS, select_action
from model import PolicyValueNet

from tqdm import tqdm


def load_model(path, device):
    model = PolicyValueNet().to(device)
    if path and Path(path).exists():
        model.load_state_dict(torch.load(path, map_location=device))
    return model


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def self_play_game(engine, mcts, temperature_moves=8):
    state = GameState(0, 0, "b")
    history = []
    step = 0
    while True:
        temp = 1.0 if step < temperature_moves else 0.0
        policy = mcts.run(state, temperature=temp, add_root_noise=True)
        history.append((state, policy))
        action = select_action(policy) if temp > 0 else int(np.argmax(policy))
        next_state = engine.apply_move(state, action)
        mcts.advance_to(state, action, next_state)
        state = next_state
        result = engine.result(state)
        if result != "n":
            break
        step += 1
    data = []
    for s, p in history:
        if result == "d":
            z = 0.0
        else:
            z = 1.0 if result == s.turn else -1.0
        data.append((s, p, z))
    return data, result


def _random_move(engine, state):
    legal = engine.legal_moves_indices(state)
    return random.choice(legal)


def _heuristic_move(engine, state):
    legal = engine.legal_moves_indices(state)
    for mv in legal:
        next_state = engine.apply_move(state, mv)
        if engine.result(next_state) == state.turn:
            return mv

    opp_turn = "w" if state.turn == "b" else "b"
    opp_state = GameState(state.black, state.white, opp_turn)
    opp_legal = engine.legal_moves_indices(opp_state)
    opp_wins = set()
    for mv in opp_legal:
        next_state = engine.apply_move(opp_state, mv)
        if engine.result(next_state) == opp_turn:
            opp_wins.add(mv)

    blocking = [mv for mv in legal if mv in opp_wins]
    if blocking:
        return random.choice(blocking)
    return random.choice(legal)


def _play_game(engine, move_b, move_w, mcts=None):
    state = GameState(0, 0, "b")
    moves = 0
    while True:
        result = engine.result(state)
        if result != "n":
            return result, moves
        if state.turn == "b":
            mv = move_b(state)
        else:
            mv = move_w(state)
        next_state = engine.apply_move(state, mv)
        if mcts is not None:
            mcts.advance_to(state, mv, next_state)
        state = next_state
        moves += 1


def _benchmark(engine, model, device, sims, games, mcts_batch):
    mcts = MCTS(engine, model, num_simulations=sims, device=device, batch_size=mcts_batch)

    def mcts_move(state):
        policy = mcts.run(state, temperature=0.0, add_root_noise=False)
        return int(np.argmax(policy))

    results = {}
    for name, opp_move in (("random", _random_move), ("heuristic", _heuristic_move)):
        wins = 0
        losses = 0
        draws = 0
        move_counts = []
        for g in range(games):
            model_is_black = (g % 2) == 0
            if model_is_black:
                result, moves = _play_game(engine, mcts_move, lambda s: opp_move(engine, s), mcts=mcts)
                model_color = "b"
            else:
                result, moves = _play_game(engine, lambda s: opp_move(engine, s), mcts_move, mcts=mcts)
                model_color = "w"

            if result == "d":
                draws += 1
            elif result == model_color:
                wins += 1
            else:
                losses += 1
            move_counts.append(moves)
        if move_counts:
            move_min = min(move_counts)
            move_max = max(move_counts)
            move_avg = sum(move_counts) / len(move_counts)
        else:
            move_min = move_max = move_avg = 0.0
        results[name] = {
            "win": wins,
            "loss": losses,
            "draw": draws,
            "games": games,
            "moves_min": move_min,
            "moves_avg": move_avg,
            "moves_max": move_max,
        }
    return results


def train_model(model, data, device, batch_size, epochs, lr):
    states = torch.stack([encode_state(s) for s, _, _ in data]).to(device)
    policies = torch.from_numpy(np.stack([p for _, p, _ in data])).float().to(device)
    values = torch.from_numpy(np.array([[z] for _, _, z in data], dtype=np.float32)).to(device)

    dataset = TensorDataset(states, policies, values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        n_batches = 0
        sum_policy = 0.0
        sum_value = 0.0
        sum_total = 0.0
        for batch_states, batch_policies, batch_values in loader:
            logits, pred_values = model(batch_states)
            policy_loss = -(batch_policies * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(pred_values, batch_values)
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_batches += 1
            sum_policy += float(policy_loss.item())
            sum_value += float(value_loss.item())
            sum_total += float(loss.item())

        if n_batches:
            print(
                f"  epoch {epoch + 1}/{epochs}: "
                f"loss={sum_total / n_batches:.4f} "
                f"(policy={sum_policy / n_batches:.4f}, value={sum_value / n_batches:.4f})"
            )


def cmd_selfplay(args):
    device = torch.device(args.device)
    engine = Engine()
    model = load_model(args.model, device)
    mcts = MCTS(engine, model, num_simulations=args.sims, device=device, batch_size=args.mcts_batch)

    all_data = []
    results = {"b": 0, "w": 0, "d": 0}
    for _ in range(args.games):
        data, result = self_play_game(engine, mcts, temperature_moves=args.temp_moves)
        all_data.extend(data)
        results[result] += 1

    print(f"self-play results: {results}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        states = torch.stack([encode_state(s) for s, _, _ in all_data]).numpy()
        policies = np.stack([p for _, p, _ in all_data])
        values = np.array([z for _, _, z in all_data], dtype=np.float32)
        np.savez_compressed(out, states=states, policies=policies, values=values)
        print(f"saved data: {out}")


def cmd_train(args):
    print(args.device)
    device = torch.device(args.device)
    engine = Engine()
    model = load_model(args.model, device)

    for it in tqdm(range(args.iters)):
        mcts = MCTS(engine, model, num_simulations=args.sims, device=device, batch_size=args.mcts_batch)
        all_data = []
        results = {"b": 0, "w": 0, "d": 0}
        for _ in range(args.games_per_iter):
            data, result = self_play_game(engine, mcts, temperature_moves=args.temp_moves)
            all_data.extend(data)
            results[result] += 1
        hits, misses, hit_rate = mcts.cache_stats()
        print(
            f"iter {it + 1}: self-play {results}, samples={len(all_data)}, "
            f"cache hit {hit_rate * 100:.1f}% ({hits}/{hits + misses})"
        )
        train_model(model, all_data, device, args.batch_size, args.epochs, args.lr)

        if args.out:
            save_model(model, args.out)
            print(f"saved model: {args.out}")

        if args.bench_interval and (it + 1) % args.bench_interval == 0:
            bench = _benchmark(engine, model, device, args.sims, args.bench_games, args.mcts_batch)
            for name, stats in bench.items():
                print(
                    f"benchmark vs {name}: "
                    f"win {stats['win']}, loss {stats['loss']}, draw {stats['draw']} "
                    f"({stats['games']} games) "
                    f"moves min/avg/max "
                    f"{stats['moves_min']}/"
                    f"{stats['moves_avg']:.1f}/"
                    f"{stats['moves_max']}"
                )


def cmd_play(args):
    device = torch.device(args.device)
    engine = Engine()
    model = load_model(args.model, device)
    mcts = MCTS(engine, model, num_simulations=args.sims, device=device, batch_size=args.mcts_batch)

    human = args.human
    state = GameState(0, 0, "b")
    while True:
        print(render_board(state))
        result = engine.result(state)
        if result != "n":
            print(f"result: {result}")
            break

        if state.turn == human:
            legal = engine.legal_moves_indices(state)
            print(f"legal moves: {legal}")
            move = None
            while move is None:
                try:
                    raw = input("enter move index (0-63): ").strip()
                    mv = int(raw)
                    if mv in legal:
                        move = mv
                    else:
                        print("illegal move")
                except (ValueError, EOFError):
                    print("invalid input")
            next_state = engine.apply_move(state, move)
            mcts.advance_to(state, move, next_state)
            state = next_state
        else:
            policy = mcts.run(state, temperature=0.0, add_root_noise=False)
            action = int(np.argmax(policy))
            next_state = engine.apply_move(state, action)
            mcts.advance_to(state, action, next_state)
            state = next_state


def build_parser():
    p = argparse.ArgumentParser(description="Score-four AlphaZero minimal runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("selfplay", help="generate self-play data")
    sp.add_argument("--model", default="", help="path to model file")
    sp.add_argument("--games", type=int, default=2)
    sp.add_argument("--sims", type=int, default=200)
    sp.add_argument("--mcts-batch", type=int, default=1, help="MCTS inference batch size")
    sp.add_argument("--temp-moves", type=int, default=8)
    sp.add_argument("--out", default="", help="npz output path")
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=cmd_selfplay)

    tr = sub.add_parser("train", help="self-play + train loop")
    tr.add_argument("--model", default="", help="path to initial model file")
    tr.add_argument("--out", default="score_four_az/models/latest.pt")
    tr.add_argument("--iters", type=int, default=3)
    tr.add_argument("--games-per-iter", type=int, default=4)
    tr.add_argument("--sims", type=int, default=200)
    tr.add_argument("--mcts-batch", type=int, default=1, help="MCTS inference batch size")
    tr.add_argument("--temp-moves", type=int, default=8)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--epochs", type=int, default=2)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--bench-interval", type=int, default=0, help="run benchmark every N iters (0=disable)")
    tr.add_argument("--bench-games", type=int, default=6, help="games per benchmark opponent")
    tr.add_argument("--device", default="cpu")
    tr.set_defaults(func=cmd_train)

    pl = sub.add_parser("play", help="play vs AI")
    pl.add_argument("--model", default="", help="path to model file")
    pl.add_argument("--sims", type=int, default=200)
    pl.add_argument("--mcts-batch", type=int, default=1, help="MCTS inference batch size")
    pl.add_argument("--human", choices=["b", "w", "n"], default="b")
    pl.add_argument("--device", default="cpu")
    pl.set_defaults(func=cmd_play)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "human") and args.human == "n":
        args.human = "x"
    args.func(args)


if __name__ == "__main__":
    main()
