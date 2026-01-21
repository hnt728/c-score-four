import math
import random

import numpy as np
import torch

from env import encode_state


class Node:
    __slots__ = ("N", "W", "P", "children", "expanded")

    def __init__(self):
        self.N = 0
        self.W = 0.0
        self.P = {}
        self.children = {}
        self.expanded = False

    @property
    def Q(self):
        if self.N == 0:
            return 0.0
        return self.W / self.N


class MCTS:
    def __init__(
        self,
        engine,
        model,
        num_simulations=200,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        device="cpu",
        batch_size=1,
    ):
        self.engine = engine
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = device
        self.batch_size = batch_size
        self._root = None
        self._root_state = None
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self.model.eval()

    def reset(self):
        self._root = None
        self._root_state = None
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_root(self, state):
        if self._root is None or self._root_state != state:
            self._root = Node()
            self._root_state = state
        return self._root

    def advance_to(self, state, action, next_state=None):
        if next_state is None:
            next_state = self.engine.apply_move(state, action)
        if self._root is None or self._root_state != state:
            self._root = Node()
        else:
            child = self._root.children.get(action)
            if child is None:
                child = Node()
            self._root = child
        self._root_state = next_state

    def _add_root_noise(self, node):
        legal_moves = list(node.P.keys())
        if not legal_moves:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
        for i, a in enumerate(legal_moves):
            node.P[a] = (1.0 - self.dirichlet_eps) * node.P[a] + self.dirichlet_eps * noise[i]

    def run(self, state, temperature=1.0, add_root_noise=True):
        root = self._get_root(state)
        if add_root_noise and root.expanded:
            self._add_root_noise(root)
            add_root_noise = False

        if self.batch_size <= 1:
            with torch.inference_mode():
                for _ in range(self.num_simulations):
                    self._search(state, root, add_root_noise=add_root_noise)
        else:
            pending = []
            sims_done = 0
            while sims_done < self.num_simulations:
                kind, payload = self._select_leaf(state, root, add_root_noise=add_root_noise)
                if kind == "value":
                    path, value = payload
                    self._backup(path, value)
                    sims_done += 1
                else:
                    pending.append(payload)
                    sims_done += 1
                    if len(pending) >= self.batch_size:
                        self._batch_infer(pending)
                        pending.clear()
            if pending:
                self._batch_infer(pending)

        policy = np.zeros(64, dtype=np.float32)
        for action, child in root.children.items():
            policy[action] = child.N

        if temperature <= 0:
            best = int(np.argmax(policy))
            out = np.zeros_like(policy)
            out[best] = 1.0
            return out

        if temperature != 1.0:
            policy = np.power(policy, 1.0 / temperature)
        total = policy.sum()
        if total > 0:
            policy /= total
        else:
            policy[:] = 1.0 / 64.0
        return policy

    def _search(self, state, node, add_root_noise=False):
        result = self.engine.result(state)
        if result != "n":
            if result == "d":
                value = 0.0
            else:
                value = 1.0 if result == state.turn else -1.0
            node.N += 1
            node.W += value
            return value

        if not node.expanded:
            value = self._expand(state, node, add_root_noise=add_root_noise)
            node.N += 1
            node.W += value
            return value

        best_action = None
        best_score = -1e9
        sqrt_N = math.sqrt(node.N + 1e-8)
        for action, prior in node.P.items():
            child = node.children.get(action)
            n = child.N if child else 0
            # child.Q is stored from the child node's current-player perspective,
            # so flip sign to evaluate actions from this node's perspective.
            q = (-child.Q) if child else 0.0
            u = self.c_puct * prior * sqrt_N / (1.0 + n)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None:
            return 0.0

        child = node.children.get(best_action)
        if child is None:
            child = Node()
            node.children[best_action] = child

        next_state = self.engine.apply_move(state, best_action)
        value = -self._search(next_state, child, add_root_noise=False)
        node.N += 1
        node.W += value
        return value

    def _expand(self, state, node, add_root_noise=False):
        key = (state.black, state.white, state.turn)
        cached = self._cache.get(key)
        if cached is None:
            self._cache_misses += 1
            legal_moves = self.engine.legal_moves_indices(state)
            if not legal_moves:
                node.expanded = True
                return 0.0
            x = encode_state(state).unsqueeze(0).to(self.device)
            logits, value = self.model(x)
            logits = logits.squeeze(0)
            value = float(value.squeeze(0).item())
            idx = torch.tensor(legal_moves, device=logits.device)
            probs_legal = torch.softmax(logits.index_select(0, idx), dim=0)
            base_probs_legal = probs_legal.detach().cpu().numpy()
            self._cache[key] = (tuple(legal_moves), base_probs_legal, value)
        else:
            self._cache_hits += 1
            legal_moves, base_probs_legal, value = cached

        if add_root_noise:
            probs = base_probs_legal.copy()
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
            probs = (1.0 - self.dirichlet_eps) * probs + self.dirichlet_eps * noise
        else:
            probs = base_probs_legal

        node.P = {a: float(probs[i]) for i, a in enumerate(legal_moves)}
        node.expanded = True
        return value

    def _backup(self, path, value):
        for node in reversed(path):
            node.N += 1
            node.W += value
            value = -value

    def _expand_from_cache(self, node, legal_moves, base_probs_legal, add_root_noise=False):
        if add_root_noise:
            probs = base_probs_legal.copy()
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
            probs = (1.0 - self.dirichlet_eps) * probs + self.dirichlet_eps * noise
        else:
            probs = base_probs_legal
        node.P = {a: float(probs[i]) for i, a in enumerate(legal_moves)}
        node.expanded = True

    def _select_leaf(self, state, root, add_root_noise=False):
        node = root
        cur_state = state
        path = []
        while True:
            path.append(node)
            result = self.engine.result(cur_state)
            if result != "n":
                if result == "d":
                    value = 0.0
                else:
                    value = 1.0 if result == cur_state.turn else -1.0
                return "value", (path, value)

            if not node.expanded:
                key = (cur_state.black, cur_state.white, cur_state.turn)
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache_hits += 1
                    legal_moves, base_probs_legal, value = cached
                    self._expand_from_cache(node, legal_moves, base_probs_legal, add_root_noise=add_root_noise)
                    return "value", (path, value)
                return "infer", (path, cur_state, node, add_root_noise)

            best_action = None
            best_score = -1e9
            sqrt_N = math.sqrt(node.N + 1e-8)
            for action, prior in node.P.items():
                child = node.children.get(action)
                n = child.N if child else 0
                q = (-child.Q) if child else 0.0
                u = self.c_puct * prior * sqrt_N / (1.0 + n)
                score = q + u
                if score > best_score:
                    best_score = score
                    best_action = action

            if best_action is None:
                return "value", (path, 0.0)

            child = node.children.get(best_action)
            if child is None:
                child = Node()
                node.children[best_action] = child

            cur_state = self.engine.apply_move(cur_state, best_action)
            node = child
            add_root_noise = False

    def _batch_infer(self, items):
        if not items:
            return
        self._cache_misses += len(items)
        states = torch.stack([encode_state(s) for _, s, _, _ in items]).to(self.device)
        with torch.inference_mode():
            logits, values = self.model(states)
        for i, (path, state, node, add_root_noise) in enumerate(items):
            legal_moves = self.engine.legal_moves_indices(state)
            if not legal_moves:
                self._backup(path, 0.0)
                continue
            logit = logits[i]
            value = float(values[i].squeeze(0).item())
            idx = torch.tensor(legal_moves, device=logit.device)
            probs_legal = torch.softmax(logit.index_select(0, idx), dim=0)
            base_probs_legal = probs_legal.detach().cpu().numpy()
            self._cache[(state.black, state.white, state.turn)] = (tuple(legal_moves), base_probs_legal, value)
            if not node.expanded:
                self._expand_from_cache(node, legal_moves, base_probs_legal, add_root_noise=add_root_noise)
            self._backup(path, value)

    def cache_stats(self):
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total) if total else 0.0
        return self._cache_hits, self._cache_misses, hit_rate


def select_action(policy):
    r = random.random()
    cdf = 0.0
    for i, p in enumerate(policy):
        cdf += float(p)
        if r <= cdf:
            return i
    return int(np.argmax(policy))
