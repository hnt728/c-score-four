import os
from dataclasses import dataclass
from pathlib import Path
import ctypes

import torch


def _load_lib():
    override = os.environ.get("SCORE_FOUR_LIB")
    if override:
        lib_path = Path(override)
    else:
        root = Path(__file__).resolve().parents[1]
        lib_path = root / "src-c" / "libscorefour.so"
    if not lib_path.exists():
        raise RuntimeError(f"shared library not found: {lib_path}")
    lib = ctypes.CDLL(str(lib_path))

    lib.az_init.argtypes = []
    lib.az_init.restype = None

    lib.az_legal_moves.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    lib.az_legal_moves.restype = ctypes.c_int

    lib.az_result.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    lib.az_result.restype = ctypes.c_char

    lib.az_apply_move.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_char,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.az_apply_move.restype = None

    lib.az_move_index.argtypes = [ctypes.c_uint64]
    lib.az_move_index.restype = ctypes.c_int

    lib.az_move_bit.argtypes = [ctypes.c_int]
    lib.az_move_bit.restype = ctypes.c_uint64

    lib.az_init()
    return lib


_LIB = _load_lib()


@dataclass(frozen=True)
class GameState:
    black: int
    white: int
    turn: str  # 'b' or 'w'


class Engine:
    def __init__(self):
        self._lib = _LIB

    def legal_moves_bits(self, state: GameState):
        arr = (ctypes.c_uint64 * 16)()
        n = self._lib.az_legal_moves(state.black, state.white, arr)
        return [arr[i] for i in range(n)]

    def legal_moves_indices(self, state: GameState):
        moves = self.legal_moves_bits(state)
        return [int(self._lib.az_move_index(mv)) for mv in moves]

    def apply_move(self, state: GameState, move):
        if isinstance(move, int) and 0 <= move < 64:
            mv = int(self._lib.az_move_bit(move))
        else:
            mv = int(move)
        out_black = ctypes.c_uint64()
        out_white = ctypes.c_uint64()
        self._lib.az_apply_move(
            state.black, state.white, ctypes.c_char(state.turn.encode("ascii")), mv, out_black, out_white
        )
        next_turn = "w" if state.turn == "b" else "b"
        return GameState(int(out_black.value), int(out_white.value), next_turn)

    def result(self, state: GameState):
        res = self._lib.az_result(state.black, state.white)
        return res.decode("ascii")

    def move_bit(self, index: int):
        return int(self._lib.az_move_bit(index))

    def move_index(self, bit: int):
        return int(self._lib.az_move_index(bit))


def encode_state(state: GameState):
    planes = torch.zeros((2, 4, 4, 4), dtype=torch.float32)
    if state.turn == "b":
        cur = state.black
        opp = state.white
    else:
        cur = state.white
        opp = state.black

    _fill_plane(planes[0], cur)
    _fill_plane(planes[1], opp)
    return planes


def _build_layer_lut():
    lut = torch.zeros((1 << 16, 4, 4), dtype=torch.float32)
    bits = torch.arange(1 << 16, dtype=torch.int32)
    for i in range(16):
        y = i // 4
        x = i % 4
        lut[:, y, x] = ((bits >> (15 - i)) & 1).float()
    return lut


_LAYER_LUT = _build_layer_lut()


def _fill_plane(plane, board_bits: int):
    for layer in range(4):
        shift = 48 - layer * 16
        chunk = (board_bits >> shift) & 0xFFFF
        plane[layer] = _LAYER_LUT[chunk]


def render_board(state: GameState):
    chars = []
    for idx in range(64):
        bit = 1 << (63 - idx)
        if state.black & bit:
            chars.append("X")
        elif state.white & bit:
            chars.append("O")
        else:
            chars.append(".")
    lines = []
    for layer in range(4):
        lines.append(f"layer {layer}")
        base = layer * 16
        for row in range(4):
            off = base + row * 4
            lines.append(" ".join(chars[off : off + 4]))
        lines.append("")
    return "\n".join(lines)
