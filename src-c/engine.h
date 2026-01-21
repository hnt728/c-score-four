#ifndef SCORE_FOUR_ENGINE_H
#define SCORE_FOUR_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize internal lookup tables. Must be called once before other APIs.
void az_init(void);

// Returns number of legal moves. out_moves must have capacity 16.
int az_legal_moves(uint64_t black, uint64_t white, uint64_t out_moves[16]);

// Returns 'b', 'w', 'd', or 'n' (ongoing).
char az_result(uint64_t black, uint64_t white);

// Apply move (bitboard) for turn ('b' or 'w'), output new boards.
void az_apply_move(uint64_t black, uint64_t white, char turn, uint64_t move,
                   uint64_t *out_black, uint64_t *out_white);

// Convert move bit <-> index (0-63).
int az_move_index(uint64_t move);
uint64_t az_move_bit(int index);

#ifdef __cplusplus
}
#endif

#endif
