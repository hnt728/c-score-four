#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include <limits.h>
#include <getopt.h>
#include <omp.h>

typedef unsigned long ulong;

ulong decimal2binary(int decimal_num);
int binary2decimal(ulong binary_num);

// ----------------------------
// RNG (xoshiro256** + splitmix64)
// ----------------------------
typedef struct {
    uint64_t s[4];
} Rng;

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t splitmix64_next(uint64_t *x) {
    uint64_t z = (*x += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static inline void rng_seed(Rng *rng, uint64_t seed) {
    uint64_t x = seed;
    rng->s[0] = splitmix64_next(&x);
    rng->s[1] = splitmix64_next(&x);
    rng->s[2] = splitmix64_next(&x);
    rng->s[3] = splitmix64_next(&x);
    if ((rng->s[0] | rng->s[1] | rng->s[2] | rng->s[3]) == 0) {
        rng->s[0] = UINT64_C(1);
    }
}

static inline uint64_t rng_next_u64(Rng *rng) {
    // xoshiro256** (Blackman & Vigna, 2018)
    const uint64_t result = rotl64(rng->s[1] * 5, 7) * 9;
    const uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = rotl64(rng->s[3], 45);
    return result;
}

static inline uint32_t rng_next_u32(Rng *rng) {
    return (uint32_t)(rng_next_u64(rng) >> 32);
}

static inline uint32_t rng_uniform_u32(Rng *rng, uint32_t bound) {
    // Lemire (2019): fast unbiased modulo reduction
    if (bound == 0) return 0;
    uint64_t x = (uint64_t)rng_next_u32(rng);
    uint64_t m = x * (uint64_t)bound;
    uint32_t l = (uint32_t)m;
    if (l < bound) {
        uint32_t t = (uint32_t)(-bound) % bound;
        while (l < t) {
            x = (uint64_t)rng_next_u32(rng);
            m = x * (uint64_t)bound;
            l = (uint32_t)m;
        }
    }
    return (uint32_t)(m >> 32);
}

static uint64_t auto_seed64(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t x = (uint64_t)ts.tv_sec ^ (uint64_t)ts.tv_nsec;
    x ^= (uint64_t)getpid() * UINT64_C(0x9e3779b97f4a7c15);
    x ^= (uint64_t)(uintptr_t)&ts;
    x ^= (uint64_t)(uintptr_t)&x;
    x ^= (uint64_t)(omp_get_wtime() * 1e9);
    // mix once through splitmix for good measure
    return splitmix64_next(&x);
}

const ulong conditions[76] = {
    0b1111000000000000000000000000000000000000000000000000000000000000,
    0b0000111100000000000000000000000000000000000000000000000000000000,
    0b0000000011110000000000000000000000000000000000000000000000000000,
    0b0000000000001111000000000000000000000000000000000000000000000000,
    0b0000000000000000111100000000000000000000000000000000000000000000,
    0b0000000000000000000011110000000000000000000000000000000000000000,
    0b0000000000000000000000001111000000000000000000000000000000000000,
    0b0000000000000000000000000000111100000000000000000000000000000000,
    0b0000000000000000000000000000000011110000000000000000000000000000,
    0b0000000000000000000000000000000000001111000000000000000000000000,
    0b0000000000000000000000000000000000000000111100000000000000000000,
    0b0000000000000000000000000000000000000000000011110000000000000000,
    0b0000000000000000000000000000000000000000000000001111000000000000,
    0b0000000000000000000000000000000000000000000000000000111100000000,
    0b0000000000000000000000000000000000000000000000000000000011110000,
    0b0000000000000000000000000000000000000000000000000000000000001111,
    0b1000100010001000000000000000000000000000000000000000000000000000,
    0b0100010001000100000000000000000000000000000000000000000000000000,
    0b0010001000100010000000000000000000000000000000000000000000000000,
    0b0001000100010001000000000000000000000000000000000000000000000000,
    0b0000000000000000100010001000100000000000000000000000000000000000,
    0b0000000000000000010001000100010000000000000000000000000000000000,
    0b0000000000000000001000100010001000000000000000000000000000000000,
    0b0000000000000000000100010001000100000000000000000000000000000000,
    0b0000000000000000000000000000000010001000100010000000000000000000,
    0b0000000000000000000000000000000001000100010001000000000000000000,
    0b0000000000000000000000000000000000100010001000100000000000000000,
    0b0000000000000000000000000000000000010001000100010000000000000000,
    0b0000000000000000000000000000000000000000000000001000100010001000,
    0b0000000000000000000000000000000000000000000000000100010001000100,
    0b0000000000000000000000000000000000000000000000000010001000100010,
    0b0000000000000000000000000000000000000000000000000001000100010001,
    0b1000000000000000100000000000000010000000000000001000000000000000,
    0b0100000000000000010000000000000001000000000000000100000000000000,
    0b0010000000000000001000000000000000100000000000000010000000000000,
    0b0001000000000000000100000000000000010000000000000001000000000000,
    0b0000100000000000000010000000000000001000000000000000100000000000,
    0b0000010000000000000001000000000000000100000000000000010000000000,
    0b0000001000000000000000100000000000000010000000000000001000000000,
    0b0000000100000000000000010000000000000001000000000000000100000000,
    0b0000000010000000000000001000000000000000100000000000000010000000,
    0b0000000001000000000000000100000000000000010000000000000001000000,
    0b0000000000100000000000000010000000000000001000000000000000100000,
    0b0000000000010000000000000001000000000000000100000000000000010000,
    0b0000000000001000000000000000100000000000000010000000000000001000,
    0b0000000000000100000000000000010000000000000001000000000000000100,
    0b0000000000000010000000000000001000000000000000100000000000000010,
    0b0000000000000001000000000000000100000000000000010000000000000001,
    0b1000010000100001000000000000000000000000000000000000000000000000,
    0b0000000000000000100001000010000100000000000000000000000000000000,
    0b0000000000000000000000000000000010000100001000010000000000000000,
    0b0000000000000000000000000000000000000000000000001000010000100001,
    0b0001001001001000000000000000000000000000000000000000000000000000,
    0b0000000000000000000100100100100000000000000000000000000000000000,
    0b0000000000000000000000000000000000010010010010000000000000000000,
    0b0000000000000000000000000000000000000000000000000001001001001000,
    0b0001000000000000001000000000000001000000000000001000000000000000,
    0b0000000100000000000000100000000000000100000000000000100000000000,
    0b0000000000010000000000000010000000000000010000000000000010000000,
    0b0000000000000001000000000000001000000000000001000000000000001000,
    0b1000000000000000010000000000000000100000000000000001000000000000,
    0b0000100000000000000001000000000000000010000000000000000100000000,
    0b0000000010000000000000000100000000000000000100000000000000010000,
    0b0000000000001000000000000000010000000000000000010000000000000001,
    0b1000000000000000000010000000000000000000100000000000000000001000,
    0b0100000000000000000001000000000000000000010000000000000000000100,
    0b0010000000000000000000100000000000000000001000000000000000000010,
    0b0001000000000000000000010000000000000000000100000000000000000001,
    0b0000000000001000000000001000000000001000000000001000000000000000,
    0b0000000000000100000000000100000000000100000000000100000000000000,
    0b0000000000000010000000000010000000000010000000000010000000000000,
    0b0000000000000001000000000001000000000001000000000001000000000000,
    0b0000000000000001000000000010000000000100000000001000000000000000,
    0b1000000000000000000001000000000000000000001000000000000000000001,
    0b0001000000000000000000100000000000000000100000000000000000001000,
    0b0000000000001000000000000100000000000010000000000001000000000000
};

// For fast "did the last move win?" checks in rollouts/expansions.
#define MAX_CELL_LINES 16
static uint8_t g_cell_lines_count[64];
static ulong g_cell_lines[64][MAX_CELL_LINES];

static void init_cell_lines(void) {
    memset(g_cell_lines_count, 0, sizeof(g_cell_lines_count));
    const ulong top_bit = (ulong)UINT64_C(0x8000000000000000);
    for (int li = 0; li < 76; li++) {
        const ulong mask = conditions[li];
        for (int idx = 0; idx < 64; idx++) {
            const ulong bit = top_bit >> idx;
            if ((mask & bit) == 0) continue;
            uint8_t n = g_cell_lines_count[idx];
            if (n < MAX_CELL_LINES) {
                g_cell_lines[idx][n] = mask;
                g_cell_lines_count[idx] = (uint8_t)(n + 1);
            }
        }
    }
}

static inline bool is_win_after_move(const ulong player_board_after, const ulong last_move_bit) {
    const int idx = binary2decimal(last_move_bit);
    if (idx < 0 || idx >= 64) return false;
    const uint8_t n = g_cell_lines_count[idx];
    for (uint8_t i = 0; i < n; i++) {
        const ulong mask = g_cell_lines[idx][i];
        if ((player_board_after & mask) == mask) {
            return true;
        }
    }
    return false;
}

ulong decimal2binary(int decimal_num) {
    if (decimal_num < 0 || decimal_num >= 64) {
        return 0;
    }
    return (ulong)UINT64_C(0x8000000000000000) >> (unsigned)decimal_num;
}

int binary2decimal(ulong binary_num) {
    if (binary_num == 0) {
        return -1;
    }
#if ULONG_MAX == 0xffffffffffffffffUL
    return (int)__builtin_clzl(binary_num);
#else
    // Fallback: shift until MSB.
    int x = 0;
    ulong msb = (ulong)1 << ((int)(sizeof(ulong) * 8) - 1);
    while ((binary_num & msb) == 0) {
        binary_num <<= 1;
        x++;
    }
    return x;
#endif
}

void binary2arrayboard(ulong binary, signed char board[64]) {
    ulong bottom_bit = 0x0000000000000001;
    for (int i=0; i<64; i++) {
        board[63-i] = (binary >> i) & bottom_bit;
    }
}

char convert_turn(char turn) {
    if (turn == 'b') {
        return 'w';
    } else if (turn == 'w') {
        return 'b';
    } else {
        return 'n';
    }
}

ulong get_possible_pos_board(const ulong black_board, const ulong white_board) {
    ulong board = black_board | white_board;
    ulong first_floor = 0b1111111111111111000000000000000000000000000000000000000000000000;
    return ((board >> 16) ^ board) ^ first_floor;
}

bool is_possible_pos(const ulong black_board, const ulong white_board, const ulong index) {
    return 0 < (get_possible_pos_board(black_board, white_board) & index);
}

int get_possible_poses_binary(const ulong black_board, const ulong white_board, ulong array[16]) {
    ulong board = 0b1000000000000000000000000000000000000000000000000000000000000000;
    ulong possible_pos_board = get_possible_pos_board(black_board, white_board);
    int array_index = 0;
    for (int i=0; i<64; i++) {
        if ((possible_pos_board & (board >> i)) != 0) {
            array[array_index] = board >> i;
            array_index++;
        }
    }
    return array_index;
}

char which_is_win(const ulong black_board, const ulong white_board) {
    for (int i=0; i<76; i++) {
        if (__builtin_popcountl(black_board & conditions[i]) == 4) {
            return 'b';
        } else if (__builtin_popcountl(white_board & conditions[i]) == 4) {
            return 'w';
        }
    }
    if (get_possible_pos_board(black_board, white_board) == 0) {
        return 'd';
    }

    return 'n';
}

void print_board(const ulong black_board, const ulong white_board) {
    ulong bottom_bit = 0x8000000000000000;
    char chars[64][18];
    for (int i=0; i<64; i++) {
        if (((black_board << i) & bottom_bit ) == 0x8000000000000000) {
            strcpy(chars[i], "\x1b[40mX\x1b[42m\0");
        } else if (((white_board << i) & bottom_bit) == 0x8000000000000000) {
            strcpy(chars[i], "\x1b[47mO\x1b[42m\0");
        } else {
            strcpy(chars[i], " ");
        }
    }
    printf("\x1b[42m|||||||||||||||||\n");
    printf("=================\n");
    for (int i=0; i<4; i++) {
        printf("| %s | %s | %s | %s |\n", chars[i*16], chars[i*16+1], chars[i*16+2], chars[i*16+3]);
        printf("-----------------\n");
        printf("| %s | %s | %s | %s |\n", chars[i*16+4], chars[i*16+5], chars[i*16+6], chars[i*16+7]);
        printf("-----------------\n");
        printf("| %s | %s | %s | %s |\n", chars[i*16+8], chars[i*16+9], chars[i*16+10], chars[i*16+11]);
        printf("-----------------\n");
        printf("| %s | %s | %s | %s |\n", chars[i*16+12], chars[i*16+13], chars[i*16+14], chars[i*16+15]);
        printf("=================\n");
    }
    printf("|||||||||||||||||\x1b[49m\n");
}

ulong human_act(const ulong black_board, const ulong white_board) {
    int input = 0;
    do {
        printf("Enter index:");
        int result = scanf("%d", &input);
        printf("result: %d\n", result);
        printf("\n");
    } while (!is_possible_pos(black_board, white_board, decimal2binary(input)));
    return decimal2binary(input);
}

ulong random_act(const ulong black_board, const ulong white_board, Rng *rng) {
    ulong possible_poses[16];
    int possible_poses_len = get_possible_poses_binary(black_board, white_board, possible_poses);
    if (possible_poses_len <= 0) {
        return 0;
    }
    return possible_poses[(int)rng_uniform_u32(rng, (uint32_t)possible_poses_len)];
}

int max_index(int nums[], int n) {
    int max_value;
    int max_index;

    max_value = nums[0];
    max_index = 0;

    for (int i=0; i<n; i++) {
        if (nums[i] > max_value) {
            max_value = nums[i];
            max_index = i;
        }
    }

    return max_index;
}

int get_children(const ulong black_board, const ulong white_board, char my_turn, ulong children[16][2]) {
    int children_index = 0;
    ulong possible_pos_boards[16];
    int possible_pos_boards_len = get_possible_poses_binary(black_board, white_board, possible_pos_boards);
    for (int i=0; i<possible_pos_boards_len; i++) {
        ulong new_black_board = black_board;
        ulong new_white_board = white_board;
        if (my_turn == 'b') {
            new_black_board |= possible_pos_boards[i];
        } else {
            new_white_board |= possible_pos_boards[i];
        }
        children[children_index][0] = new_black_board;
        children[children_index][1] = new_white_board;
        children_index++;
    }

    return children_index;
}

int get_score(const ulong black_board, const ulong white_board, char my_turn) {
    int score = 0;
    if (which_is_win(black_board, white_board) == my_turn) {
        score = 100;
    } else if (which_is_win(black_board, white_board) == convert_turn(my_turn)) {
        score =  -100;
    } else if (which_is_win(black_board, white_board) == 'd') {
        score =  0;
    } else {
        if (my_turn == 'b') {
            for (int i=0; i<76; i++) {
                if (((black_board & conditions[i]) > 0) && ((white_board & conditions[i]) == 0)) {
                    score++;
                } else if (((black_board & conditions[i]) == 0) && ((white_board & conditions[i]) > 0)) {
                    score--;
                }
            }
        } else {
            for (int i=0; i<76; i++) {
                if (((black_board & conditions[i]) > 0) && ((white_board & conditions[i]) == 0)) {
                    score--;
                } else if (((black_board & conditions[i]) == 0) && ((white_board & conditions[i]) > 0)) {
                    score++;
                }
            }
        }
    }

    return score;
}

int alphabeta(const ulong black_board, const ulong white_board, int depth, int alpha, int beta,
                char turn, char my_turn) {
    if (depth == 0 || which_is_win(black_board, white_board) != 'n') {
        return get_score(black_board, white_board, my_turn);
    }

    ulong children_nodes[16][2];
    int children_nodes_len = get_children(black_board, white_board, turn, children_nodes);
    if (children_nodes_len == 0) {
        return get_score(black_board, white_board, my_turn);
    }

    int scores[16];
    for (int i=0; i<children_nodes_len; i++) {
        scores[i] = get_score(children_nodes[i][0], children_nodes[i][1], my_turn);
    }

    // sort
    for (int i=0; i<children_nodes_len-1; i++) {
        for (int j=i+1; j<children_nodes_len; j++) {
            if ((turn == my_turn && scores[i] < scores[j])
                || (turn != my_turn && scores[i] > scores[j])) {
                int temp_score = scores[i];
                scores[i] = scores[j];
                scores[j] = temp_score;
                ulong temp_black = children_nodes[i][0];
                ulong temp_white = children_nodes[i][1];
                children_nodes[i][0] = children_nodes[j][0];
                children_nodes[i][1] = children_nodes[j][1];
                children_nodes[j][0] = temp_black;
                children_nodes[j][1] = temp_white;
            }
        }
    }

    if (turn == my_turn) {
        int value = -10000;
        for (int i = 0; i < children_nodes_len; i++) {
            int score = alphabeta(children_nodes[i][0], children_nodes[i][1], depth-1, alpha,
                                    beta, convert_turn(turn), my_turn);
            if (score > value) {
                value = score;
            }
            if (value > alpha) {
                alpha = value;
            }
            if (alpha >= beta) {
                break;
            }
        }
        return value;
    } else {
        int value = 10000;
        for (int i=0; i<children_nodes_len; i++) {
            int score = alphabeta(children_nodes[i][0], children_nodes[i][1], depth-1, alpha, beta,
                                    convert_turn(turn), my_turn);
            if (score < value) {
                value = score;
            }
            if (value < beta) {
                beta = value;
            }
            if (alpha >= beta) {
                break;
            }
        }
        return value;
    }
}

ulong minmax_act(const ulong black_board, const ulong white_board, char my_turn, int depth) {
    ulong next_boards[16][2];
    int next_boards_len = get_children(black_board, white_board, my_turn, next_boards);

    int scores[16];
    #pragma omp parallel for
    for (int i=0; i<next_boards_len; i++) {
        int score = alphabeta(next_boards[i][0], next_boards[i][1], depth, -10000, 10000, convert_turn(my_turn), my_turn);
        printf("%16lx: %d\n", next_boards[i][0] | next_boards[i][1], score);
        scores[i] = score;
    }

    return (black_board | white_board) ^ (next_boards[max_index(scores, next_boards_len)][0]
                | next_boards[max_index(scores, next_boards_len)][1]);
}

// ----------------------------
// MCTS (root-parallel UCT)
// ----------------------------
typedef struct {
    long long iterations;      // <=0: no iteration limit (requires time_ms > 0)
    int time_ms;               // <=0: no time limit
    int threads;               // <=0: omp_get_max_threads()
    double c;                  // UCT exploration constant
    int rollout_max_depth;     // max rollout length
    long long max_nodes;       // per-thread node cap (<=0: auto)
    int verbose;               // 0: quiet, >=1: per-move stats
    uint64_t seed;             // 0: auto
} MctsConfig;

typedef struct {
    ulong black;
    ulong white;
    int parent;
    uint32_t visits;
    float wins;                // from root player's perspective
    uint8_t child_count;
    uint32_t children[16];
    char turn;                 // player to move at this node ('b'/'w')
    char result;               // 'n' ongoing, 'b','w','d'
} MctsNode;

static inline float reward_from_result(char result, char root_turn) {
    if (result == 'd') return 0.5f;
    if (result == root_turn) return 1.0f;
    return 0.0f;
}

static inline uint32_t mcts_select_child_uct(const MctsNode *nodes, uint32_t node_idx, double c) {
    const MctsNode *node = &nodes[node_idx];
    const double log_parent = log((double)node->visits + 1.0);
    uint32_t best_child = node->children[0];
    double best = -1e300;
    for (uint8_t i = 0; i < node->child_count; i++) {
        const uint32_t ci = node->children[i];
        const MctsNode *child = &nodes[ci];
        if (child->visits == 0) {
            return ci;
        }
        const double mean = (double)child->wins / (double)child->visits;
        const double uct = mean + c * sqrt(log_parent / (double)child->visits);
        if (uct > best) {
            best = uct;
            best_child = ci;
        }
    }
    return best_child;
}

static inline bool mcts_move_is_expanded(const MctsNode *nodes, uint32_t parent_idx, const ulong move_bit) {
    const MctsNode *parent = &nodes[parent_idx];
    const ulong parent_occ = parent->black | parent->white;
    for (uint8_t i = 0; i < parent->child_count; i++) {
        const MctsNode *child = &nodes[parent->children[i]];
        const ulong child_occ = child->black | child->white;
        if ((child_occ ^ parent_occ) == move_bit) {
            return true;
        }
    }
    return false;
}

static inline ulong mcts_rollout_pick_move(ulong black, ulong white, char turn, Rng *rng) {
    ulong moves[16];
    const int n = get_possible_poses_binary(black, white, moves);
    if (n <= 0) return 0;

    // 1) winning move
    for (int i = 0; i < n; i++) {
        const ulong mv = moves[i];
        if (turn == 'b') {
            const ulong b2 = black | mv;
            if (is_win_after_move(b2, mv)) return mv;
        } else {
            const ulong w2 = white | mv;
            if (is_win_after_move(w2, mv)) return mv;
        }
    }

    // 2) block opponent's immediate win
    const char opp = convert_turn(turn);
    for (int i = 0; i < n; i++) {
        const ulong mv = moves[i];
        if (opp == 'b') {
            const ulong b2 = black | mv;
            if (is_win_after_move(b2, mv)) return mv;
        } else {
            const ulong w2 = white | mv;
            if (is_win_after_move(w2, mv)) return mv;
        }
    }

    // 3) random
    return moves[(int)rng_uniform_u32(rng, (uint32_t)n)];
}

static inline float mcts_rollout_value(ulong black, ulong white, char turn, char root_turn, int max_depth, Rng *rng) {
    char res = which_is_win(black, white);
    if (res != 'n') {
        return reward_from_result(res, root_turn);
    }

    for (int d = 0; d < max_depth; d++) {
        const ulong mv = mcts_rollout_pick_move(black, white, turn, rng);
        if (mv == 0) {
            return 0.5f;
        }

        if (turn == 'b') {
            black |= mv;
            if (is_win_after_move(black, mv)) {
                return reward_from_result('b', root_turn);
            }
        } else {
            white |= mv;
            if (is_win_after_move(white, mv)) {
                return reward_from_result('w', root_turn);
            }
        }
        if (get_possible_pos_board(black, white) == 0) {
            return 0.5f;
        }
        turn = convert_turn(turn);
    }

    // Depth cutoff: cheap heuristic as a small bias around 0.5.
    const int score = get_score(black, white, root_turn);
    const double v = 0.5 + 0.25 * tanh((double)score / 20.0);
    if (v <= 0.0) return 0.0f;
    if (v >= 1.0) return 1.0f;
    return (float)v;
}

static ulong mcts_act(const ulong black_board, const ulong white_board, char my_turn, const MctsConfig *cfg) {
    const int threads = (cfg->threads > 0) ? cfg->threads : omp_get_max_threads();
    const double start = omp_get_wtime();
    const double end_time = (cfg->time_ms > 0) ? (start + (double)cfg->time_ms / 1000.0) : 1e300;
    const long long iter_target = (cfg->iterations > 0) ? cfg->iterations : LLONG_MAX;

    ulong root_moves[16];
    const int root_moves_len = get_possible_poses_binary(black_board, white_board, root_moves);
    if (root_moves_len <= 0) return 0;

    const uint64_t base_seed = (cfg->seed != 0) ? cfg->seed : auto_seed64();

    long long total_visits[16];
    double total_wins[16];
    for (int i = 0; i < root_moves_len; i++) {
        total_visits[i] = 0;
        total_wins[i] = 0.0;
    }

    long long sims_done = 0;
    long long nodes_used_sum = 0;

    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        Rng rng;
        rng_seed(&rng, base_seed + (uint64_t)tid * UINT64_C(0x9e3779b97f4a7c15));

        long long per_thread_nodes;
        if (cfg->max_nodes > 0) {
            per_thread_nodes = cfg->max_nodes;
        } else if (cfg->iterations > 0) {
            const long long it_pt = (cfg->iterations + threads - 1) / threads;
            per_thread_nodes = it_pt + 2048;
        } else {
            per_thread_nodes = 100000;
        }
        if (per_thread_nodes < 4096) per_thread_nodes = 4096;
        if (per_thread_nodes > 2000000) per_thread_nodes = 2000000;

        MctsNode *nodes = (MctsNode*)calloc((size_t)per_thread_nodes, sizeof(MctsNode));
        if (!nodes) {
            // OOM: best-effort fallback to a random legal move.
            #pragma omp critical
            {
                if (cfg->verbose >= 1) {
                    fprintf(stderr, "mcts: OOM (thread=%d), falling back to random.\n", tid);
                }
            }
        } else {
            uint32_t node_count = 1;
            nodes[0].black = black_board;
            nodes[0].white = white_board;
            nodes[0].parent = -1;
            nodes[0].turn = my_turn;
            nodes[0].result = which_is_win(black_board, white_board);

            long long local_sims = 0;
            long long pending = 0;
            while (1) {
                if ((pending & 0x3f) == 0) {
                    if (cfg->time_ms > 0 && omp_get_wtime() >= end_time) break;
                    if (cfg->iterations > 0) {
                        long long cur;
                        #pragma omp atomic read
                        cur = sims_done;
                        if (cur >= iter_target) break;
                    }
                }

                uint32_t cur = 0;
                // Selection
                while (1) {
                    MctsNode *n = &nodes[cur];
                    if (n->result != 'n') break;
                    ulong legal[16];
                    const int legal_len = get_possible_poses_binary(n->black, n->white, legal);
                    if (legal_len == 0) {
                        n->result = 'd';
                        break;
                    }
                    if (n->child_count < (uint8_t)legal_len && node_count < (uint32_t)per_thread_nodes) {
                        break; // expandable
                    }
                    if (n->child_count == 0) break;
                    cur = mcts_select_child_uct(nodes, cur, cfg->c);
                }

                // Expansion (at most 1 new node)
                MctsNode *n = &nodes[cur];
                if (n->result == 'n' && node_count < (uint32_t)per_thread_nodes) {
                    ulong legal[16];
                    const int legal_len = get_possible_poses_binary(n->black, n->white, legal);
                    if (legal_len > 0 && n->child_count < (uint8_t)legal_len) {
                        uint32_t start_i = rng_uniform_u32(&rng, (uint32_t)legal_len);
                        ulong chosen = 0;
                        for (int k = 0; k < legal_len; k++) {
                            const ulong mv = legal[(int)((start_i + (uint32_t)k) % (uint32_t)legal_len)];
                            if (!mcts_move_is_expanded(nodes, cur, mv)) {
                                chosen = mv;
                                break;
                            }
                        }
                        if (chosen != 0) {
                            const uint32_t child = node_count++;
                            nodes[child].black = n->black;
                            nodes[child].white = n->white;
                            nodes[child].parent = (int)cur;
                            nodes[child].turn = convert_turn(n->turn);

                            if (n->turn == 'b') {
                                nodes[child].black |= chosen;
                                if (is_win_after_move(nodes[child].black, chosen)) {
                                    nodes[child].result = 'b';
                                }
                            } else {
                                nodes[child].white |= chosen;
                                if (is_win_after_move(nodes[child].white, chosen)) {
                                    nodes[child].result = 'w';
                                }
                            }
                            if (nodes[child].result == 0) {
                                nodes[child].result = (get_possible_pos_board(nodes[child].black, nodes[child].white) == 0) ? 'd' : 'n';
                            }

                            n->children[n->child_count++] = child;
                            cur = child;
                        }
                    }
                }

                // Simulation
                const MctsNode *leaf = &nodes[cur];
                float value;
                if (leaf->result != 'n') {
                    value = reward_from_result(leaf->result, my_turn);
                } else {
                    value = mcts_rollout_value(leaf->black, leaf->white, leaf->turn, my_turn, cfg->rollout_max_depth, &rng);
                }

                // Backprop
                uint32_t bp = cur;
                while (1) {
                    nodes[bp].visits++;
                    nodes[bp].wins += value;
                    if (nodes[bp].parent < 0) break;
                    bp = (uint32_t)nodes[bp].parent;
                }

                local_sims++;
                pending++;
                if ((pending & 0x3f) == 0) {
                    #pragma omp atomic
                    sims_done += 64;
                    pending = 0;
                }
                if (cfg->time_ms > 0 && omp_get_wtime() >= end_time) {
                    break;
                }
            }

            if (pending > 0) {
                #pragma omp atomic
                sims_done += pending;
            }

            // Aggregate root stats into shared arrays.
            const ulong root_occ = (nodes[0].black | nodes[0].white);
            for (uint8_t i = 0; i < nodes[0].child_count; i++) {
                const uint32_t ci = nodes[0].children[i];
                const ulong mv = (nodes[ci].black | nodes[ci].white) ^ root_occ;
                int mi = -1;
                for (int j = 0; j < root_moves_len; j++) {
                    if (root_moves[j] == mv) { mi = j; break; }
                }
                if (mi >= 0) {
                    #pragma omp atomic
                    total_visits[mi] += (long long)nodes[ci].visits;
                    #pragma omp atomic
                    total_wins[mi] += (double)nodes[ci].wins;
                }
            }

            #pragma omp atomic
            nodes_used_sum += (long long)node_count;

            free(nodes);
        }
    }

    // Choose by max visits; tie-break by winrate.
    int best_i = 0;
    long long best_v = -1;
    double best_wr = -1.0;
    for (int i = 0; i < root_moves_len; i++) {
        const long long v = total_visits[i];
        const double wr = (v > 0) ? (total_wins[i] / (double)v) : 0.0;
        if (v > best_v || (v == best_v && wr > best_wr)) {
            best_v = v;
            best_wr = wr;
            best_i = i;
        }
    }

    if (cfg->verbose >= 1) {
        const double elapsed_ms = (omp_get_wtime() - start) * 1000.0;
        printf("mcts turn=%c sims=%lld time=%.1fms threads=%d C=%.6f rollout_depth=%d nodes_avg=%.0f\n",
               my_turn, sims_done, elapsed_ms, threads, cfg->c, cfg->rollout_max_depth,
               (threads > 0) ? ((double)nodes_used_sum / (double)threads) : 0.0);
        for (int i = 0; i < root_moves_len; i++) {
            const long long v = total_visits[i];
            const double wr = (v > 0) ? (total_wins[i] / (double)v) : 0.0;
            printf("  move=%2d visits=%8lld winrate=%.4f\n", binary2decimal(root_moves[i]), v, wr);
        }
    }

    return root_moves[best_i];
}

void game_start(char player1, char player2, bool enable_show_board, bool enable_show_result,
                int depth1, int depth2, const MctsConfig *mcts1, const MctsConfig *mcts2, uint64_t rng_seed64) {
    ulong black_board = 0;
    ulong white_board = 0;
    char now_player = player1;
    char now_player_turn = 'b';
    char result = 'n';
    int player1_depth = depth1;
    int player2_depth = depth2;
    Rng game_rng;
    rng_seed(&game_rng, rng_seed64);

    while (result == 'n') {
        unsigned long act = 0;
        printf("turn: %c %c\n", now_player_turn, now_player);
        if (now_player == 'h') {
            act = human_act(black_board, white_board);
        } else if (now_player == 'r') {
            act = random_act(black_board, white_board, &game_rng);
        } else if (now_player == 'm') {
            if (now_player_turn == 'b') {
                act = minmax_act(black_board, white_board, now_player_turn, player1_depth);
            } else {
                act = minmax_act(black_board, white_board, now_player_turn, player2_depth);
            }
        } else if (now_player == 'c') {
            const MctsConfig *cfg = (now_player_turn == 'b') ? mcts1 : mcts2;
            act = mcts_act(black_board, white_board, now_player_turn, cfg);
        }

        if (now_player_turn == 'b') {
            printf("black put %d\n", binary2decimal(act));
        } else if (now_player_turn == 'w') {
            printf("white put %d\n", binary2decimal(act));
        }

        if (now_player_turn == 'b') {
            black_board = black_board | act;
        } else if (now_player_turn == 'w') {
            white_board = white_board | act;
        }

        if (enable_show_board) {
            print_board(black_board, white_board);
        }

        now_player_turn = convert_turn(now_player_turn);
        if (now_player_turn == 'b') {
            now_player = player1;
        } else {
            now_player = player2;
        }

        result = which_is_win(black_board, white_board);
    }
    if (enable_show_result) {
        if (result == 'w') {
            printf("winner is white!\n");
        } else if (result == 'b') {
            printf("winner is black!\n");
        } else {
            printf("draw!\n");
        }
    }
}

int main(int argc, char *argv[]) {
    char player1 = 'h';
    char player2 = 'h';
    int depth1 = 0;
    int depth2 = 0;
    bool enable_show_board = true;
    bool enable_show_result = true;
    uint64_t program_seed = 0;

    MctsConfig mcts_global = {
        .iterations = 20000,
        .time_ms = 0,
        .threads = 0,
        .c = 1.41421356237,
        .rollout_max_depth = 64,
        .max_nodes = 0,
        .verbose = 1,
        .seed = 0,
    };
    MctsConfig mcts_p1 = mcts_global;
    MctsConfig mcts_p2 = mcts_global;

    enum {
        OPT_NO_BOARD = 1000,
        OPT_NO_RESULT,
        OPT_MCTS_ITERATIONS,
        OPT_MCTS_TIME_MS,
        OPT_MCTS_THREADS,
        OPT_MCTS_C,
        OPT_MCTS_ROLLOUT_DEPTH,
        OPT_MCTS_MAX_NODES,
        OPT_MCTS_VERBOSE,
        OPT_MCTS_SEED,
        OPT_P1_MCTS_ITERATIONS,
        OPT_P2_MCTS_ITERATIONS,
        OPT_P1_MCTS_TIME_MS,
        OPT_P2_MCTS_TIME_MS,
    };

    struct option long_options[] = {
        {"player1", required_argument, NULL, '1'},
        {"player2", required_argument, NULL, '2'},
        {"player1-depth", required_argument, NULL, 'd'},
        {"depth1", required_argument, NULL, 'd'},
        {"player2-depth", required_argument, NULL, 'D'},
        {"depth2", required_argument, NULL, 'D'},
        {"no-board", no_argument, NULL, OPT_NO_BOARD},
        {"no-result", no_argument, NULL, OPT_NO_RESULT},
        {"mcts-iterations", required_argument, NULL, OPT_MCTS_ITERATIONS},
        {"mcts-time-ms", required_argument, NULL, OPT_MCTS_TIME_MS},
        {"mcts-threads", required_argument, NULL, OPT_MCTS_THREADS},
        {"mcts-c", required_argument, NULL, OPT_MCTS_C},
        {"mcts-rollout-depth", required_argument, NULL, OPT_MCTS_ROLLOUT_DEPTH},
        {"mcts-max-nodes", required_argument, NULL, OPT_MCTS_MAX_NODES},
        {"mcts-verbose", required_argument, NULL, OPT_MCTS_VERBOSE},
        {"mcts-seed", required_argument, NULL, OPT_MCTS_SEED},
        {"player1-mcts-iterations", required_argument, NULL, OPT_P1_MCTS_ITERATIONS},
        {"player2-mcts-iterations", required_argument, NULL, OPT_P2_MCTS_ITERATIONS},
        {"player1-mcts-time-ms", required_argument, NULL, OPT_P1_MCTS_TIME_MS},
        {"player2-mcts-time-ms", required_argument, NULL, OPT_P2_MCTS_TIME_MS},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "1:2:d:D:", long_options, &option_index)) != -1) {
        switch (c) {
            case '1':
                if (optarg[0] == 'h' || optarg[0] == 'r' || optarg[0] == 'm' || optarg[0] == 'c') {
                    player1 = optarg[0];
                } else {
                    fprintf(stderr, "Invalid player1 type. Use 'h', 'm', 'c', or 'r'.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            case '2':
                if (optarg[0] == 'h' || optarg[0] == 'r' || optarg[0] == 'm' || optarg[0] == 'c') {
                    player2 = optarg[0];
                } else {
                    fprintf(stderr, "Invalid player2 type. Use 'h', 'm', 'c', or 'r'.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            case 'd':
                depth1 = atoi(optarg);
                break;
            case 'D':
                depth2 = atoi(optarg);
                break;
            case OPT_NO_BOARD:
                enable_show_board = false;
                break;
            case OPT_NO_RESULT:
                enable_show_result = false;
                break;
            case OPT_MCTS_ITERATIONS: {
                long long v = strtoll(optarg, NULL, 10);
                mcts_global.iterations = v;
                mcts_p1.iterations = v;
                mcts_p2.iterations = v;
                break;
            }
            case OPT_MCTS_TIME_MS: {
                int v = (int)strtol(optarg, NULL, 10);
                mcts_global.time_ms = v;
                mcts_p1.time_ms = v;
                mcts_p2.time_ms = v;
                break;
            }
            case OPT_MCTS_THREADS: {
                int v = (int)strtol(optarg, NULL, 10);
                mcts_global.threads = v;
                mcts_p1.threads = v;
                mcts_p2.threads = v;
                break;
            }
            case OPT_MCTS_C: {
                double v = strtod(optarg, NULL);
                mcts_global.c = v;
                mcts_p1.c = v;
                mcts_p2.c = v;
                break;
            }
            case OPT_MCTS_ROLLOUT_DEPTH: {
                int v = (int)strtol(optarg, NULL, 10);
                mcts_global.rollout_max_depth = v;
                mcts_p1.rollout_max_depth = v;
                mcts_p2.rollout_max_depth = v;
                break;
            }
            case OPT_MCTS_MAX_NODES: {
                long long v = strtoll(optarg, NULL, 10);
                mcts_global.max_nodes = v;
                mcts_p1.max_nodes = v;
                mcts_p2.max_nodes = v;
                break;
            }
            case OPT_MCTS_VERBOSE: {
                int v = (int)strtol(optarg, NULL, 10);
                mcts_global.verbose = v;
                mcts_p1.verbose = v;
                mcts_p2.verbose = v;
                break;
            }
            case OPT_MCTS_SEED: {
                uint64_t v = (uint64_t)strtoull(optarg, NULL, 10);
                mcts_global.seed = v;
                mcts_p1.seed = v;
                mcts_p2.seed = v;
                program_seed = v;
                break;
            }
            case OPT_P1_MCTS_ITERATIONS:
                mcts_p1.iterations = strtoll(optarg, NULL, 10);
                break;
            case OPT_P2_MCTS_ITERATIONS:
                mcts_p2.iterations = strtoll(optarg, NULL, 10);
                break;
            case OPT_P1_MCTS_TIME_MS:
                mcts_p1.time_ms = (int)strtol(optarg, NULL, 10);
                break;
            case OPT_P2_MCTS_TIME_MS:
                mcts_p2.time_ms = (int)strtol(optarg, NULL, 10);
                break;
            default:
                fprintf(stderr, "Usage: %s --player1 [h|m|c|r] --player2 [h|m|c|r] [--player1-depth N] [--player2-depth N] [--mcts-* ...]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (player1 == 'm' && depth1 <= 0) {
        fprintf(stderr, "Error: When using 'm' for player1, you must specify --player1-depth.\n");
        exit(EXIT_FAILURE);
    }
    if (player2 == 'm' && depth2 <= 0) {
        fprintf(stderr, "Error: When using 'm' for player2, you must specify --player2-depth.\n");
        exit(EXIT_FAILURE);
    }
    if ((player1 == 'c' && mcts_p1.iterations <= 0 && mcts_p1.time_ms <= 0) ||
        (player2 == 'c' && mcts_p2.iterations <= 0 && mcts_p2.time_ms <= 0)) {
        fprintf(stderr, "Error: When using 'c' (MCTS) with --mcts-iterations <= 0, you must specify --mcts-time-ms (or per-player override).\n");
        exit(EXIT_FAILURE);
    }
    if (mcts_p1.rollout_max_depth <= 0) mcts_p1.rollout_max_depth = 64;
    if (mcts_p2.rollout_max_depth <= 0) mcts_p2.rollout_max_depth = 64;
    if (mcts_p1.c <= 0.0) mcts_p1.c = 1.41421356237;
    if (mcts_p2.c <= 0.0) mcts_p2.c = 1.41421356237;

    init_cell_lines();
    if (program_seed == 0) {
        program_seed = (mcts_global.seed != 0) ? mcts_global.seed : auto_seed64();
    }

    printf("player1: %c\n", player1);
    printf("player2: %c\n", player2);
    printf("player1-depth: %d\n", depth1);
    printf("player2-depth: %d\n", depth2);
    game_start(player1, player2, enable_show_board, enable_show_result, depth1, depth2, &mcts_p1, &mcts_p2, program_seed ^ UINT64_C(0x243f6a8885a308d3));
}
