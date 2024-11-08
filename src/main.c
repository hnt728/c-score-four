#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
#include <omp.h>

typedef unsigned long ulong;

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

ulong decimal2binary(int decimal_num) {
    ulong binary_num = 0b1000000000000000000000000000000000000000000000000000000000000000;
    binary_num = binary_num >> decimal_num;
    return binary_num;
}

int binary2decimal(ulong binary_num) {
    int x = 0;
    while (0 < binary_num) {
        binary_num = binary_num << 1;
        x++;
    }
    return x - 1;
}

void binary2arrayboard(ulong binary, signed char board[64]) {
    ulong bottom_bit = 0x0000000000000001;
    for (int i=0; i<64; i++) {
        board[63-i] = (binary >> i) & bottom_bit;
    }
}

int max(int a, int b) {
    if (a < b) {
        return b;
    } else {
        return a;
    }
}

int min(int a, int b) {
    if (a < b) {
        return a;
    } else {
        return b;
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

int get_possible_poses(const ulong black_board, const ulong white_board, int array[16]) {
    int array_index = 0;
    ulong board = 0b1000000000000000000000000000000000000000000000000000000000000000;
    for (int i=0; i<64; i++) {
        if (is_possible_pos(black_board, white_board, board >> i)) {
            array[array_index] = i;
            array_index++;
        }
    }
    return array_index;
}

int get_possible_poses_binary(const ulong black_board, const ulong white_board, ulong array[16]) {
    ulong possible_pos_board = get_possible_pos_board(black_board, white_board);
    ulong board = 0b1000000000000000000000000000000000000000000000000000000000000000;
    int array_index = 0;
    for (int i=0; i<64; i++) {
        if (is_possible_pos(black_board, white_board, board >> i)) {
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

ulong random_act(const ulong black_board, const ulong white_board) {
    ulong possible_poses[16];
    int possible_poses_len = get_possible_poses_binary(black_board, white_board, possible_poses);
    return possible_poses[rand() % possible_poses_len];
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

int max_value(int nums[], int n) {
    int max_val;

    max_val = nums[0];

    for (int i=1; i<n; i++) {
        if (nums[i] > max_val) {
            max_val = nums[i];
        }
    }

    return max_val;
}

int get_children(const ulong black_board, const ulong white_board, char my_turn, ulong children[16][2]) {
    int children_index = 0;
    ulong possible_pos_boards[16];
    int possible_pos_boards_len = get_possible_poses_binary(black_board, white_board, possible_pos_boards);
    ulong board = black_board | white_board;
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
    int value;
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

int minmax(const ulong black_board, const ulong white_board, int depth, char turn, char my_turn) {
    if (depth == 0) {
        return get_score(black_board, white_board, my_turn);
    }
    char winner = which_is_win(black_board, white_board);
    if (winner != 'n') {
        return get_score(black_board, white_board, my_turn);
    }
    ulong children_nodes[16][2];
    int children_nodes_len = get_children(black_board, white_board, turn, children_nodes);
    if (children_nodes_len == 0) {
        return get_score(black_board, white_board, my_turn);
    }
    if (turn == my_turn) {
        int max_score = -10000;
        for (int i=0; i<children_nodes_len; i++) {
            int score = minmax(children_nodes[i][0], children_nodes[i][1], depth-1,
                                convert_turn(turn), my_turn);
            if (score > max_score) {
                max_score = score;
            }
        }
        return max_score;
    } else {
        int min_score = 10000;
        for (int i=0; i<children_nodes_len; i++) {
            int score = minmax(children_nodes[i][0], children_nodes[i][1], depth-1,
                                convert_turn(turn), my_turn);
            if (score < min_score) {
                min_score = score;
            }
        }
        return min_score;
    }
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

void game_start(char player1, char player2, bool enable_show_board, bool enable_show_result, int depth1, int depth2) {
    ulong black_board = 0;
    ulong white_board = 0;
    char now_player = player1;
    char now_player_turn = 'b';
    char wich_is_win = 'n';
    char result = 'n';
    int player1_depth = depth1;
    int player2_depth = depth2;

    while (result == 'n') {
        unsigned long act = 0;
        printf("turn: %c %c\n", now_player_turn, now_player);
        if (now_player == 'h') {
            act = human_act(black_board, white_board);
        } else if (now_player == 'r') {
            act = random_act(black_board, white_board);
        } else if (now_player == 'm') {
            if (now_player_turn == 'b') {
                act = minmax_act(black_board, white_board, now_player_turn, player1_depth);
            } else {
                act = minmax_act(black_board, white_board, now_player_turn, player2_depth);
            }
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
    srand(time(NULL));

    char player1 = 'h';
    char player2 = 'h';
    int depth1 = 0;
    int depth2 = 0;
    bool enable_show_board = true;
    bool enable_show_result = true;

    struct option long_options[] = {
        {"player1", required_argument, NULL, '1'},
        {"player2", required_argument, NULL, '2'},
        {"player1-depth", required_argument, NULL, 'd'},
        {"player2-depth", required_argument, NULL, 'D'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "1:2:d:D:", long_options, &option_index)) != -1) {
        switch (c) {
            case '1':
                if (optarg[0] == 'h' || optarg[0] == 'r' || optarg[0] == 'm') {
                    player1 = optarg[0];
                } else {
                    fprintf(stderr, "Invalid player1 type. Use 'h', 'm', or 'r'.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            case '2':
                if (optarg[0] == 'h' || optarg[0] == 'r' || optarg[0] == 'm') {
                    player2 = optarg[0];
                } else {
                    fprintf(stderr, "Invalid player2 type. Use 'h', 'm', or 'r'.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            case 'd':
                depth1 = atoi(optarg);
                break;
            case 'D':
                depth2 = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s --player1 [h|m|r] --player2 [h|m|r] [--player1-depth number] [--player2-depth number]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    
    if (player1 == 'm' && player1 <= 0) {
        fprintf(stderr, "Error: When using 'm' for player1, you must specify --player1-depth.\n");
        exit(EXIT_FAILURE);
    }
    if (player2 == 'm' && player2 <= 0) {
        fprintf(stderr, "Error: When using 'm' for player2, you must specify --player2-depth.\n");
        exit(EXIT_FAILURE);
    }
    printf("player1: %c\n", player1);
    printf("player2: %c\n", player2);
    printf("player1-depth: %d\n", depth1);
    printf("player2-depth: %d\n", depth2);
    game_start(player1, player2, enable_show_board, enable_show_result, depth1, depth2);
}
