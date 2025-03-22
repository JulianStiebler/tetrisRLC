#include "api.h"
#include "board.h"
#include "pieces.h"
#include "nn.h"
#include "game.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Game state structure for internal use
typedef struct {
    int board[HEIGHT][WIDTH];
    int score;
    int current_piece_index;
    int possible_moves;
    int game_over;
    
    // Last move data
    int last_piece_index;
    int last_rotation;
    int last_x;
    int last_y;
    int last_lines_cleared;
    float last_td_error;
    
    // Available moves storage
    Action available_moves[100];
    int num_available_moves;
} GameState;

// Initialize game and neural network
TetrisRLState* init_game_and_nn(void) {
    TetrisRLState* state = (TetrisRLState*)malloc(sizeof(TetrisRLState));
    if (!state) return NULL;
    
    // Initialize tetris game state
    GameState* game_state = (GameState*)malloc(sizeof(GameState));
    if (!game_state) {
        free(state);
        return NULL;
    }
    
    // Clear board
    memset(game_state->board, 0, sizeof(game_state->board));
    game_state->score = 0;
    game_state->current_piece_index = rand() % NUM_PIECES;
    game_state->possible_moves = 0;
    game_state->game_over = 0;
    
    // Initialize last move data
    game_state->last_piece_index = 0;
    game_state->last_rotation = 0;
    game_state->last_x = 0;
    game_state->last_y = 0;
    game_state->last_lines_cleared = 0;
    game_state->last_td_error = 0.0f;
    game_state->num_available_moves = 0;
    
    // Initialize neural network
    nn_initialize();
    
    // Initialize pieces
    init_pieces();
    
    // Store pointers in the interface structure
    state->tetris_state = game_state;
    state->nn_state = NULL;
    
    return state;
}

double get_weight1(int hidden_idx, int input_idx) { 
    return W1[hidden_idx][input_idx];
}

double get_weight2(int hidden2_idx, int hidden1_idx) {
    return W2[hidden2_idx][hidden1_idx];
}

double get_weight3(int hidden2_idx) {
    return W3[0][hidden2_idx];  // W3 is a 1×HIDDEN_SIZE2 matrix
}

// Free resources
void free_game(TetrisRLState* state) {
    if (state) {
        if (state->tetris_state) {
            free(state->tetris_state);
        }
        free(state);
    }
}

// Get current board state
void get_board_state(TetrisRLState* state, int* buffer, int buffer_size) {
    if (!state || !buffer || buffer_size < WIDTH * HEIGHT) return;
    
    GameState* game_state = (GameState*)state->tetris_state;
    
    int idx = 0;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            buffer[idx++] = game_state->board[i][j];
        }
    }
}

// Get board dimensions
void get_board_dimensions(int* width, int* height) {
    if (width) *width = WIDTH;
    if (height) *height = HEIGHT;
}

// Get score
int get_score(TetrisRLState* state) {
    if (!state) return 0;
    
    GameState* game_state = (GameState*)state->tetris_state;
    return game_state->score;
}

// Get the next piece type
int get_next_piece(TetrisRLState* state) {
    if (!state) return 0;
    
    GameState* game_state = (GameState*)state->tetris_state;
    return game_state->current_piece_index;
}

// Get the number of possible moves
int get_possible_move_count(TetrisRLState* state) {
    if (!state) return 0;
    
    GameState* game_state = (GameState*)state->tetris_state;
    
    // If we already have the count, return it
    if (game_state->num_available_moves > 0) {
        return game_state->num_available_moves;
    }
    
    // Otherwise compute it
    Piece current_piece = pieces[game_state->current_piece_index];
    
    // Save current board state
    int saved_board[HEIGHT][WIDTH];
    copy_board(saved_board, board);
    
    // Copy game board to global board for move generation
    copy_board(board, game_state->board);
    
    // Generate all possible moves
    int num_actions = generate_actions(current_piece, game_state->available_moves, 100);
    game_state->num_available_moves = num_actions;
    
    // Restore global board
    copy_board(board, saved_board);
    
    return num_actions;
}

// Get neural network layer sizes
int get_nn_layer_sizes(TetrisRLState* state, int* sizes, int max_layers) {
    if (!sizes || max_layers < 3) return 0;
    
    sizes[0] = INPUT_SIZE;
    sizes[1] = HIDDEN_SIZE1;
    sizes[2] = HIDDEN_SIZE2;
    sizes[3] = OUTPUT_SIZE;
    
    return 4; // Number of layers including input layer
}

// And fix the get_nn_weights function
void get_nn_weights(TetrisRLState* state, int from_layer, int to_layer, float* buffer, int buffer_size) {
    if (!buffer) return;
    
    // Check which weights to get
    if (from_layer == 0 && to_layer == 1) {
        // Input -> Hidden1 weights
        if (buffer_size < INPUT_SIZE * HIDDEN_SIZE1) return;
        
        for (int i = 0; i < HIDDEN_SIZE1; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                buffer[i * INPUT_SIZE + j] = (float)W1[i][j];
            }
        }
    } else if (from_layer == 1 && to_layer == 2) {
        // Hidden1 -> Hidden2 weights
        if (buffer_size < HIDDEN_SIZE1 * HIDDEN_SIZE2) return;
        
        for (int i = 0; i < HIDDEN_SIZE2; i++) {
            for (int j = 0; j < HIDDEN_SIZE1; j++) {
                buffer[i * HIDDEN_SIZE1 + j] = (float)W2[i][j];
            }
        }
    } else if (from_layer == 2 && to_layer == 3) {
        // Hidden2 -> Output weights
        if (buffer_size < HIDDEN_SIZE2 * OUTPUT_SIZE) return;
        
        for (int i = 0; i < HIDDEN_SIZE2; i++) {
            buffer[i] = (float)W3[0][i];  // W3 is a 1×HIDDEN_SIZE2 matrix
        }
    }
}

void get_nn_activations(TetrisRLState* state, int layer, float* buffer, int buffer_size) {
    if (!state || !buffer) return;
    
    GameState* game_state = (GameState*)state->tetris_state;
    
    // Calculate features and forward pass
    double features[INPUT_SIZE];
    double hidden1[HIDDEN_SIZE1];
    double hidden2[HIDDEN_SIZE2];
    
    compute_features(game_state->board, features);
    
    // Calculate activations for the forward pass
    for (int i = 0; i < HIDDEN_SIZE1; i++) {
        hidden1[i] = b1[i];  // Start with bias
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden1[i] += features[j] * W1[i][j];
        }
        hidden1[i] = tanh(hidden1[i]);
    }
    
    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        hidden2[i] = b2[i];  // Start with bias
        for (int j = 0; j < HIDDEN_SIZE1; j++) {
            hidden2[i] += hidden1[j] * W2[i][j];
        }
        hidden2[i] = tanh(hidden2[i]);
    }
    
    double output = b3;  // Start with bias
    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        output += hidden2[i] * W3[0][i];  // W3 is a 1×HIDDEN_SIZE2 matrix
    }
    
    if (layer == 0) {
        int size = (buffer_size < INPUT_SIZE) ? buffer_size : INPUT_SIZE;
        for (int i = 0; i < size; i++) {
            buffer[i] = (float)features[i];
        }
    } else if (layer == 1) {
        int size = (buffer_size < HIDDEN_SIZE1) ? buffer_size : HIDDEN_SIZE1;
        for (int i = 0; i < size; i++) {
            buffer[i] = (float)hidden1[i];
        }
    } else if (layer == 2) {
        int size = (buffer_size < HIDDEN_SIZE2) ? buffer_size : HIDDEN_SIZE2;
        for (int i = 0; i < size; i++) {
            buffer[i] = (float)hidden2[i];
        }
    } else if (layer == 3) {
        if (buffer_size >= 1) {
            buffer[0] = (float)output;
        }
    }
}

// Get feature values for current board state
void get_feature_values(TetrisRLState* state, float* buffer, int buffer_size) {
    if (!state || !buffer || buffer_size < INPUT_SIZE) return;
    
    GameState* game_state = (GameState*)state->tetris_state;
    
    double features[INPUT_SIZE];
    compute_features(game_state->board, features);
    
    for (int i = 0; i < INPUT_SIZE && i < buffer_size; i++) {
        buffer[i] = (float)features[i];
    }
}

// Make a specific move
int make_move(TetrisRLState* state, int move_idx) {
    if (!state) return 0;
    
    GameState* game_state = (GameState*)state->tetris_state;
    
    // Ensure moves are generated
    if (game_state->num_available_moves == 0) {
        get_possible_move_count(state);
    }
    
    // Check if the move is valid
    if (move_idx < 0 || move_idx >= game_state->num_available_moves) {
        return 0;
    }
    
    // Store move details for animation
    game_state->last_piece_index = game_state->current_piece_index;
    game_state->last_rotation = game_state->available_moves[move_idx].rotation;
    game_state->last_x = game_state->available_moves[move_idx].x;
    game_state->last_y = game_state->available_moves[move_idx].y;
    game_state->last_lines_cleared = game_state->available_moves[move_idx].lines_cleared;
    
    // Apply the move
    copy_board(game_state->board, game_state->available_moves[move_idx].resulting_board);
    game_state->score += game_state->last_lines_cleared;
    
    // Generate a new piece
    game_state->current_piece_index = rand() % NUM_PIECES;
    game_state->num_available_moves = 0;  // Force recalculation of moves for new piece
    
    return 1;
}

/* --- Implementation for new functions --- */

void get_current_piece_position(TetrisRLState* state, int* piece_idx, int* rotation, int* x, int* y) {
    if (!state) return;
    
    GameState* game_state = (GameState*)state->tetris_state;
    
    if (piece_idx) *piece_idx = game_state->current_piece_index;
    // Default rotation and position values if none are set yet
    if (rotation) *rotation = 0;
    if (x) *x = WIDTH / 2 - 2;
    if (y) *y = 0;
}

void get_piece_shape(int piece_idx, int rotation, int* buffer, int buffer_size) {
    if (!buffer || buffer_size < 16 || piece_idx < 0 || piece_idx >= NUM_PIECES) return;
    
    if (rotation < 0 || rotation >= pieces[piece_idx].rotations) return;
    
    // Copy the 4x4 piece shape to the buffer
    int idx = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            buffer[idx++] = pieces[piece_idx].shapes[rotation][i][j];
        }
    }
}

int get_piece_count(void) {
    return NUM_PIECES;
}

int get_piece_rotations(int piece_idx) {
    if (piece_idx < 0 || piece_idx >= NUM_PIECES) return 0;
    return pieces[piece_idx].rotations;
}

int is_game_over(TetrisRLState* state) {
    if (!state) return 0;
    
    GameState* game_state = (GameState*)state->tetris_state;
    return game_state->game_over;
}

void get_move_info(TetrisRLState* state, int move_idx, int* rotation, int* x, int* y, int* lines_cleared, float* value) {
    if (!state) return;
    
    GameState* game_state = (GameState*)state->tetris_state;
    
    // Check if we have cached moves and if index is valid
    if (game_state->num_available_moves == 0 || move_idx < 0 || move_idx >= game_state->num_available_moves) {
        // Generate moves if needed
        Piece current_piece = pieces[game_state->current_piece_index];
        
        // Save current board state
        int saved_board[HEIGHT][WIDTH];
        copy_board(saved_board, board);
        
        // Copy game board to global board for move generation
        copy_board(board, game_state->board);
        
        // Generate all possible moves
        int num_actions = generate_actions(current_piece, game_state->available_moves, 100);
        game_state->num_available_moves = num_actions;
        
        // Restore global board
        copy_board(board, saved_board);
        
        // Check if move_idx is still invalid
        if (move_idx < 0 || move_idx >= game_state->num_available_moves) return;
    }
    
    // Return information about the move
    if (rotation) *rotation = game_state->available_moves[move_idx].rotation;
    if (x) *x = game_state->available_moves[move_idx].x;
    if (y) *y = game_state->available_moves[move_idx].y;
    if (lines_cleared) *lines_cleared = game_state->available_moves[move_idx].lines_cleared;
    if (value) *value = (float)game_state->available_moves[move_idx].value;
}

void get_last_move_info(TetrisRLState* state, int* piece_idx, int* rotation, int* x, int* y, int* lines_cleared) {
    if (!state) return;
    
    GameState* game_state = (GameState*)state->tetris_state;
    
    if (piece_idx) *piece_idx = game_state->last_piece_index;
    if (rotation) *rotation = game_state->last_rotation;
    if (x) *x = game_state->last_x;
    if (y) *y = game_state->last_y;
    if (lines_cleared) *lines_cleared = game_state->last_lines_cleared;
}

float get_last_td_error(TetrisRLState* state) {
    if (!state) return 0.0f;
    
    GameState* game_state = (GameState*)state->tetris_state;
    return game_state->last_td_error;
}

void get_training_parameters(TetrisRLState* state, float* alpha_ptr, float* gamma_ptr, float* epsilon_ptr, float* lambda_ptr) {
    if (alpha_ptr) *alpha_ptr = (float)alpha;
    if (gamma_ptr) *gamma_ptr = (float)_gamma;
    if (epsilon_ptr) *epsilon_ptr = (float)epsilon;
    if (lambda_ptr) *lambda_ptr = (float)lambda;
}

int generate_falling_animation(TetrisRLState* state, int piece_idx, int rotation, int x, int final_y, int* board_frames, int max_frames) {
    if (!state || !board_frames || max_frames <= 0) return 0;
    
    GameState* game_state = (GameState*)state->tetris_state;
    
    // Get piece shape
    if (piece_idx < 0 || piece_idx >= NUM_PIECES) return 0;
    if (rotation < 0 || rotation >= pieces[piece_idx].rotations) return 0;
    
    // Generate frames showing the piece falling
    int frame_count = 0;
    int steps = max_frames > final_y + 2 ? final_y + 2 : max_frames;
    
    for (int step = 0; step < steps && frame_count < max_frames; step++) {
        // Calculate the current y position of the piece
        int current_y = (step * final_y) / (steps - 1);
        
        // Create a copy of the board
        int temp_board[HEIGHT][WIDTH];
        copy_board(temp_board, game_state->board);
        
        // Place the piece on the temporary board
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (pieces[piece_idx].shapes[rotation][i][j]) {
                    int bx = x + j;
                    int by = current_y + i;
                    if (bx >= 0 && bx < WIDTH && by >= 0 && by < HEIGHT) {
                        // Use a special value (2) for the falling piece
                        temp_board[by][bx] = 2;
                    }
                }
            }
        }
        
        // Copy the frame to the output buffer
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                board_frames[frame_count * WIDTH * HEIGHT + y * WIDTH + x] = temp_board[y][x];
            }
        }
        frame_count++;
    }
    
    return frame_count;
}

void place_piece_for_preview(TetrisRLState* state, int piece_idx, int rotation, int x, int y, int* output_board) {
    if (!state || !output_board || piece_idx < 0 || piece_idx >= NUM_PIECES) return;
    if (rotation < 0 || rotation >= pieces[piece_idx].rotations) return;
    
    GameState* game_state = (GameState*)state->tetris_state;
    
    // Copy current board to output
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            output_board[i * WIDTH + j] = game_state->board[i][j];
        }
    }
    
    // Add the piece at the specified position
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (pieces[piece_idx].shapes[rotation][i][j]) {
                int bx = x + j;
                int by = y + i;
                if (bx >= 0 && bx < WIDTH && by >= 0 && by < HEIGHT) {
                    // Use piece_idx + 1 to distinguish different piece types
                    output_board[by * WIDTH + bx] = piece_idx + 1;
                }
            }
        }
    }
}

int clear_lines_on_board(int* board_buffer, int width, int height) {
    if (!board_buffer || width <= 0 || height <= 0) return 0;
    
    // Convert the buffer into a 2D array format
    int board[HEIGHT][WIDTH];
    for (int y = 0; y < height && y < HEIGHT; y++) {
        for (int x = 0; x < width && x < WIDTH; x++) {
            board[y][x] = board_buffer[y * width + x];
        }
    }
    
    // Clear lines
    int lines = clear_lines(board);
    
    // Copy the result back to the buffer
    for (int y = 0; y < height && y < HEIGHT; y++) {
        for (int x = 0; x < width && x < WIDTH; x++) {
            board_buffer[y * width + x] = board[y][x];
        }
    }
    
    return lines;
}

// Update the make_best_move function to track the last move
int make_best_move(TetrisRLState* state) {
    if (!state) return 0;
    
    GameState* game_state = (GameState*)state->tetris_state;
    Piece current_piece = pieces[game_state->current_piece_index];
    
    // Save current piece for tracking
    game_state->last_piece_index = game_state->current_piece_index;
    
    // Save current board state
    int saved_board[HEIGHT][WIDTH];
    copy_board(saved_board, board);
    
    // Copy game board to global board for move generation
    copy_board(board, game_state->board);
    
    // Generate all possible moves
    Action actions[100];
    int num_actions = generate_actions(current_piece, actions, 100);
    game_state->num_available_moves = num_actions;
    memcpy(game_state->available_moves, actions, num_actions * sizeof(Action));
    
    if (num_actions == 0) {
        // No valid moves, game over
        copy_board(board, saved_board);
        game_state->game_over = 1;
        return 0;
    }
    
    // Find the best move
    int best_idx = 0;
    double best_value = -1e9;
    
    for (int i = 0; i < num_actions; i++) {
        if (actions[i].value > best_value) {
            best_value = actions[i].value;
            best_idx = i;
        }
    }
    
    // Store move details for animation
    game_state->last_rotation = actions[best_idx].rotation;
    game_state->last_x = actions[best_idx].x;
    game_state->last_y = actions[best_idx].y;
    game_state->last_lines_cleared = actions[best_idx].lines_cleared;
    
    // Calculate TD error
    double features_before[INPUT_SIZE];
    compute_features(game_state->board, features_before);
    double hidden1[HIDDEN_SIZE1], hidden2[HIDDEN_SIZE2];
    double q_before = nn_predict(features_before, hidden1, hidden2);
    
    // Apply the best move
    copy_board(game_state->board, actions[best_idx].resulting_board);
    game_state->score += actions[best_idx].lines_cleared;
    
    // Calculate new state TD error
    double features_after[INPUT_SIZE];
    compute_features(game_state->board, features_after);
    double q_after = nn_predict(features_after, hidden1, hidden2);
    game_state->last_td_error = (float)(actions[best_idx].lines_cleared + _gamma * q_after - q_before);
    
    // Generate a new piece
    game_state->current_piece_index = rand() % NUM_PIECES;
    
    // Restore global board
    copy_board(board, saved_board);
    
    return 1;  // Success
}