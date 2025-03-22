#ifndef TETRIS_RL_INTERFACE_H
#define TETRIS_RL_INTERFACE_H

#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Game state structure to be exported
typedef struct {
    void* tetris_state;  // Pointer to internal tetris game state
    void* nn_state;      // Pointer to neural network state
} TetrisRLState;

/* --- BASIC GAME MANAGEMENT --- */

// Initialize game and neural network
TETRIS_API TetrisRLState* init_game_and_nn(void);

// Clean up resources
TETRIS_API void free_game(TetrisRLState* state);

/* --- BOARD INFORMATION --- */

// Get current board state (returns array of integer values)
TETRIS_API void get_board_state(TetrisRLState* state, int* buffer, int buffer_size);

// Get dimensions of the board
TETRIS_API void get_board_dimensions(int* width, int* height);

// Get score
TETRIS_API int get_score(TetrisRLState* state);

/* --- PIECE INFORMATION --- */

// Get the next piece type (returns int 0-6 for pieces I,O,T,S,Z,J,L)
TETRIS_API int get_next_piece(TetrisRLState* state);

// Get the current piece position (x, y, rotation)
TETRIS_API void get_current_piece_position(TetrisRLState* state, int* piece_idx, int* rotation, int* x, int* y);

// Get shape data for a specific piece and rotation (4x4 grid)
TETRIS_API void get_piece_shape(int piece_idx, int rotation, int* buffer, int buffer_size);

// Get total number of piece types
TETRIS_API int get_piece_count(void);

// Get number of rotations for a specific piece
TETRIS_API int get_piece_rotations(int piece_idx);

/* --- GAMEPLAY ACTIONS --- */

// Get number of possible moves for current piece
TETRIS_API int get_possible_move_count(TetrisRLState* state);

// Make the AI choose the best move and execute it
TETRIS_API int make_best_move(TetrisRLState* state);

// Make a specific move (move_idx selects from possible rotations/positions)
TETRIS_API int make_move(TetrisRLState* state, int move_idx);

// Check if game is over
TETRIS_API int is_game_over(TetrisRLState* state);

/* --- MOVE INFORMATION --- */

// Get information about a specific possible move
TETRIS_API void get_move_info(TetrisRLState* state, int move_idx, int* rotation, int* x, int* y, int* lines_cleared, float* value);

// Get information about the last move performed
TETRIS_API void get_last_move_info(TetrisRLState* state, int* piece_idx, int* rotation, int* x, int* y, int* lines_cleared);

/* --- NEURAL NETWORK --- */

// Get neural network layer sizes
TETRIS_API int get_nn_layer_sizes(TetrisRLState* state, int* sizes, int max_layers);

// Get weights between layers (from_layer and to_layer are 0-indexed)
TETRIS_API void get_nn_weights(TetrisRLState* state, int from_layer, int to_layer, float* buffer, int buffer_size);

// Get activation values at each layer during forward pass
TETRIS_API void get_nn_activations(TetrisRLState* state, int layer, float* buffer, int buffer_size);

// Get feature values for current board state
TETRIS_API void get_feature_values(TetrisRLState* state, float* buffer, int buffer_size);

// Get TD error from the last move
TETRIS_API float get_last_td_error(TetrisRLState* state);

// Get current training parameters
TETRIS_API void get_training_parameters(TetrisRLState* state, float* alpha, float* gamma, float* epsilon, float* lambda);

/* --- ANIMATION HELPERS --- */

// Generate animation frames for a piece falling from top to final position
TETRIS_API int generate_falling_animation(TetrisRLState* state, int piece_idx, int rotation, int x, int final_y, int* board_frames, int max_frames);

// Place a piece on the board at a specific position for visualization
TETRIS_API void place_piece_for_preview(TetrisRLState* state, int piece_idx, int rotation, int x, int y, int* output_board);

// Clear lines on a board and return the number cleared (animation helper)
TETRIS_API int clear_lines_on_board(int* board, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // TETRIS_RL_INTERFACE_H