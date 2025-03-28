from animation import GameStateType, IntPtrType, FloatPtrType

# --- Basic Game Management ---
def init_game_and_nn() -> GameStateType: ...
def free_game(game_state: GameStateType) -> None: ...

# --- Board Information ---
def get_board_state(game_state: GameStateType, board_buffer: IntPtrType, buffer_size: int) -> None: ...
def get_board_dimensions(width_ptr: IntPtrType, height_ptr: IntPtrType) -> None: ...
def get_score(game_state: GameStateType) -> int: ...

# --- Piece Information ---
def get_next_piece(game_state: GameStateType) -> int: ...
def get_current_piece_position(game_state: GameStateType, piece_idx: IntPtrType, rotation: IntPtrType, x: IntPtrType, y: IntPtrType) -> None: ...
def get_piece_shape(piece_idx: int, rotation: int, buffer: IntPtrType, buffer_size: int) -> None: ...
def get_piece_count() -> int: ...
def get_piece_rotations(piece_idx: int) -> int: ...

# --- Gameplay Actions ---
def get_possible_move_count(game_state: GameStateType) -> int: ...
def make_best_move(game_state: GameStateType) -> int: ...
def make_move(game_state: GameStateType, move_idx: int) -> int: ...
def is_game_over(game_state: GameStateType) -> int: ...

# --- Move Information ---
def get_move_info(game_state: GameStateType, move_idx: int, rotation: IntPtrType, x: IntPtrType, 
                  y: IntPtrType, lines_cleared: IntPtrType, value: FloatPtrType) -> None: ...
def get_last_move_info(game_state: GameStateType, piece_idx: IntPtrType, rotation: IntPtrType, 
                        x: IntPtrType, y: IntPtrType, lines_cleared: IntPtrType) -> None: ...

# --- Neural Network ---
def get_nn_layer_sizes(game_state: GameStateType, layer_sizes: IntPtrType, max_layers: int) -> int: ...
def get_nn_weights(game_state: GameStateType, layer: int, neuron: int, weights: FloatPtrType, weights_count: int) -> None: ...
def get_nn_activations(game_state: GameStateType, layer: int, activations: FloatPtrType, activations_count: int) -> None: ...
def get_feature_values(game_state: GameStateType, features: FloatPtrType, feature_count: int) -> None: ...
def get_last_td_error(game_state: GameStateType) -> float: ...
def get_training_parameters(game_state: GameStateType, alpha: FloatPtrType, gamma: FloatPtrType, 
                            epsilon: FloatPtrType, lambda_ptr: FloatPtrType) -> None: ...

# --- Animation Helpers ---
def generate_falling_animation(game_state: GameStateType, piece_idx: int, rotation: int, x: int, 
                              final_y: int, board_frames: IntPtrType, max_frames: int) -> int: ...
def place_piece_for_preview(game_state: GameStateType, piece_idx: int, rotation: int, x: int, 
                           y: int, output_board: IntPtrType) -> None: ...
def clear_lines_on_board(board: IntPtrType, width: int, height: int) -> int: ...