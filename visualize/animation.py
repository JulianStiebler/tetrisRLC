from manim import (
    Scene,
    Text,
    Rectangle,
    Line,
    Square,
    Circle,
    VGroup,
    Transform,
    ManimColor,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    PI,
    BLACK,
    WHITE,
    GRAY,
    RED,
    BLUE,
    FadeIn,
    FadeOut,
    Succession,
    interpolate_color,
    config,
)
from typing import NewType
from types import ModuleType
import ctypes
import numpy as np
import os
import traceback
import sys
import json

GameStateType = NewType("GameStateType", ctypes.c_void_p)
IntPtrType = NewType("IntPtrType", ctypes.POINTER(ctypes.c_int))
FloatPtrType = NewType("FloatPtrType", ctypes.POINTER(ctypes.c_float))

try:
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.json"
    )
    with open(config_path, "r") as f:
        cfg = json.load(f)
    print(f"Loaded configuration from {config_path}")
except Exception as e:
    print(f"Error loading config.json: {e}. Using default configuration.")
    # Default configuration
    cfg = {
        "manim": {
            "output_dir": "../out/media",
            "quality": "low_quality",
            "disable_caching": False,
            "frame_rate": 15,
            "pixel_height": 1920,
            "pixel_width": 1080,
        },
        "animation": {
            "max_moves": 1,
            "animation_speed": 0.1,
            "pause_between_moves": 0.1,
        },
        "board": {
            "cell_size": 0.15,
            "scale": 1.3,
            "grid_stroke_width": 1,
            "border_stroke_width": 2,
            "background_color": "#1E1E1E",
        },
        "neural_network": {
            "layer_spacing": 2.2,
            "neuron_spacing": 0.6,
            "neuron_radius": 0.18,
            "connection_stroke_width": 4.5,
            "connection_opacity": 0.4,
            "pulse_animation": {
                "min_stroke_width": 1.0,
                "max_stroke_width": 5.0,
                "weight_scaling": 4.0,
                "pulse_extra_width": 2,
                "min_opacity": 0.3,
                "duration": 0.2,
            },
            "scale": 1.2,
        },
        "features": {
            "bar_width": 0.25,
            "bar_height": 2.0,
            "bar_spacing": 0.35,
            "text_size": 9,
            "scale": 0.9,
            "max_values": {
                "Height": 20.0,
                "Holes": 30.0,
                "Bumps": 20.0,
                "Lines": 4.0,
                "Wells": 15.0,
                "Blocks": 20.0,
                "Var": 10.0,
                "Row T": 20.0,
                "Col T": 20.0,
                "Empty": 50.0,
            },
        },
        "colors": {
            "pieces": {
                "0": "#000000",
                "1": "#00CED1",
                "2": "#FFD700",
                "3": "#9932CC",
                "4": "#32CD32",
                "5": "#FF0000",
                "6": "#0000FF",
                "7": "#FF8C00",
            },
            "positive_weight": "#32CD32",
            "negative_weight": "#FF0000",
            "pulse": "#FFFF00",
            "neuron_inactive": "#0000FF",
            "neuron_active": "#00FF00",
        },
        "layout": {"title_buff": 0.3, "section_buff": 0.5, "feature_board_buff": 0.2},
        "text": {"title_size": 24, "label_size": 16, "value_size": 9},
    }

# Configure Manim's output directory
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), cfg["manim"]["output_dir"]
)
config.media_dir = output_dir
config.quality = cfg["manim"]["quality"]
config.disable_caching = cfg["manim"]["disable_caching"]
config.frame_rate = cfg["manim"]["frame_rate"]
config.pixel_height = cfg["manim"]["pixel_height"]
config.pixel_width = cfg["manim"]["pixel_width"]

# Global constants
MAX_MOVES = cfg["animation"]["max_moves"]
ANIMATION_SPEED = cfg["animation"]["animation_speed"]


class TetrisRLAnimation(Scene):
    def construct(self):
        self.error_encountered = False  # Add flag to track errors
        self.game_state = None  # Initialize game_state to None

        root_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(root_dir, "..", "out", "build", "x64-Debug")
        shared_dll = "api.dll" if os.name == "nt" else "api.so"
        lib_path = os.path.join(target_dir, shared_dll)

        if not os.path.exists(lib_path):
            error_text = Text(f"Library not found at: {lib_path}", color=RED)
            self.add(error_text)
            self.wait(2)
            self.error_encountered = True  # Set error flag
            return

        try:
            if os.name == "nt":
                self.lib = ctypes.WinDLL(lib_path)
            else:
                self.lib = ctypes.CDLL(lib_path)
            print(f"Successfully loaded library from {lib_path}")

            # Create a module-like object for type hinting
            api_module = ModuleType("api")
            sys.modules["api"] = api_module

            # Set up the function types
            self.setup_c_functions()
        except Exception as e:
            error_msg = f"Error loading C library: {e}"
            print(error_msg)
            error_text = Text(error_msg, color=RED)
            self.add(error_text)
            self.wait(2)
            self.error_encountered = True  # Set error flag
            return

        # Only proceed if no errors were encountered
        if self.error_encountered:
            return

        width = ctypes.c_int(0)
        height = ctypes.c_int(0)
        self.lib.get_board_dimensions(ctypes.byref(width), ctypes.byref(height))
        self.BOARD_WIDTH, self.BOARD_HEIGHT = width.value, height.value
        print(f"Board dimensions: {self.BOARD_WIDTH} x {self.BOARD_HEIGHT}")

        self.game_state = self.lib.init_game_and_nn()
        if not self.game_state:
            error_text = Text("Failed to initialize game state", color=RED)
            self.add(error_text)
            self.wait(2)
            self.error_encountered = True
            return
        print("Game state initialized successfully")

        # Get neural network layer sizes directly from C API
        layer_sizes = (ctypes.c_int * 10)()
        num_layers = self.lib.get_nn_layer_sizes(self.game_state, layer_sizes, 10)
        if num_layers == 0:
            print("Warning: No neural network layers returned")
            self.nn_layers = [3, 5, 5, 1]
        else:
            self.nn_layers = [layer_sizes[i] for i in range(num_layers)]
        print(f"Neural network layers: {self.nn_layers}")

        try:
            self.create_scene_components()
            self.animate_gameplay()
        except Exception as e:
            error_msg = f"Error during animation: {e}"
            print(error_msg)
            error_text = Text(error_msg, color=RED, font_size=24)
            self.add(error_text)
            self.wait(2)

        # Clean up resources only if game_state was initialized and no earlier errors
        if self.game_state and not self.error_encountered:
            self.lib.free_game(self.game_state)

    def setup_c_functions(self):
        # Define the C function signatures
        self.lib.init_game_and_nn.restype = ctypes.c_void_p

        self.lib.get_board_dimensions.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]

        self.lib.get_board_state.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]

        self.lib.get_next_piece.argtypes = [ctypes.c_void_p]
        self.lib.get_next_piece.restype = ctypes.c_int

        self.lib.get_nn_layer_sizes.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self.lib.get_nn_layer_sizes.restype = ctypes.c_int

        self.lib.get_nn_weights.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

        self.lib.get_nn_activations.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

        self.lib.get_feature_values.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

        self.lib.make_best_move.argtypes = [ctypes.c_void_p]
        self.lib.make_best_move.restype = ctypes.c_int

        self.lib.get_score.argtypes = [ctypes.c_void_p]
        self.lib.get_score.restype = ctypes.c_int

        self.lib.free_game.argtypes = [ctypes.c_void_p]

        # Set up animation-specific functions
        self.lib.get_last_move_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]

        self.lib.generate_falling_animation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self.lib.generate_falling_animation.restype = ctypes.c_int

        self.lib.place_piece_for_preview.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]

        self.lib.get_last_td_error.argtypes = [ctypes.c_void_p]
        self.lib.get_last_td_error.restype = ctypes.c_float

        self.lib.get_training_parameters.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]

        # Create a wrapper module that will forward attribute access to the C library
        class LibraryWrapper(ModuleType):
            def __init__(self, lib, name):
                super().__init__(name)
                self._lib = lib

            def __getattr__(self, name):
                if hasattr(self._lib, name):
                    return getattr(self._lib, name)
                raise AttributeError(f"{name} not found in library")

        # Replace the module in sys.modules with our wrapper
        wrapper = LibraryWrapper(self.lib, "api")
        sys.modules["api"] = wrapper

    def create_scene_components(self):
        title = Text(
            "Tetris RL Visualization", font_size=cfg["text"]["title_size"]
        ).to_edge(UP, buff=cfg["layout"]["title_buff"])
        self.add(title)

        # Create a container for the game section (left side)
        game_section = VGroup()

        # Create board visual
        self.board_visual = self.create_board_visual()
        game_section.add(self.board_visual)

        # Create feature visual
        self.feature_visual = self.create_feature_visual()
        self.feature_visual.scale(cfg["features"]["scale"])

        # Position features below board
        self.feature_visual.next_to(
            self.board_visual, DOWN, buff=cfg["layout"]["feature_board_buff"]
        )
        game_section.add(self.feature_visual)

        # Position game section on the left AND SHIFTED UP
        game_section.to_edge(LEFT, buff=cfg["layout"]["section_buff"])
        game_section.shift(UP * 0.8)  # Move the entire game section up by 0.8 units

        # NN visual (larger)
        self.nn_visual = self.create_nn_visual()
        self.nn_visual.scale(cfg["neural_network"]["scale"])
        self.nn_visual.to_edge(RIGHT, buff=cfg["layout"]["section_buff"])

        self.add(game_section, self.nn_visual)

        # Create info group with Score and TD Error stacked vertically
        info_group = VGroup()

        # Score text
        self.score_text = Text("Score: 0", font_size=cfg["text"]["label_size"])
        info_group.add(self.score_text)

        # TD Error text (initially empty)
        self.td_error_text = Text(
            "TD Error: 0.0000", font_size=cfg["text"]["label_size"]
        )
        self.td_error_text.next_to(
            self.score_text, DOWN, buff=0.1
        )  # Stack below score with small buffer
        info_group.add(self.td_error_text)

        # Position the info group below the title
        info_group.next_to(title, DOWN, buff=cfg["layout"]["title_buff"])

        # Add the info group to the scene
        self.add(info_group)

    def create_board_visual(self):
        # Create a visual representation of the Tetris board
        board_cfg = cfg["board"]
        cell_size = board_cfg["cell_size"]
        board = VGroup()

        # Background
        bg = Rectangle(
            width=self.BOARD_WIDTH * cell_size,
            height=self.BOARD_HEIGHT * cell_size,
            fill_color=board_cfg["background_color"],
            fill_opacity=1,
            stroke_color=WHITE,
            stroke_width=board_cfg["border_stroke_width"],
        )
        board.add(bg)

        # Grid
        for i in range(self.BOARD_WIDTH + 1):
            x = i * cell_size - self.BOARD_WIDTH * cell_size / 2
            line = Line(
                start=np.array([x, -self.BOARD_HEIGHT * cell_size / 2, 0]),
                end=np.array([x, self.BOARD_HEIGHT * cell_size / 2, 0]),
                stroke_width=board_cfg["grid_stroke_width"],
                stroke_color=GRAY,
            )
            board.add(line)

        for j in range(self.BOARD_HEIGHT + 1):
            y = j * cell_size - self.BOARD_HEIGHT * cell_size / 2
            line = Line(
                start=np.array([-self.BOARD_WIDTH * cell_size / 2, y, 0]),
                end=np.array([self.BOARD_WIDTH * cell_size / 2, y, 0]),
                stroke_width=board_cfg["grid_stroke_width"],
                stroke_color=GRAY,
            )
            board.add(line)

        # Create cells
        self.cells = []
        for y in range(self.BOARD_HEIGHT):
            row = []
            for x in range(self.BOARD_WIDTH):
                x_pos = (x - self.BOARD_WIDTH / 2 + 0.5) * cell_size
                y_pos = (
                    self.BOARD_HEIGHT / 2 - y - 0.5
                ) * cell_size  # Invert Y coordinate for proper Tetris display

                cell = Square(side_length=cell_size, fill_opacity=0, stroke_width=0)
                cell.move_to(np.array([x_pos, y_pos, 0]))
                board.add(cell)
                row.append(cell)
            self.cells.append(row)

        # Scale and label
        board.scale(board_cfg["scale"])
        label = Text("Tetris Board", font_size=cfg["text"]["label_size"])
        label.next_to(board, UP)
        board.add(label)

        return board

    def create_nn_visual(self):
        nn = VGroup()

        # Create neurons for each layer - with config from JSON
        nn_config = cfg["neural_network"]
        layer_spacing = nn_config["layer_spacing"]
        neuron_spacing = nn_config["neuron_spacing"]
        neuron_radius = nn_config["neuron_radius"]

        self.nn_neurons = []

        for layer_index, layer_size in enumerate(self.nn_layers):
            layer_neurons = []
            layer_group = VGroup()

            # Limit the number of neurons to display for very large layers
            display_count = min(layer_size, 10)

            for n in range(display_count):
                # Calculate position for this neuron
                y_pos = (n - (display_count - 1) / 2) * neuron_spacing

                # Create neuron
                neuron = Circle(
                    radius=neuron_radius,
                    fill_color=BLUE,
                    fill_opacity=0.5,
                    stroke_color=WHITE,
                )
                neuron.move_to(np.array([layer_index * layer_spacing, y_pos, 0]))
                layer_neurons.append(neuron)
                layer_group.add(neuron)

            # Add label for layer
            layer_label = Text(f"Layer {layer_index}", font_size=16)
            layer_label.next_to(layer_group, DOWN, buff=0.5)
            layer_group.add(layer_label)

            self.nn_neurons.append(layer_neurons)
            nn.add(layer_group)

        # Connect neurons between adjacent layers with THICKER stroke
        self.nn_connections = []
        for layer_index in range(len(self.nn_layers) - 1):
            layer_connections = []
            for i, n1 in enumerate(self.nn_neurons[layer_index]):
                for j, n2 in enumerate(self.nn_neurons[layer_index + 1]):
                    conn = Line(
                        n1.get_center(),
                        n2.get_center(),
                        stroke_width=nn_config["connection_stroke_width"],
                        stroke_opacity=nn_config["connection_opacity"],
                    )
                    nn.add(conn)
                    layer_connections.append(conn)
            self.nn_connections.append(layer_connections)

        # Scale and add title
        nn.scale(0.9)  # Base scale
        title = Text("Neural Network", font_size=cfg["text"]["label_size"])
        title.next_to(nn, UP)
        nn.add(title)

        return nn

    def create_feature_visual(self):
        # Get actual feature count from API
        feature_buffer = (ctypes.c_float * 20)()
        self.lib.get_feature_values(self.game_state, feature_buffer, 20)

        # Use actual feature names if known, otherwise use generic names
        feature_names = list(cfg["features"]["max_values"].keys())

        # Use only the actual number of features we have
        feature_count = min(len(feature_buffer), len(feature_names))
        feature_names = feature_names[:feature_count]

        feature_group = VGroup()
        self.feature_bars = []
        self.feature_values = []

        # Create bars for each feature using config values
        features_cfg = cfg["features"]
        bar_width = features_cfg["bar_width"]
        bar_height = features_cfg["bar_height"]
        bar_spacing = features_cfg["bar_spacing"]
        text_size = features_cfg["text_size"]

        # Create a container for the bars
        bars_group = VGroup()

        for i, name in enumerate(feature_names):
            x_pos = (i - len(feature_names) / 2 + 0.5) * bar_spacing

            # Background bar
            bg_bar = Rectangle(
                width=bar_width,
                height=bar_height,
                fill_color=GRAY,
                fill_opacity=0.3,
                stroke_color=WHITE,
            )
            bg_bar.move_to(np.array([x_pos, 0, 0]))

            # Value bar (initially empty)
            value_bar = Rectangle(
                width=bar_width,
                height=0,
                fill_color=BLUE,
                fill_opacity=0.8,
                stroke_width=0,
            )
            value_bar.align_to(bg_bar, DOWN)
            value_bar.align_to(bg_bar, LEFT)

            # Value text
            value_text = Text("0.0", font_size=text_size)
            value_text.next_to(bg_bar, UP, buff=0.05)

            # Feature name - rotated and placed inside the bar, top-aligned
            feature_text = Text(name, font_size=text_size)
            feature_text.rotate(PI / 2)  # Rotate text to be vertical

            # Position text inside the bar, aligned to the top
            # First move to center of bar
            feature_text.move_to(bg_bar.get_center())

            # Then align to top of bar with small buffer
            feature_text.shift(
                UP * (bg_bar.height / 2 - feature_text.height / 2 - 0.05)
            )

            # Group everything
            group = VGroup(bg_bar, value_bar, feature_text, value_text)
            bars_group.add(group)

            self.feature_bars.append(value_bar)
            self.feature_values.append(value_text)

        # Create title and position it to the left of the bars
        title = Text("Input Features", font_size=cfg["text"]["label_size"] * 0.85)
        # Rotate title to be vertical along the left side
        title.rotate(PI / 2)
        # Position to the left of bars_group
        title.next_to(bars_group, LEFT, buff=0.3)

        # Add both title and bars to the main feature group
        feature_group.add(title)
        feature_group.add(bars_group)

        return feature_group

    def update_board(self):
        """Update board display using the C API get_board_state function"""
        try:
            # Get current board state from C code
            board_buffer = (ctypes.c_int * (self.BOARD_WIDTH * self.BOARD_HEIGHT))()
            self.lib.get_board_state(
                self.game_state, board_buffer, self.BOARD_WIDTH * self.BOARD_HEIGHT
            )

            # Update cell colors
            animations = []

            # Use piece colors from config
            piece_colors = {}
            for key, value in cfg["colors"]["pieces"].items():
                piece_colors[int(key)] = value

            # Only animate cells that have changed
            for y in range(self.BOARD_HEIGHT):
                for x in range(self.BOARD_WIDTH):
                    idx = y * self.BOARD_WIDTH + x
                    cell_value = board_buffer[idx]

                    if y < len(self.cells) and x < len(self.cells[y]):
                        cell = self.cells[y][x]
                        color = piece_colors.get(cell_value, BLACK)
                        opacity = 0 if cell_value == 0 else 1

                        # Skip empty cells to reduce animation count
                        if cell_value != 0:
                            animations.append(
                                cell.animate.set_fill(color, opacity=opacity)
                            )

            if animations:
                self.play(*animations, run_time=cfg["animation"]["animation_speed"])

        except Exception as e:
            print(f"Error updating board: {e}")
            import traceback

            traceback.print_exc()

    def update_nn_visualization(self):
        """Update neural network visualization with activation flow animation"""
        try:
            neuron_animations = []
            connection_animations = []
            activation_animations = []

            nn_cfg = cfg["neural_network"]
            pulse_cfg = nn_cfg["pulse_animation"]
            colors = cfg["colors"]

            # Convert string hex colors to Manim color objects
            inactive_color = ManimColor(colors["neuron_inactive"])
            active_color = ManimColor(colors["neuron_active"])
            positive_color = ManimColor(colors["positive_weight"])
            negative_color = ManimColor(colors["negative_weight"])
            pulse_color = ManimColor(colors["pulse"])

            # Process all layers
            for layer_index in range(min(len(self.nn_layers), 4)):
                layer_size = self.nn_layers[layer_index]
                display_size = min(layer_size, len(self.nn_neurons[layer_index]))

                # Get actual activations from C API
                activations = (ctypes.c_float * layer_size)()
                self.lib.get_nn_activations(
                    self.game_state, layer_index, activations, layer_size
                )

                # Update neuron colors based on activations
                for i, neuron in enumerate(self.nn_neurons[layer_index]):
                    if i < display_size:
                        activation = activations[i] if i < len(activations) else 0.5
                        # Normalize activation to [0, 1] range for color interpolation
                        activation = min(max(float(activation), -1), 1) * 0.5 + 0.5
                        color = interpolate_color(
                            inactive_color, active_color, activation
                        )
                        neuron_animations.append(
                            neuron.animate.set_fill(
                                color, opacity=0.5 + 0.5 * activation
                            )
                        )

                # First play neuron animations
                if neuron_animations:
                    self.play(
                        *neuron_animations, run_time=cfg["animation"]["animation_speed"]
                    )

            # Get actual weights from C API and update connections with animation
            for layer_index in range(len(self.nn_connections)):
                from_layer = layer_index
                to_layer = layer_index + 1
                from_size = self.nn_layers[from_layer]
                to_size = self.nn_layers[to_layer]

                # Get weights from C API
                buffer_size = from_size * to_size
                weights = (ctypes.c_float * buffer_size)()
                self.lib.get_nn_weights(
                    self.game_state, from_layer, to_layer, weights, buffer_size
                )

                # Update connection appearances based on weights
                conn_idx = 0
                from_display = min(from_size, len(self.nn_neurons[from_layer]))
                to_display = min(to_size, len(self.nn_neurons[to_layer]))

                # Get activations for from_layer
                from_activations = (ctypes.c_float * from_size)()
                self.lib.get_nn_activations(
                    self.game_state, from_layer, from_activations, from_size
                )

                for i in range(from_display):
                    for j in range(to_display):
                        if conn_idx < len(self.nn_connections[layer_index]):
                            conn = self.nn_connections[layer_index][conn_idx]

                            # Calculate 1D index into weight matrix
                            weight_idx = i * to_size + j
                            if weight_idx < buffer_size:
                                weight = weights[weight_idx]
                                activation = from_activations[i] if i < from_size else 0
                                abs_weight = abs(weight)

                                # Only animate connections with significant weight and activation
                                if abs_weight > 0.1 and abs(activation) > 0.1:
                                    # Set color based on whether it's excitatory or inhibitory
                                    color = (
                                        positive_color if weight > 0 else negative_color
                                    )
                                    stroke_width = max(
                                        pulse_cfg["min_stroke_width"],
                                        min(
                                            pulse_cfg["max_stroke_width"],
                                            abs_weight * pulse_cfg["weight_scaling"],
                                        ),
                                    )

                                    # Create a "pulse" animation along the connection
                                    # Clone the connection line for animation
                                    pulse = Line(
                                        conn.get_start(),
                                        conn.get_end(),
                                        stroke_width=stroke_width
                                        + pulse_cfg["pulse_extra_width"],
                                        stroke_color=pulse_color,
                                        stroke_opacity=0,
                                    )
                                    self.add(pulse)

                                    # Signal strength based on activation and weight
                                    signal_strength = (
                                        abs(activation * weight) * 0.7 + 0.3
                                    )

                                    # First fade in at start
                                    fade_in = FadeIn(pulse, run_time=0.1)

                                    # Then animate opacity along the line
                                    pulse_anim = pulse.animate.set_stroke(
                                        color=color,
                                        width=stroke_width
                                        + pulse_cfg["pulse_extra_width"],
                                        opacity=signal_strength,
                                    )

                                    # Finally fade out
                                    fade_out = FadeOut(pulse, run_time=0.1)

                                    # Group these animations in sequence
                                    pulse_sequence = Succession(
                                        fade_in, pulse_anim, fade_out
                                    )
                                    activation_animations.append(pulse_sequence)

                                # Also update the base connection appearance
                                opacity = min(
                                    1.0, abs_weight + pulse_cfg["min_opacity"]
                                )
                                connection_animations.append(
                                    conn.animate.set_stroke(
                                        color=colors["positive_weight"]
                                        if weight > 0
                                        else colors["negative_weight"],
                                        width=max(
                                            pulse_cfg["min_stroke_width"],
                                            min(
                                                pulse_cfg["max_stroke_width"],
                                                abs_weight
                                                * pulse_cfg["weight_scaling"],
                                            ),
                                        ),
                                        opacity=opacity,
                                    )
                                )

                            conn_idx += 1

            # Play connection animations first
            if connection_animations:
                self.play(
                    *connection_animations, run_time=cfg["animation"]["animation_speed"]
                )

            # Then play activation flow animations
            if activation_animations:
                self.play(*activation_animations, run_time=pulse_cfg["duration"])

        except Exception as e:
            print(f"Error updating neural network: {e}")
            import traceback

            traceback.print_exc()

    def update_feature_visualization(self):
        """Update feature visualization with values from the C API"""
        try:
            # Get actual feature values from C API
            feature_count = len(self.feature_bars)
            features = (ctypes.c_float * feature_count)()
            self.lib.get_feature_values(self.game_state, features, feature_count)

            # Use max values from config
            max_feature_values = []
            for max_val in cfg["features"]["max_values"].values():
                max_feature_values.append(float(max_val))

            # Update feature bars (vertical bars)
            animations = []
            for i, (bar, text) in enumerate(
                zip(self.feature_bars, self.feature_values)
            ):
                if i < feature_count:
                    value = features[i]

                    # Use the appropriate max value for this feature
                    max_value = (
                        max_feature_values[i] if i < len(max_feature_values) else 10.0
                    )

                    # Normalize value to 0-1 range with respect to its max value
                    normalized_value = min(abs(float(value)) / max_value, 1.0)

                    # Scale to the height of our bar background (vertical)
                    bar_height = cfg["features"]["bar_height"] * normalized_value

                    # Color based on positive/negative value
                    color = (
                        cfg["colors"]["positive_weight"]
                        if value >= 0
                        else cfg["colors"]["negative_weight"]
                    )

                    # Create a new bar with the correct height
                    new_bar = Rectangle(
                        width=bar.width,
                        height=bar_height,
                        fill_color=color,
                        fill_opacity=0.8,
                        stroke_width=0,
                    )

                    # Position it correctly - aligned to the bottom
                    new_bar.move_to(bar.get_center())
                    new_bar.align_to(bar, DOWN)

                    # Transform the bar
                    animations.append(Transform(bar, new_bar))

                    # Format the text value with the normalized percentage
                    value_str = f"{float(value):.2f}"
                    new_text = Text(value_str, font_size=cfg["features"]["text_size"])
                    new_text.move_to(text.get_center())
                    animations.append(Transform(text, new_text))

            if animations:
                self.play(*animations, run_time=0.3)

        except Exception as e:
            print(f"Error updating features: {e}")
            import traceback

            traceback.print_exc()

    def update_score(self):
        """Update score display using the C API"""
        try:
            score = self.lib.get_score(self.game_state)
            new_score_text = Text(
                f"Score: {score}", font_size=cfg["text"]["title_size"]
            )
            new_score_text.move_to(self.score_text.get_center())
            self.play(Transform(self.score_text, new_score_text))
        except Exception as e:
            print(f"Error updating score: {e}")

    def animate_gameplay(self):
        """Main gameplay animation loop"""
        try:
            # Initial static update without animations
            self.update_static_elements()

            # Update the existing TD error text (the one below score)
            td_error = self.lib.get_last_td_error(self.game_state)
            new_td_error_text = Text(
                f"TD Error: {td_error:.4f}", font_size=cfg["text"]["label_size"]
            )
            new_td_error_text.move_to(self.td_error_text.get_center())
            self.play(Transform(self.td_error_text, new_td_error_text))

            # Run through moves
            for i in range(cfg["animation"]["max_moves"]):
                print(f"Move {i + 1}")

                # Make the best move in our C code
                result = self.lib.make_best_move(self.game_state)
                if result == 0:
                    print("Game over or invalid move")
                    break

                # Animate the piece falling using C API
                self.animate_piece_falling()

                # Update features, neural network, and score
                self.update_feature_visualization()
                self.update_nn_visualization()
                self.update_score()

                # Update TD error (using the existing text element)
                td_error = self.lib.get_last_td_error(self.game_state)
                new_td_error_text = Text(
                    f"TD Error: {td_error:.4f}", font_size=cfg["text"]["label_size"]
                )
                new_td_error_text.move_to(self.td_error_text.get_center())
                self.play(Transform(self.td_error_text, new_td_error_text))

                # Small pause between moves
                self.wait(cfg["animation"]["pause_between_moves"])

        except Exception as e:
            print(f"Error in gameplay animation: {e}")
            traceback.print_exc()

    def animate_piece_falling(self):
        """Animate the piece falling using the C API generate_falling_animation function"""
        try:
            # Get last move info from the C API
            piece_idx = ctypes.c_int()
            rotation = ctypes.c_int()
            x = ctypes.c_int()
            y = ctypes.c_int()
            lines_cleared = ctypes.c_int()

            self.lib.get_last_move_info(
                self.game_state,
                ctypes.byref(piece_idx),
                ctypes.byref(rotation),
                ctypes.byref(x),
                ctypes.byref(y),
                ctypes.byref(lines_cleared),
            )

            # Generate animation frames using C API
            max_frames = 15
            board_frames = (
                ctypes.c_int * (self.BOARD_WIDTH * self.BOARD_HEIGHT * max_frames)
            )()

            # Call C function to generate animation frames
            frame_count = self.lib.generate_falling_animation(
                self.game_state,
                piece_idx.value,
                rotation.value,
                x.value,
                y.value,
                board_frames,
                max_frames,
            )

            if frame_count > 0:
                # Display each frame
                piece_colors = {}
                for key, value in cfg["colors"]["pieces"].items():
                    piece_colors[int(key)] = value

                for frame_idx in range(frame_count):
                    animations = []

                    # Update each cell based on this frame
                    for y in range(self.BOARD_HEIGHT):
                        for x in range(self.BOARD_WIDTH):
                            idx = (
                                frame_idx * self.BOARD_WIDTH * self.BOARD_HEIGHT
                                + y * self.BOARD_WIDTH
                                + x
                            )
                            cell_value = board_frames[idx]

                            if y < len(self.cells) and x < len(self.cells[y]):
                                cell = self.cells[y][x]
                                color = piece_colors.get(cell_value, BLACK)
                                opacity = 0 if cell_value == 0 else 1
                                animations.append(
                                    cell.animate.set_fill(color, opacity=opacity)
                                )

                    if animations:
                        self.play(
                            *animations, run_time=cfg["animation"]["animation_speed"]
                        )

            # Final update after piece has settled
            self.update_board()

        except Exception as e:
            print(f"Error animating piece falling: {e}")
            traceback.print_exc()
            # Fall back to regular update
            self.update_board()

    def update_static_elements(self):
        """Update all visual elements without animation"""
        try:
            # Get current board state
            board_buffer = (ctypes.c_int * (self.BOARD_WIDTH * self.BOARD_HEIGHT))()
            self.lib.get_board_state(
                self.game_state, board_buffer, self.BOARD_WIDTH * self.BOARD_HEIGHT
            )

            piece_colors = {}
            for key, value in cfg["colors"]["pieces"].items():
                piece_colors[int(key)] = value

            # Update cells directly without animation
            for y in range(self.BOARD_HEIGHT):
                for x in range(self.BOARD_WIDTH):
                    if y < len(self.cells) and x < len(self.cells[y]):
                        idx = y * self.BOARD_WIDTH + x
                        cell_value = board_buffer[idx]
                        color = piece_colors.get(cell_value, BLACK)
                        opacity = 0 if cell_value == 0 else 1
                        self.cells[y][x].set_fill(color, opacity=opacity)

            # Update score without animation
            score = self.lib.get_score(self.game_state)
            self.score_text.become(
                Text(f"Score: {score}", font_size=cfg["text"]["title_size"]).move_to(
                    self.score_text.get_center()
                )
            )

        except Exception as e:
            print(f"Error in static update: {e}")
            traceback.print_exc()
