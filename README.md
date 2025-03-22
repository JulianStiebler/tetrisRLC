# Tetris Reinforcement Learning Agent in Pure C

A reinforcement learning Tetris agent in Pure C with external API.
Also featuring a [Manim](https://github.com/ManimCommunity/manim/) visualization of the neural network and gameplay.

## Table of Contents
- [Overview](#overview)
- [Neural Network Architecture](#neural-network-architecture)
- [Reinforcement Learning Details](#reinforcement-learning-details)
- [Training Process](#training-process-and-hyperparameters)
- [Building and Running](#building-and-running)
- [Visualization](#visualization)
- [Future Improvements](#notes-and-further-improvements)


## Overview
This repo implements a reinforcement learning agent that learns to play Tetris using a multi-layer neural network, all written in pure C without external ML libraries. The agent uses RL techniques such as experience replay, a target network, double Q-learning, and checkpointing to stabilise training and improve performance over time.

## Neural Network Architecture
The neural network estimates Q-values for different board states and consists of:

- **Input Layer:**
  - Aggregate height
  - Number of holes
  - Bumpiness

- **Hidden Layers:**
  - First Hidden Layer: 20 neurons, sigmoid activation
  - Second Hidden Layer: 10 neurons, sigmoid activation

- **Output Layer:**
  - Single neuron producing Q-value estimate

### Forward Pass
1. **Layer 1:** Compute activations using sigmoid:  

   ![\Large \color{White} h^{(1)}_i = \sigma(\sum W1_{ij} x_j + b1_i)](https://latex.codecogs.com/svg.latex?%5CLarge%20%5Ccolor%7BWhite%7D%20h%5E%7B(1)%7D_i%20%3D%20%5Csigma(%5Csum%20W1_%7Bij%7D%20x_j%20%2B%20b1_i))

2. **Layer 2:** Compute activations similarly:  

   ![\Large \color{White} h^{(2)}_k = \sigma(\sum W2_{ki} h^{(1)}_i + b2_k)](https://latex.codecogs.com/svg.latex?%5CLarge%20%5Ccolor%7BWhite%7D%20h%5E%7B(2)%7D_k%20%3D%20%5Csigma(%5Csum%20W2_%7Bki%7D%20h%5E%7B(1)%7D_i%20%2B%20b2_k))

3. **Output Layer:** Linear combination:  

   ![\Large \color{White} Q = \sum W3_{k} h^{(2)}_k + b3](https://latex.codecogs.com/svg.latex?%5CLarge%20%5Ccolor%7BWhite%7D%20Q%20%3D%20%5Csum%20W3_%7Bk%7D%20h%5E%7B(2)%7D_k%20%2B%20b3)

### Backpropagation
- Compute **TD Error**:

  ![\Large \color{White} \delta = \text{reward} + \gamma Q_{\text{next}} - Q_{\text{current}}](https://latex.codecogs.com/svg.latex?%5CLarge%20%5Ccolor%7BWhite%7D%20%5Cdelta%20%3D%20%5Ctext%7Breward%7D%20%2B%20%5Cgamma%20Q_%7B%5Ctext%7Bnext%7D%7D%20-%20Q_%7B%5Ctext%7Bcurrent%7D%7D)

- Update weights with L2 regularization:  

  ![\Large \color{White} W \leftarrow W + \alpha (\text{gradient} - \lambda W)](https://latex.codecogs.com/svg.latex?%5CLarge%20%5Ccolor%7BWhite%7D%20W%20%5Cleftarrow%20W%20%2B%20%5Calpha%20(%5Ctext%7Bgradient%7D%20-%20%5Clambda%20W))

### Visualization
Watch the neural network in action:

<video width="600" controls>
  <source src="visualize/TetrisRLAnimation.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Reinforcement Learning Details
### Action Selection
- Simulate all valid placements for each piece.
- Use an **epsilon-greedy** policy to balance exploration and exploitation.

### Experience Replay
- Stores past transitions (state, reward, next state, done flag).
- Samples mini-batches to break sequential correlation.

### Checkpointing
- Saves network parameters and hyperparameters every episode.
- Reloads the best checkpoint if performance declines beyond 80%.

## Training Process and Hyperparameters
- **Alpha (Learning Rate):** Controls update magnitude, decays slowly.
- **Gamma (Discount Factor):** Prioritises future rewards.
- **Epsilon (Exploration Rate):** Decays to favor exploitation over time.
- **Lambda (L2 Regularization):** Prevents overfitting.
- **Target Network Update:** Every 10 episodes.

## Building and Running

### Build with Make
```sh
make
```

### Build with CMake
```sh
# Create build directory
mkdir -p out/build/x64-Debug
cd out/build/x64-Debug

# Generate build files
cmake -DCMAKE_BUILD_TYPE=Debug ../..

# Build the project
cmake --build .

# Return to project root
cd ../../..
```

### Building the API DLL
The visualization requires a compiled DLL/shared library to interface with the game engine:

```sh
# Navigate to build directory
cd out/build/x64-Debug

# Build the API shared library specifically
cmake --build . --target api

# The DLL will be available at:
# Windows: out/build/x64-Debug/api.dll
# Linux: out/build/x64-Debug/libapi.so
```

### Running on Linux
```sh
./tetris
```

### Running on Windows
```sh
.\tetris.exe
```

### Running the Visualization
Make sure to build the API DLL first (see above), then configure the animation quality and such a like in visualize/config.json.

```sh
cd visualize
python -m pip install -r requirements.txt
```

Then either double click run_manim.bat or
```sh
python -m manim animation.py TetrisRLAnimation -pqm
```

The generated animation will be saved to `out/media/videos/animation/[quality]/TetrisRLAnimation.mp4`

## Notes and Further Improvements
- **Hyperparameter Tuning:** Experiment with different decay rates.
- **Feature Engineering:** Consider additional board metrics.
- **Monitoring:** Log TD errors and Q-value distributions.

## License
This project is for educational purposes. You are free to modify and extend it.