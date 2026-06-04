# BrainBlock Tetromino DRL Project

## Overview

This project solves the **BrainBlock Packing Environment** using Deep Reinforcement Learning.

The task is to fill an **8 × 5 board** using a shuffled queue of tetromino pieces:

- 2 × I
- 2 × O
- 2 × L
- 2 × Z
- 2 × T

The full board contains 40 cells, and each tetromino covers 4 cells. A successful solution places all 10 pieces legally without overlap or going outside the board.

The project includes:

- Custom Gymnasium environment
- Deep Q-Network agent using PyTorch
- Replay buffer
- Dense and sparse reward experiments
- Legal action masking
- Multi-seed training
- Deterministic evaluation
- Learning curve generation

---

## Project Structure

```text
Tetromino445-master/

├── env/
│   ├── tetromino.py
│   └── test_env.py
│
├── models/
│   └── q_network.py
│
├── saved_models/
│   ├── dqn_brainblock_seed_42.pth
│   ├── dqn_brainblock_seed_100.pth
│   ├── dqn_brainblock_seed_2026.pth
│   ├── dqn_brainblock_seed_7.pth
│   └── dqn_brainblock_seed_99.pth
│
├── train.py
├── evaluate.py
└── dense_reward_performance.png
```

---

## Requirements

Use Python 3.10 or newer.

Install the required libraries:

```bash
pip install torch gymnasium numpy matplotlib
```

If you are using a virtual environment, activate it first before installing the dependencies.

Example:

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
pip install torch gymnasium numpy matplotlib
```

---

## How to Run the Project

First, open the project folder:

```bash
cd Tetromino445-master
```

All commands below should be run from inside this folder.

---

## 1. Test the Environment

To check whether the environment runs correctly:

```bash
cd env
python test_env.py
```

This runs a random placement simulation and prints the board after each action.

After testing, return to the main project folder:

```bash
cd ..
```

---

## 2. Train the DQN Agent

To train the agent:

```bash
python train.py
```

This script runs both:

- Dense reward training
- Sparse reward training

It trains using five random seeds:

```text
42, 100, 2026, 7, 99
```

During training, the script prints progress such as:

- Episode number
- Episode reward
- Covered board area
- Current epsilon value

---

## Training Outputs

After training, the script generates performance plots:

```text
dense_reward_performance.png
sparse_reward_performance.png
```

The training script also saves newly trained model weights into:

```text
a=5 b=005 c=20/
```

Example saved model names:

```text
a=5 b=005 c=20/dqn_brainblock_seed_42_dense.pth
a=5 b=005 c=20/dqn_brainblock_seed_100_dense.pth
a=5 b=005 c=20/dqn_brainblock_seed_2026_dense.pth
a=5 b=005 c=20/dqn_brainblock_seed_7_dense.pth
a=5 b=005 c=20/dqn_brainblock_seed_99_dense.pth
```

Important note:

The repo already contains a folder called `saved_models/`, but the current `evaluate.py` script is written to look for models inside:

```text
a=5 b=005 c=20/
```

So, for the smoothest run, train first using:

```bash
python train.py
```

Then run evaluation.

---

## 3. Evaluate the Trained Agent

After training is complete, run:

```bash
python evaluate.py
```

The evaluation script loads the trained dense reward models from:

```text
a=5 b=005 c=20/
```

It then runs deterministic evaluation rollouts with exploration turned off.

During evaluation, the script prints:

- Step-by-step rollout trace
- Board visualization
- Final board coverage
- Success rate
- Mean episodic return
- Standard deviation of return
- Mean episode length

---

## State Representation

The environment observation has 50 values:

```text
40 board cells + 5 current piece values + 5 remaining piece counts = 50
```

### Board State

The 8 × 5 board is flattened into 40 values.

```text
0 = empty cell
1 = occupied cell
```

### Current Piece

The current piece is represented using one-hot encoding for:

```text
I, O, L, Z, T
```

### Remaining Pieces

The final 5 values represent how many pieces of each type remain in the queue.

---

## Action Space

The action space contains 320 discrete actions:

```text
8 orientations × 8 x-positions × 5 y-positions = 320 actions
```

Each action is decoded into:

```text
orientation index
x anchor position
y anchor position
```

---

## Reward Functions

The environment supports two reward modes:

### Dense Reward

Dense reward gives feedback after each legal placement.

It includes:

- Reward for covering cells
- Bonus for completing the full board
- Penalty for invalid actions

This reward is better for learning because the agent receives feedback throughout the episode.

### Sparse Reward

Sparse reward mainly rewards full completion.

It includes:

- Completion reward
- Invalid action penalty

This is harder to learn from because the agent receives less feedback during early training.

---

## Action Masking

This project includes legal action masking.

The environment provides:

```python
get_valid_action_mask()
```

it returns a boolean mask of all 320 actions:

```text
True  = legal action
False = illegal action
```

During training and evaluation, invalid actions are blocked by assigning their Q-values to negative infinity.

This helps the agent choose only legal placements and makes learning much more stable.

---

## Neural Network

The main DQN model is defined in:

```text
models/q_network.py
```

The final MLP architecture is:

```text
Input: 50

Linear 50 → 256
ReLU

Linear 256 → 256
ReLU

Linear 256 → 128
ReLU

Linear 128 → 320
```

The output layer gives one Q-value for each possible action.

There is also an alternative CNN model in the same file, but as explained in the code comments explain that the MLP performed better because the board is small.

---

## Evaluation Metrics

The evaluation script reports:

- Success Rate
- Mean Final Grid Coverage
- Mean Episodic Return
- Standard Deviation of Return
- Mean Episode Length

---

## Common Issues

### Problem: `evaluate.py` says model file not found

Reason:

You probably have not run training yet, or the model files are in `saved_models/` instead of `a=5 b=005 c=20/`.

Fix:

Run:

```bash
python train.py
```

Then run:

```bash
python evaluate.py
```

Or edit the model paths in `evaluate.py` to point to `saved_models/`.

---

### Problem: `ModuleNotFoundError: No module named env`

Reason:

You are probably running the script from the wrong directory.

Fix:

Make sure you are inside the main project folder:

```bash
cd Tetromino445-master
python train.py
```

Do not run `train.py` from inside the `env/` folder.

---

### Problem: Training takes time

Reason:

The default training runs:

```text
2000 episodes × 5 seeds × 2 reward modes
```

This is expected.

For a quick test, you can temporarily reduce `num_episodes` in `train.py`:

```python
run_experiment_pipeline(reward_mode="dense", num_episodes=100, filename="dense_reward_performance.png")
```

---

## Recommended Run Order

```bash
cd Tetromino445-master
pip install torch gymnasium numpy matplotlib
python train.py
python evaluate.py
```

For a quick environment-only check:

```bash
cd Tetromino445-master/env
python test_env.py
```

---

## Notes

This project uses a custom DRL environment, so there is no external dataset required. The board and piece queue are generated inside the environment at the start of each episode.

The agent does not memorize a fixed puzzle order because the tetromino queue is shuffled during environment reset.

---

## Project About

CS445 / CS545 Deep Reinforcement Learning   
BrainBlock Packing Environment Project

Members:

- Poyraz Koroglu
- Abdul Rehman Khan
- Baki Cabri
