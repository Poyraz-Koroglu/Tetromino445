import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Tetromino(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,alpha=1.0,beta=0.1,gamma=1.0):
        super(Tetromino, self).__init__()

        # Grid parameters [cite: 11, 13]
        self.W = 8
        self.H = 5

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # 1. Action Space: Discrete index up to 319 [cite: 74, 76]
        # Total size = 8 orientations * 8 x-anchors * 5 y-anchors = 320 [cite: 75, 76]
        self.action_space = spaces.Discrete(8 * self.W * self.H)

        # 2. State Representation (Exactly 50 continuous/binary features)
        # Board (40 values) + Current Piece One-Hot (5 values) + Remaining Count (5 values)
        self.observation_space = spaces.Box(
            low=0.0,
            high=10.0,  # Remaining counts can go up to 2
            shape=(50,),
            dtype=np.float32
        )

        # Define the 5 piece types [cite: 28]
        self.piece_types = ['I', 'O', 'L', 'Z', 'T']

        # Core geometric layout definitions for all 8 transformations (rotations/reflections)
        # Each tuple represents relative (dx, dy) from the anchor coordinate (0,0) [cite: 6]
        self.piece_shapes = {
            'I': {
                0: [(0, 0), (1, 0), (2, 0), (3, 0)],  # Horizontal right
                1: [(0, 0), (0, 1), (0, 2), (0, 3)],  # Vertical up
                2: [(0, 0), (-1, 0), (-2, 0), (-3, 0)],  # Horizontal left
                3: [(0, 0), (0, -1), (0, -2), (0, -3)],  # Vertical down
                4: [(0, 0), (1, 0), (2, 0), (3, 0)],  # Redundant symmetric variations
                5: [(0, 0), (0, 1), (0, 2), (0, 3)],
                6: [(0, 0), (-1, 0), (-2, 0), (-3, 0)],
                7: [(0, 0), (0, -1), (0, -2), (0, -3)]
            },
            'O': {  # Completely symmetric: all orientations map to the same 2x2 square box
                i: [(0, 0), (1, 0), (0, 1), (1, 1)] for i in range(8)
            },
            'L': {
                0: [(0, 0), (0, 1), (0, 2), (1, 0)],
                1: [(0, 0), (1, 0), (2, 0), (0, 1)],
                2: [(0, 0), (1, 0), (1, 1), (1, 2)],
                3: [(0, 0), (0, 1), (-1, 1), (-2, 1)],
                4: [(0, 0), (1, 0), (2, 0), (2, 1)],  # Reflected orientations
                5: [(0, 0), (0, 1), (0, 2), (-1, 0)],
                6: [(0, 0), (0, 1), (1, 1), (2, 1)],
                7: [(0, 0), (1, 0), (0, 1), (0, 2)]
            },
            'Z': {
                0: [(0, 0), (1, 0), (1, 1), (2, 1)],
                1: [(0, 0), (0, 1), (-1, 1), (-1, 2)],
                2: [(0, 0), (1, 0), (1, 1), (2, 1)],  # Symmetric rotations
                3: [(0, 0), (0, 1), (-1, 1), (-1, 2)],
                4: [(0, 0), (1, 0), (0, 1), (-1, 1)],  # Flipped variations
                5: [(0, 0), (0, 1), (1, 1), (1, 2)],
                6: [(0, 0), (1, 0), (0, 1), (-1, 1)],
                7: [(0, 0), (0, 1), (1, 1), (1, 2)]
            },
            'T': {
                0: [(0, 0), (-1, 0), (1, 0), (0, 1)],  # T pointing up
                1: [(0, 0), (0, 1), (0, -1), (1, 0)],  # T pointing right
                2: [(0, 0), (-1, 0), (1, 0), (0, -1)],  # T pointing down
                3: [(0, 0), (0, 1), (0, -1), (-1, 0)],  # T pointing left
                4: [(0, 0), (-1, 0), (1, 0), (0, 1)],  # Remaining indices map to duplicates
                5: [(0, 0), (0, 1), (0, -1), (1, 0)],
                6: [(0, 0), (-1, 0), (1, 0), (0, -1)],
                7: [(0, 0), (0, 1), (0, -1), (-1, 0)]
            }
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize an empty 8x5 grid matrix [cite: 11, 13]
        self.board = np.zeros((self.H, self.W), dtype=np.int32)

        # Build inventory queue: 2 copies of each tetromino [cite: 31]
        self.queue = ['I', 'I', 'O', 'O', 'L', 'L', 'Z', 'Z', 'T', 'T']
        self.np_random.shuffle(self.queue)  # Shuffled pool sequence [cite: 47]

        self.current_step = 0

        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Flatten the board state -> 40 values
        flattened_board = self.board.flatten().astype(np.float32)

        # 2. Get the current active piece type -> 5 values
        piece_one_hot = np.zeros(5, dtype=np.float32)
        if self.current_step < 10:
            current_piece = self.queue[self.current_step]
            piece_one_hot[self.piece_types.index(current_piece)] = 1.0

        # 3. Calculate remaining counts in queue pool -> 5 values [cite: 69]
        remaining_counts = np.zeros(5, dtype=np.float32)
        for p in self.queue[self.current_step:]:
            remaining_counts[self.piece_types.index(p)] += 1.0

        # Total concatenated length = 40 + 5 + 5 = 50 values
        return np.concatenate([flattened_board, piece_one_hot, remaining_counts])

    def step(self, action):
        orientation = action // (self.W * self.H)
        remainder = action % (self.W * self.H)
        x_anchor = remainder // self.H
        y_anchor = remainder % self.H

        terminated = False
        truncated = False
        reward = 0.0

        if self.current_step >= 10:
            return self._get_obs(), 0.0, True, False, {}

        current_piece_type = self.queue[self.current_step]
        relative_offsets = self.piece_shapes[current_piece_type][orientation]

        legal_placement = True
        absolute_coordinates = []

        for (dx, dy) in relative_offsets:
            target_x = x_anchor + dx
            target_y = y_anchor + dy

            if not (0 <= target_x < self.W and 0 <= target_y < self.H):
                legal_placement = False
                break
            if self.board[target_y, target_x] == 1:
                legal_placement = False
                break

            absolute_coordinates.append((target_x, target_y))

        # --- IMPLEMENTING YOUR REWARD FUNCTION SPECIFICATION ---
        if legal_placement:
            # Commit piece to board
            for (tx, ty) in absolute_coordinates:
                self.board[ty, tx] = 1

            # Component 1: alpha * (new cells covered / 40)
            # Each tetromino piece occupies exactly 4 unit cells
            new_cells_covered = 4
            reward += self.alpha * (new_cells_covered / 40.0)

            self.current_step += 1

            # Component 3: gamma * (1 if episode completed else 0)
            if self.current_step == 10:
                reward += self.gamma * 1.0
                terminated = True
        else:
            # Component 2: -beta * (1 if invalid action else 0)
            reward -= self.beta * 1.0
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        # Simple ASCII display console visual representation
        print("\n" + "=" * 19)
        for r in reversed(range(self.H)):
            row_str = " ".join(["█" if self.board[r, c] == 1 else "." for c in range(self.W)])
            print(f"{r} | {row_str}")
        print("  " + "-" * 17)
        print("    0 1 2 3 4 5 6 7")
        if self.current_step < 10:
            print(f"Next Piece: {self.queue[self.current_step]}")
        print("=" * 19)