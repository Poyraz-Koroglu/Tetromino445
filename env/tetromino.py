import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Tetromino(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, alpha=5.0, beta=0.05, gamma=20.0, reward_mode="dense"):
        super(Tetromino, self).__init__()

        # Grid parameters
        self.W = 8
        self.H = 5

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reward_mode = reward_mode  # "dense" or "sparse"

        # Action Space: 8 orientations * 8 x-anchors * 5 y-anchors = 320
        self.action_space = spaces.Discrete(8 * self.W * self.H)

        # Observation space: board (40) + current piece one-hot (5) + remaining counts (5) = 50
        self.observation_space = spaces.Box(
            low=0.0,
            high=10.0,
            shape=(50,),
            dtype=np.float32
        )

        self.piece_types = ['I', 'O', 'L', 'Z', 'T']

        # All 8 transformations per piece type.
        # Orientations 0-3: rotations. 4-7: reflections (mirrored variants).
        # Redundant indices for symmetric pieces map to equivalent transforms as required.
        self.piece_shapes = {
            'I': {
                0: [(0, 0), (1, 0), (2, 0), (3, 0)],   # Horizontal right
                1: [(0, 0), (0, 1), (0, 2), (0, 3)],   # Vertical up
                2: [(0, 0), (-1, 0), (-2, 0), (-3, 0)], # Horizontal left (redundant)
                3: [(0, 0), (0, -1), (0, -2), (0, -3)], # Vertical down (redundant)
                4: [(0, 0), (1, 0), (2, 0), (3, 0)],
                5: [(0, 0), (0, 1), (0, 2), (0, 3)],
                6: [(0, 0), (-1, 0), (-2, 0), (-3, 0)],
                7: [(0, 0), (0, -1), (0, -2), (0, -3)]
            },
            'O': {  # Fully symmetric: all orientations identical
                i: [(0, 0), (1, 0), (0, 1), (1, 1)] for i in range(8)
            },
            'L': {
                0: [(0, 0), (0, 1), (0, 2), (1, 0)],
                1: [(0, 0), (1, 0), (2, 0), (0, 1)],
                2: [(0, 0), (1, 0), (1, 1), (1, 2)],
                3: [(0, 0), (0, 1), (-1, 1), (-2, 1)],
                4: [(0, 0), (1, 0), (2, 0), (2, 1)],   # Mirrored (J-shape)
                5: [(0, 0), (0, 1), (0, 2), (-1, 0)],
                6: [(0, 0), (0, 1), (1, 1), (2, 1)],
                7: [(0, 0), (1, 0), (0, 1), (0, 2)]
            },
            'Z': {
                0: [(0, 0), (1, 0), (1, 1), (2, 1)],
                1: [(0, 0), (0, 1), (-1, 1), (-1, 2)],
                2: [(0, 0), (1, 0), (1, 1), (2, 1)],   # Redundant rotation
                3: [(0, 0), (0, 1), (-1, 1), (-1, 2)],
                4: [(0, 0), (1, 0), (0, 1), (-1, 1)],  # S-shape (flipped Z)
                5: [(0, 0), (0, 1), (1, 1), (1, 2)],
                6: [(0, 0), (1, 0), (0, 1), (-1, 1)],
                7: [(0, 0), (0, 1), (1, 1), (1, 2)]
            },
            'T': {
                0: [(0, 0), (-1, 0), (1, 0), (0, 1)],  # T pointing up
                1: [(0, 0), (0, 1), (0, -1), (1, 0)],  # T pointing right
                2: [(0, 0), (-1, 0), (1, 0), (0, -1)], # T pointing down
                3: [(0, 0), (0, 1), (0, -1), (-1, 0)], # T pointing left
                4: [(0, 0), (-1, 0), (1, 0), (0, 1)],
                5: [(0, 0), (0, 1), (0, -1), (1, 0)],
                6: [(0, 0), (-1, 0), (1, 0), (0, -1)],
                7: [(0, 0), (0, 1), (0, -1), (-1, 0)]
            }
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = np.zeros((self.H, self.W), dtype=np.int32)

        self.queue = ['I', 'I', 'O', 'O', 'L', 'L', 'Z', 'Z', 'T', 'T']
        random.shuffle(self.queue)
        #print(f"DEBUG queue: {self.queue}")
        self.current_step = 0
        self.total_actions_taken = 0

        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Board state: 40 binary values
        board_flat = self.board.flatten().astype(np.float32)

        # 2. Current piece one-hot: 5 values
        piece_one_hot = np.zeros(5, dtype=np.float32)
        if self.current_step < 10:
            current_piece = self.queue[self.current_step]
            piece_one_hot[self.piece_types.index(current_piece)] = 1.0

        # 3. Remaining piece counts: 5 values (one per piece type)
        remaining_counts = np.zeros(5, dtype=np.float32)
        for p in self.queue[self.current_step:]:
            remaining_counts[self.piece_types.index(p)] += 1.0

        # Concatenate: 40 + 5 + 5 = 50 values (no hacks, no overwrites)
        obs = np.concatenate([board_flat, piece_one_hot, remaining_counts])
        return obs

    def _is_legal(self, piece_type, orientation, x_anchor, y_anchor):
        """Check if a placement is legal. Returns (bool, list of cells)."""
        offsets = self.piece_shapes[piece_type][orientation]
        cells = []
        for (dx, dy) in offsets:
            tx = x_anchor + dx
            ty = y_anchor + dy
            if not (0 <= tx < self.W and 0 <= ty < self.H):
                return False, []
            if self.board[ty, tx] == 1:
                return False, []
            cells.append((tx, ty))
        return True, cells

    def get_valid_action_mask(self):
        """
        Returns a boolean array of shape (320,) where True = valid action.
        Used for action masking in the DQN to eliminate invalid action exploration.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        if self.current_step >= 10:
            return mask

        current_piece = self.queue[self.current_step]
        for action in range(self.action_space.n):
            orientation = action // (self.W * self.H)
            remainder = action % (self.W * self.H)
            x_anchor = remainder // self.H
            y_anchor = remainder % self.H
            legal, _ = self._is_legal(current_piece, orientation, x_anchor, y_anchor)
            if legal:
                mask[action] = True
        return mask

    def step(self, action):
        self.total_actions_taken += 1

        # Decode action
        orientation = action // (self.W * self.H)
        remainder = action % (self.W * self.H)
        x_anchor = remainder // self.H
        y_anchor = remainder % self.H

        terminated = False
        truncated = False
        reward = 0.0

        # Guard: episode already over
        if self.current_step >= 10:
            return self._get_obs(), 0.0, True, False, {}

        current_piece = self.queue[self.current_step]
        legal, cells = self._is_legal(current_piece, orientation, x_anchor, y_anchor)

        if legal:
            # Place piece on board
            for (tx, ty) in cells:
                self.board[ty, tx] = 1

            self.current_step += 1

            if self.reward_mode == "dense":
                # Coverage reward: alpha * (4 cells placed / 40 total cells)
                reward += self.alpha * (4.0 / 40.0)

                # Completion bonus
                if self.current_step == 10:
                    reward += self.gamma
                    terminated = True

            else:  # sparse
                # No intermediate reward; only reward on full completion
                if self.current_step == 10:
                    reward += self.gamma
                    terminated = True


        else:

            reward -= self.beta

            # Debug: check if any valid actions exist for current piece

            valid_mask = self.get_valid_action_mask()
            if np.sum(valid_mask) == 0:
                # Board is unsolvable — terminate with penalty
                reward -= self.beta
                truncated = True
                return self._get_obs(), reward, terminated, truncated, {}

            reward -= self.beta

        if self.total_actions_taken >= 100:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        print("\n" + "=" * 19)
        for r in reversed(range(self.H)):
            row_str = " ".join(["█" if self.board[r, c] == 1 else "." for c in range(self.W)])
            print(f"{r} | {row_str}")
        print("  " + "-" * 17)
        print("    0 1 2 3 4 5 6 7")
        if self.current_step < 10:
            print(f"Next Piece: {self.queue[self.current_step]}")
        print("=" * 19)