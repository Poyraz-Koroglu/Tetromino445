import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# MLP NETWORK (RECOMMENDED — outperformed CNN in initial tests)
# Simple fully connected network. Board is small (40 cells) so MLP handles
# spatial relationships well without the added complexity of convolutions.
# =============================================================================
class BrainBlockQNet(nn.Module):
    def __init__(self, state_dim=50, action_dim=320):
        super(BrainBlockQNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


# =============================================================================
# CNN NETWORK (ALTERNATIVE — kept for report comparison only)
# Splits observation into board (40) and piece info (10), processes board
# through conv layers then concatenates with piece features.
# Underperformed MLP in tests — likely due to small board size (8x5).
# =============================================================================
class BrainBlockQNetCNN(nn.Module):
    def __init__(self, state_dim=50, action_dim=320):
        super(BrainBlockQNetCNN, self).__init__()

        # Spatial stream: processes 8x5 board
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()  # 64 * 5 * 8 = 2560
        )

        # Vector stream: processes piece one-hot + remaining counts (10 values)
        self.piece_fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU()
        )

        # Combined head: 2560 + 32 = 2592
        self.policy_head = nn.Sequential(
            nn.Linear(2560 + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        board_features = x[:, :40].reshape(-1, 1, 5, 8)
        piece_features = x[:, 40:]

        spatial_out = self.conv_net(board_features)
        vector_out = self.piece_fc(piece_features)

        combined = torch.cat((spatial_out, vector_out), dim=1)
        return self.policy_head(combined)


# =============================================================================
# REPLAY BUFFER
# =============================================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(np.array(action), dtype=torch.long),
            torch.tensor(np.array(reward), dtype=torch.float32),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(np.array(done), dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)