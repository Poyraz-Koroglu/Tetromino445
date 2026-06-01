import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# Import your environment and model components
from env.tetromino import Tetromino
from models.q_network import BrainBlockQNet, ReplayBuffer


def train_dqn(seed=42, num_episodes=1000, alpha=1.0, beta=0.1, gamma=1.0, save_model=True):
    # Set random seeds for reproducibility (required by your evaluation protocol)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Detect execution device automatically (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize environment with your exact reward function weights
    env = Tetromino(alpha=alpha, beta=beta, gamma=gamma)

    # 2. Hyperparameters
    state_dim = 50  # 40 board cells + 5 active piece hot vector + 5 remaining count
    action_dim = 320  # 8 orientations * 8 x-coords * 5 y-coords
    lr = 1e-3  # Learning rate
    discount = 0.99  # Gamma discount factor for future rewards
    batch_size = 64  # Mini-batch size sampled from memory buffer
    memory_cap = 10000  # Replay buffer maximum size

    # Epsilon-Greedy exploration parameters
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    # 3. Instantiate Networks and Replay Buffer (pushed to device)
    policy_net = BrainBlockQNet(state_dim, action_dim).to(device)
    target_net = BrainBlockQNet(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())  # Synchronize networks
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(memory_cap)

    # 4. Metric lists to save for your report figures
    history_rewards = []
    history_covered_area = []
    history_lengths = []

    print(f"--- Starting Training on Seed {seed} (Device: {device}) ---")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed)
        episode_reward = 0.0
        steps = 0

        while True:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(state_t)
                    action = q_values.argmax().item()  # Exploit

            # Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Save transition to memory buffer
            memory.push(state, action, reward, next_state, done)

            # Move network forward
            state = next_state
            episode_reward += reward
            steps += 1

            # Optimize the network if memory has enough experiences
            if len(memory) >= batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)

                # Move sampled batch tensors to the active device
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                # Compute current Q values: Q(s, a)
                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute target Q values: r + discount * max_a Q_target(s', a)
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + (discount * max_next_q * (1 - dones))

                # Loss computation (Mean Squared Error)
                loss = nn.MSELoss()(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Update target network every few episodes
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Calculate covered area (each unique piece correctly placed covers 4 cells)
        successful_placements = steps if episode_reward > 0.5 else (steps - 1)
        covered_area = max(0, successful_placements * 4)

        # Save historical records for plotting figures later
        history_rewards.append(episode_reward)
        history_covered_area.append(covered_area)
        history_lengths.append(steps)

        # Log progress periodically
        if episode % 100 == 0 or episode == 1:
            print(
                f"Episode {episode:4d} | Reward: {episode_reward:6.2f} | Covered Cells: {covered_area:2d}/40 | Epsilon: {epsilon:.3f}")

    print(f"--- Seed {seed} Training Completed! ---")

    if save_model:
        os.makedirs("saved_models", exist_ok=True)
        torch.save(policy_net.state_dict(), f"saved_models/dqn_brainblock_seed_{seed}.pth")

    return history_rewards, history_covered_area, history_lengths


if __name__ == "__main__":
    # --- EXPERIMENTAL EVALUATION PROTOCOL SETUP ---
    # Define the 5 unique seeds required to fulfill the rubric
    seeds = [42, 100, 2026, 7, 99]
    num_episodes_per_seed = 500  # 500 is optimal for stable convergence here

    all_seeds_rewards = []
    all_seeds_coverage = []
    all_seeds_lengths = []

    # Run through each seed sequentially using our base training function
    for idx, current_seed in enumerate(seeds):
        print(f"\n" + "=" * 50)
        print(f"EXECUTING EXPERIMENT SEED {idx + 1}/5 (SEED ID: {current_seed})")
        print("=" * 50)

        rewards, coverage, lengths = train_dqn(
            seed=current_seed,
            num_episodes=num_episodes_per_seed,
            alpha=1.0,
            beta=0.1,
            gamma=1.0,
            save_model=True
        )

        all_seeds_rewards.append(rewards)
        all_seeds_coverage.append(coverage)
        all_seeds_lengths.append(lengths)

    # Convert data to numpy arrays for matrix averaging operations [Shape: (5, 500)]
    all_seeds_rewards = np.array(all_seeds_rewards)
    all_seeds_coverage = np.array(all_seeds_coverage)
    all_seeds_lengths = np.array(all_seeds_lengths)

    # Compute performance means across all 5 seeds
    mean_rewards = np.mean(all_seeds_rewards, axis=0)
    mean_coverage = np.mean(all_seeds_coverage, axis=0)
    mean_lengths = np.mean(all_seeds_lengths, axis=0)

    # 5. Generate and save the required report plots automatically [cite: 104, 118]
    episodes_range = np.arange(1, num_episodes_per_seed + 1)
    plt.figure(figsize=(15, 4))

    # Plot 1: Total Reward vs Episode [cite: 119]
    plt.subplot(1, 3, 1)
    plt.plot(episodes_range, mean_rewards, color='blue')
    plt.title('Total Reward vs. Episode #')
    plt.xlabel('Episode #')
    plt.ylabel('Return')
    plt.grid(True)

    # Plot 2: Total Covered Area vs Episode [cite: 120]
    plt.subplot(1, 3, 2)
    plt.plot(episodes_range, mean_coverage, color='green')
    plt.title('Total Covered Area vs. Episode #')
    plt.xlabel('Episode #')
    plt.ylabel('Covered Cells (Max 40)')
    plt.grid(True)

    # Plot 3: Episode Length over Time [cite: 121]
    plt.subplot(1, 3, 3)
    plt.plot(episodes_range, mean_lengths, color='red')
    plt.title('Episode Length over Time')
    plt.xlabel('Episode #')
    plt.ylabel('Steps Taken')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('dense_reward_performance.png')
    print("\n[Success] All 5 seeds trained! Learning curves saved as 'dense_reward_performance.png'.")