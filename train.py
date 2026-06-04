import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

from env.tetromino import Tetromino
from models.q_network import BrainBlockQNet, ReplayBuffer


def train_dqn(seed=42, num_episodes=2000, alpha=5.0, beta=0.05, gamma=20.0, reward_mode="dense", save_model=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Tetromino(alpha=alpha, beta=beta, gamma=gamma, reward_mode=reward_mode)

    state_dim = 50
    action_dim = 320
    lr = 1e-3
    discount = 0.99
    batch_size = 64
    memory_cap = 10000

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.998

    policy_net = BrainBlockQNet(state_dim, action_dim).to(device)
    target_net = BrainBlockQNet(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(memory_cap)

    history_rewards = []
    history_covered_area = []
    history_lengths = []

    print(f"--- Starting Training on Seed {seed} | Mode: {reward_mode} | Device: {device} ---")

    for episode in range(1, num_episodes + 1):
        # FIX: Only pass seed on episode 1 so queue shuffles differently each episode.
        # Passing seed every reset made every episode identical — agent couldn't generalize.
        if episode == 1:
            state, _ = env.reset(seed=seed)
        else:
            state, _ = env.reset()

        episode_reward = 0.0
        steps = 0

        while True:
            # Get valid action mask from environment
            valid_mask = env.get_valid_action_mask()  # shape: (320,) bool array
            valid_indices = np.where(valid_mask)[0]

            if random.random() < epsilon:
                # Explore: sample only from valid actions (not random over all 320)
                if len(valid_indices) > 0:
                    action = int(np.random.choice(valid_indices))
                else:
                    action = env.action_space.sample()  # fallback (shouldn't happen)
            else:
                # Exploit: set Q-values of invalid actions to -inf before argmax
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(state_t).squeeze(0)  # shape: (320,)
                    mask_t = torch.tensor(valid_mask, dtype=torch.bool).to(device)
                    q_values[~mask_t] = float('-inf')  # block all invalid actions
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            steps += 1

            if steps % 4 == 0 and len(memory) >= batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)

                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + (discount * max_next_q * (1 - dones))

                loss = nn.MSELoss()(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        covered_area = env.current_step * 4
        history_rewards.append(episode_reward)
        history_covered_area.append(covered_area)
        history_lengths.append(steps)

        if episode % 100 == 0 or episode == 1:
            print(f"Episode {episode:4d} | Reward: {episode_reward:6.2f} | Covered: {covered_area:2d}/40 | Epsilon: {epsilon:.3f}")

    print(f"--- Seed {seed} ({reward_mode}) Training Complete ---")

    if save_model:
        os.makedirs("a=5 b=005 c=20", exist_ok=True)
        #torch.save(policy_net.state_dict(), f"saved_models/dqn_brainblock_seed_{seed}_{reward_mode}.pth")
        #torch.save(policy_net.state_dict(), f"a=2 b=01 g=5/dqn_brainblock_seed_{seed}_{reward_mode}.pth")
        torch.save(policy_net.state_dict(), f"a=5 b=005 c=20/dqn_brainblock_seed_{seed}_{reward_mode}.pth")
    return history_rewards, history_covered_area, history_lengths


def run_experiment_pipeline(reward_mode="dense", num_episodes=2000, filename="dense_reward_performance.png"):
    seeds = [42, 100, 2026, 7, 99]

    all_seeds_rewards = []
    all_seeds_coverage = []
    all_seeds_lengths = []

    for idx, current_seed in enumerate(seeds):
        print(f"\n" + "=" * 50)
        print(f"EXECUTING {reward_mode.upper()} RUN: SEED {idx + 1}/5 (ID: {current_seed})")
        print("=" * 50)

        rewards, coverage, lengths = train_dqn(
            seed=current_seed,
            num_episodes=num_episodes,
            alpha=0.5,   # as agreed
            beta=0.1,    # starting at 0.5, decrease if needed after tests
            gamma=1.0,
            reward_mode=reward_mode,
            save_model=True
        )

        all_seeds_rewards.append(rewards)
        all_seeds_coverage.append(coverage)
        all_seeds_lengths.append(lengths)

    all_seeds_rewards = np.array(all_seeds_rewards)
    all_seeds_coverage = np.array(all_seeds_coverage)
    all_seeds_lengths = np.array(all_seeds_lengths)

    mean_rewards = np.mean(all_seeds_rewards, axis=0)
    mean_coverage = np.mean(all_seeds_coverage, axis=0)
    mean_lengths = np.mean(all_seeds_lengths, axis=0)

    episodes_range = np.arange(1, num_episodes + 1)
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(episodes_range, mean_rewards, color='blue')
    plt.title(f'Total Reward ({reward_mode})')
    plt.xlabel('Episode #')
    plt.ylabel('Return')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(episodes_range, mean_coverage, color='green')
    plt.title(f'Covered Area ({reward_mode})')
    plt.xlabel('Episode #')
    plt.ylabel('Covered Cells (Max 40)')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(episodes_range, mean_lengths, color='red')
    plt.title(f'Episode Length ({reward_mode})')
    plt.xlabel('Episode #')
    plt.ylabel('Steps Taken')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n[Success] Figures saved to '{filename}'")


if __name__ == "__main__":

    run_experiment_pipeline(reward_mode="dense", num_episodes=2000, filename="dense_reward_performance.png")
    run_experiment_pipeline(reward_mode="sparse", num_episodes=2000, filename="sparse_reward_performance.png")
    print("\nAll pipelines complete.")