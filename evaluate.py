import os
import torch
import numpy as np
from env.tetromino import Tetromino
from models.q_network import BrainBlockQNet


def evaluate_pretrained_agent(model_path, reward_mode="dense", num_evaluation_episodes=20):
    """
    Runs deterministic evaluation rollouts with exploration turned off (epsilon = 0).
    Applies real-time action masking to align perfectly with updated training rules.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environment cleanly matching the current configuration choices
    env = Tetromino(alpha=5.0, beta=0.05, gamma=20.0, reward_mode=reward_mode)

    state_dim = 50
    action_dim = 320

    # Load trained model and freeze weights for pure evaluation validation
    model = BrainBlockQNet(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    returns = []
    board_coverages = []
    episode_lengths = []
    successful_episodes = 0

    print(f"--- Starting Deterministic Masked Evaluation Using {model_path} | Mode: {reward_mode} ---")

    for ep in range(1, num_evaluation_episodes + 1):
        state, _ = env.reset()
        ep_return = 0.0
        steps = 0

        show_trace = (ep == 1)
        if show_trace:
            print(f"\n*** QUALITATIVE VALIDATION STEP-BY-STEP ROLLOUT TRACE ({reward_mode.upper()}) ***")
            env.render()

        while True:
            # 1. Extract valid action matrix from the environment state
            valid_mask = env.get_valid_action_mask()

            # Deterministic Action Selection with Active Masking Constraints
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = model(state_t).squeeze(0)  # shape: (320,)

                # Apply Boolean Mask: Force invalid branches to negative infinity
                mask_t = torch.tensor(valid_mask, dtype=torch.bool).to(device)
                q_values[~mask_t] = float('-inf')

                action = q_values.argmax().item()

            # Decode indices for visualization trace tracking output prints
            orientation = action // (env.W * env.H)
            rem = action % (env.W * env.H)
            x = rem // env.H
            y = rem % env.H

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_return += reward
            steps += 1
            state = next_state

            if show_trace:
                print(
                    f"[Step {steps}] Legally Placed -> Orient: {orientation}, Anchor: ({x}, {y}) | Step Reward: {reward:.3f}")
                env.render()

            if done:
                # Track exact cell counts completed on grid board
                actual_cells_covered = np.sum(env.board == 1)
                board_coverages.append(actual_cells_covered)

                if actual_cells_covered == 40:
                    successful_episodes += 1
                    print(f"\n🎉 PERFECT TASK SOLUTION COMPLETE! Printing layout:")
                    env.render()
                break

        returns.append(ep_return)
        episode_lengths.append(steps)

    # Compute official experimental pipeline mean/std statistics metrics
    success_rate = (successful_episodes / num_evaluation_episodes) * 100
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_length = np.mean(episode_lengths)
    mean_coverage = np.mean(board_coverages)

    print("\n" + "=" * 50)
    print(f"    OFFICIAL EVALUATION METRICS SUMMARY ({reward_mode.upper()})    ")
    print("=" * 50)
    print(f"Success Rate (Fraction Solved):     {success_rate:.1f}%")
    print(f"Mean Final Grid Coverage:           {mean_coverage:.1f}/40 cells")
    print(f"Mean Episodic Return:               {mean_return:.3f}")
    print(f"Standard Deviation of Return:       {std_return:.3f}")
    print(f"Mean Episode Length (Steps taken):  {mean_length:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    # Path configuration mapped directly to your 5-seed pipelines target runs
    dense_model_targets = [
        "a=5 b=005 c=20/dqn_brainblock_seed_42_dense.pth",
        "a=5 b=005 c=20/dqn_brainblock_seed_100_dense.pth",
        "a=5 b=005 c=20/dqn_brainblock_seed_2026_dense.pth",
        "a=5 b=005 c=20/dqn_brainblock_seed_7_dense.pth",
        "a=5 b=005 c=20/dqn_brainblock_seed_99_dense.pth"
    ]

    for target in dense_model_targets:
        if os.path.exists(target):
            evaluate_pretrained_agent(model_path=target, reward_mode="dense", num_evaluation_episodes=20)
        else:
            print(f"[Notice] Could not locate target weights file for validation loop at: {target}")