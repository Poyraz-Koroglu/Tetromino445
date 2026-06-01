import os
import torch
import numpy as np
from env.tetromino import Tetromino
from models.q_network import BrainBlockQNet

def evaluate_pretrained_agent(model_path, num_evaluation_episodes=20):
    """
    Runs deterministic evaluation rollouts with exploration turned off (epsilon = 0).
    Loads a pretrained model weights file and prints full summary statistics.
    """
    # Detect execution hardware device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize environment cleanly with standard evaluation weights
    env = Tetromino(alpha=1.0, beta=0.1, gamma=1.0)

    state_dim = 50
    action_dim = 320

    # 2. Load the trained network weights and push to device
    model = BrainBlockQNet(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set network to evaluation mode (turns off dropout/batchnorm if any)

    returns = []
    episode_lengths = []
    successful_episodes = 0

    print(f"--- Starting Deterministic Evaluation Using {model_path} (Device: {device}) ---")

    for ep in range(1, num_evaluation_episodes + 1):
        state, _ = env.reset()
        ep_return = 0.0
        steps = 0

        # We only print out a full qualitative step trace for the very first episode
        show_trace = (ep == 1)
        if show_trace:
            print("\n*** QUALITATIVE FIRST-EPISODE STEP-BY-STEP ROLLOUT TRACE ***")
            env.render()

        while True:
            # Deterministic Action Selection: No exploration (Epsilon = 0)
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = model(state_t)
                action = q_values.argmax().item()

            # Extract position parameters from discrete index for the trace printout
            orientation = action // (env.W * env.H)
            rem = action % (env.W * env.H)
            x = rem // env.H
            y = rem % env.H

            # Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_return += reward
            steps += 1
            state = next_state

            if show_trace:
                print(
                    f"\n[Step {steps}] Placed Piece -> Orient: {orientation}, Anchor: ({x}, {y}) | Reward: {reward:.3f}")
                env.render()

            if done:
                # If it successfully places all 10 pieces with no invalid action, it's a win
                if steps == 10 and ep_return > 1.0:
                    successful_episodes += 1
                break

        returns.append(ep_return)
        episode_lengths.append(steps)

    # 3. Calculate and print precise metrics required by Section 5 of the PDF
    success_rate = (successful_episodes / num_evaluation_episodes) * 100
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_length = np.mean(episode_lengths)

    print("\n" + "=" * 50)
    print("        OFFICIAL EVALUATION METRICS SUMMARY       ")
    print("=" * 50)
    print(f"Success Rate (Fraction Solved):     {success_rate:.1f}%")
    print(f"Mean Episodic Return:               {mean_return:.3f}")
    print(f"Standard Deviation of Return:       {std_return:.3f}")
    print(f"Mean Episode Length (Steps taken):  {mean_length:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    # Path to one of your saved models from the 5-seed training loop
    saved_model_targets = ["saved_models/dqn_brainblock_seed_7.pth",
                           "saved_models/dqn_brainblock_seed_42.pth",
                           "saved_models/dqn_brainblock_seed_99.pth",
                           "saved_models/dqn_brainblock_seed_100.pth",
                           "saved_models/dqn_brainblock_seed_2026.pth"]

    # Optional tweak for cleaner console reading
    for idx, saved_model_target in enumerate(saved_model_targets):
        if os.path.exists(saved_model_target):
            print(f"\n" + "###" * 15)
            print(f"EVALUATING MODEL: {saved_model_target}")
            print("###" * 15)

            # Pass a flag or customize if you don't want 5 massive step-traces printing out
            evaluate_pretrained_agent(model_path=saved_model_target, num_evaluation_episodes=20)
        else:
            print(f"[Error] Could not find any trained weights file at '{saved_model_target}'.")
            print("Please ensure your train.py has successfully completed a run first to create the saved_models folder!")

