# test_env.py
import numpy as np
from tetromino import Tetromino


def run_manual_test_simulation():
    env = Tetromino()
    print("Initializing environment test run...")

    # 1. Reset Environment
    obs, info = env.reset()
    env.render()

    print(f"Initial State Shape: {obs.shape}")
    print(f"Remaining counts at start: {obs[-5:]} (Should be 2.0 for all positions)")

    total_reward = 0.0
    steps = 0

    while True:
        # 2. Select a random discrete action index across the 320 scope range
        action = env.action_space.sample()

        # Decode components just to display them in our logs
        orientation = action // (env.W * env.H)
        rem = action % (env.W * env.H)
        x = rem // env.H
        y = rem % env.H

        print(f"\nStep {steps + 1}: Testing random placement -> Orient: {orientation}, Anchor: ({x}, {y})")

        # 3. Process execution step
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Show updated board representation layout
        env.render()
        print(f"Step Reward received: {reward:.3f} | Accumulated Return: {total_reward:.3f}")

        if terminated or truncated:
            print(f"\nEpisode finished! Final Status -> Steps Taken: {steps}, Total Score: {total_reward:.3f}")
            break


if __name__ == "__main__":
    run_manual_test_simulation()