"""
Deep Q-Network (DQN) Training Pipeline
General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI

Trains Dueling DDQN agent across 50 episodes of diverse traffic flow scenarios:
- Morning Rush Hour
- Evening Rush Hour
- Dual Axis Surges (L + D)
- Asymmetric Single Surge (U = 150 v/m)
- Balanced High Inflow
"""

import os
import sys
import time
import math
import random
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_engine.rl_agent.dqn_model import DQNAgent
from ai_engine.rl_agent.traffic_env import TrafficRLEnvironment

def train_dqn_agent(num_episodes: int = 50):
    print("=" * 75)
    print("STARTING DEEP Q-NETWORK (DQN) REINFORCEMENT LEARNING TRAINING")
    print(f"Total Episodes: {num_episodes} (each episode = 300s simulated traffic)")
    print("State Dimension: 18 | Action Space: 9 Discrete Actions")
    print("Reward Objectives: Min AWT + Min Max-Wait + Max Throughput")
    print("=" * 75)

    env = TrafficRLEnvironment(step_duration_sec=3.0)
    agent = DQNAgent(
        state_dim=18,
        action_dim=9,
        lr=1e-3,
        gamma=0.96,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.985,
        target_update_freq=50,
        batch_size=64
    )

    scenarios = ["user_experiment", "morning_rush", "evening_rush", "dual_surge", "random"]
    episode_rewards = []
    episode_awts = []
    episode_max_waits = []
    episode_cleared = []

    start_wall = time.time()

    for ep in range(1, num_episodes + 1):
        scenario = scenarios[ep % len(scenarios)]
        state = env.reset(scenario=scenario)
        total_reward = 0.0
        losses = []
        done = False

        while not done:
            action, _ = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, float(done))

            loss = agent.update()
            if loss > 0: losses.append(loss)

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_awts.append(info['awt'])
        episode_max_waits.append(info['max_wait'])
        episode_cleared.append(info['cleared'])

        avg_loss = np.mean(losses) if losses else 0.0
        elapsed_sec = time.time() - start_wall

        if ep % 5 == 0 or ep == 1 or ep == num_episodes:
            print(f"Ep {ep:2d}/{num_episodes} [{scenario:<15}] | Total Reward: {total_reward:6.1f} | AWT: {info['awt']:4.1f}s | MaxWait: {info['max_wait']:4.1f}s | Cleared: {info['cleared']:3d} veh | Epsilon: {agent.epsilon:.3f} | Wall: {elapsed_sec:.1f}s")

    model_dir = os.path.join(PROJECT_ROOT, "ai_engine", "traffic_predictor", "saved_models")
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "dqn_traffic_agent.pth")
    agent.save(model_save_path)

    total_training_wall = time.time() - start_wall
    print("=" * 75)
    print(f"DQN TRAINING COMPLETED in {total_training_wall:.2f}s")
    print(f"Final Episode Performance: AWT = {episode_awts[-1]:.2f}s | Max Wait = {episode_max_waits[-1]:.2f}s | Cleared = {episode_cleared[-1]} veh")
    print(f"Model successfully saved to:\n{model_save_path}")
    print("=" * 75)

    return model_save_path

if __name__ == "__main__":
    train_dqn_agent(num_episodes=50)
