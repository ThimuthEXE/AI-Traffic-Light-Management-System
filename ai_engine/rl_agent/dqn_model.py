"""
Deep Q-Network (DQN) Model Architecture & Experience Replay
General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI

Implements Dueling Double Deep Q-Network (Dueling DDQN) with Experience Replay Buffer.
Decomposes Q(s, a) into State-Value V(s) and Action-Advantage A(s, a):
  Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
"""

import os
import sys
import random
import math
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network.
    Input: State Vector S in R^18
    Output: Q-Values for all discrete traffic signal actions in R^9
    """
    def __init__(self, state_dim: int = 18, action_dim: int = 9, hidden_dim: int = 128):
        super(DuelingDQN, self).__init__()
        
        # Shared Feature Extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean(A))
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values


class ReplayBuffer:
    """Experience Replay Buffer for storing and sampling (s, a, r, s', done) transitions."""
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Learning Agent with Target Network and Epsilon-Greedy Exploration.
    """
    def __init__(
        self,
        state_dim: int = 18,
        action_dim: int = 9,
        lr: float = 1e-3,
        gamma: float = 0.96,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 200,
        batch_size: int = 64
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.train_step_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy and Target Networks
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss for training stability

        self.memory = ReplayBuffer(capacity=50000)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> tuple:
        """Select action using epsilon-greedy or deterministic argmax Q(s, a)."""
        if not evaluate and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            q_vals = np.zeros(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_tensor = self.policy_net(state_tensor)
                q_vals = q_tensor.cpu().numpy()[0]
                action = int(np.argmax(q_vals))
        return action, q_vals

    def update(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # Current Q-values: Q(s, a)
        curr_q = self.policy_net(states).gather(1, actions)

        # Double DQN target computation: a* = argmax_a Q_policy(s', a), Q_target(s', a*)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay exploration epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, filepath)
        print(f"[DQN Agent] Model weights saved to: {filepath}")

    def load(self, filepath: str):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.target_net.load_state_dict(checkpoint['model_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
            self.policy_net.eval()
            print(f"[DQN Agent] Loaded trained weights from: {filepath}")
            return True
        return False
