"""
@brief: DQN Agent
@author: czx
@date: 2022.7.16
"""

from sre_parse import State
import numpy as np
import torch
import torch.nn as nn
import random


class Net(nn.Module):
    """
    the DQN network
    """

    def __init__(self, state_size, action_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )

    def forward(self, x):
        x = self.net(x)

        return x


class DQN:
    """
    DQN Agent
    """

    def __init__(self, env) -> None:
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.eval_net = Net(self.state_size, self.action_size)
        self.target_net = Net(self.state_size, self.action_size)
        self.memory = []
        self.learn_step = 0
        self.gamma = 0.9
        self.epsilon = 0.9
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.learn_rate = 0.001
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=self.learn_rate)

    def choose_action(self, state):
        """
        choose action
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            q_value = self.eval_net(state)
            action = torch.max(q_value, 1)[1].data.numpy()[0]
            return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        store transition
        """
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """
        learn
        """
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        
        state_batch = torch.from_numpy(np.vstack([x[0] for x in batch])).float()
        action_batch = torch.from_numpy(np.vstack([x[1] for x in batch])).long()
        reward_batch = torch.from_numpy(np.vstack([x[2] for x in batch])).float()
        next_state_batch = torch.from_numpy(np.vstack([x[3] for x in batch])).float()
        done_batch = torch.from_numpy(np.vstack([x[4] for x in batch])).float()

        q_eval = self.eval_net(state_batch).gather(1, action_batch)
        q_next = self.target_net(next_state_batch).detach()
        q_target = reward_batch + self.gamma * \
            q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step += 1
        if self.learn_step % 1000 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path)

    def load_model(self, path):
        self.eval_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))
