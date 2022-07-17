from DQN import DQN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

if __name__ == "__main__":
    # DQN
    env = gym.make('CartPole-v1')
    dqn = DQN(env)
    dqn.load_model("./DQN/trained.pth")
    for i in range(1000):
        state = env.reset()
        for j in range(2000):
            # env.render()
            action = dqn.choose_action(state)
            next_state, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            dqn.store_transition(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        dqn.learn()
        print(i, j)
    dqn.save_model("./DQN/trained.pth")
    env.close()
    print('Done')
