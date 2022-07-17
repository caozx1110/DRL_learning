import gym

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env.reset()
    print(env.action_space)
    print(env.observation_space)
    done = False
    for i in range(1000):
        while not done:
            env.render()
            chosen_action = env.action_space.sample()
            observation, reward, done, info = env.step(chosen_action)
            print(observation, reward, done, info)
            
    env.close()
    print('Done')
    