import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import env
from model.dqn_model import DQNAgent

if __name__ == '__main__':
    # Environment parameters:
    num_agents = 3
    num_tasks = 15
    boundary = 50
    render_mode = "rgb_array"
    # render_mode = None

    env = gym.make('env/mTSP-v0', num_agents=num_agents,
                    num_tasks=num_tasks,
                    boundary=boundary, render_mode=render_mode)
    
    agent = DQNAgent(gamma=0.99,
                  input_dims=(env.observation_space.shape),
                  epsilon=1.0, batch_size=64, n_actions=env.action_space.n,
                  eps_end=0.01, lr=0.003, T=1000)
    
    score = 0
    random_seed = 8
    done = False
    observation, info = env.reset(seed=random_seed)

    model_filepath = './data/dqn_model.pt'
    agent.load(model_filepath)

    while not done:
        action = agent.extract_greedy_action(observation)
        observation_, reward, done, truncated, info = env.step(action)

        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_  # state = new state
    print('Final solution ','score %.2f' % score)

    rgb_array = env.render()
    plt.imsave("data/img/dqn_solution.png", rgb_array)