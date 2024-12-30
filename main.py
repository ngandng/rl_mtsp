import gymnasium as gym
import env
import numpy as np
import matplotlib.pyplot as plt

from dqn_model import DQNAgent
from utils import plot_learning_curve

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
    scores, eps_history = [], []
    n_games = 1500
    random_seed = 8

    for i in range(n_games):
        score = 0
        done = False
        observation, info = env.reset(seed=random_seed)

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)

            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_  # state = new state

        scores.append(score)
        # clipping score
        # scores = scores[-300:]
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-50:])

        print('episode ', i, 'score %.2f' % score, 
                'average score %.2f' % avg_score, 
                'epsilon %.2f' % agent.epsilon)
        
    # extract final solution
    score = 0
    done = False
    observation, info = env.reset(seed=random_seed)

    while not done:
        action = agent.extract_greedy_action(observation)
        observation_, reward, done, truncated, info = env.step(action)

        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_  # state = new state
    print('Final solution ','score %.2f' % score)
    rgb_array = env.render()
    # plt.imshow(rgb_array)
    # plt.axis("off")  # Remove axis for better visualization
    # plt.show()
    plt.imsave("dqn_final_sol.png", rgb_array)

    x = [i+1 for i in range(n_games)]
    filename = 'mtsp.png'
    plot_learning_curve(x, scores, eps_history, filename)

    env.close()