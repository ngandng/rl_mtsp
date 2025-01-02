import gymnasium as gym
import env
import matplotlib.pyplot as plt

from stable_baselines3 import DQN

# Environment parameters:
num_agents = 3
num_tasks = 15
boundary = 50
render_mode = "rgb_array"
# render_mode = None

random_seed = 8

env = gym.make('env/mTSP-v0', num_agents=num_agents,
                num_tasks=num_tasks,
                boundary=boundary, render_mode=render_mode)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=700000, log_interval=4)
model.save("dqn_mtsp")

print("Training is done!!!")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_mtsp")

obs, info = env.reset(seed=random_seed)
terminated = False
score = 0

while not terminated:
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)

    score += reward
    print("Step score ", score)
    # if terminated or truncated:
    #     obs, info = env.reset()

print('Final solution ','score %.2f' % score)
rgb_array = env.render()
plt.imsave("dqn_final_sol.png", rgb_array)