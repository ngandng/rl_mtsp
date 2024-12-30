import gymnasium as gym
import env

from stable_baselines3 import DQN

# Environment parameters:
num_agents = 3
num_tasks = 15
boundary = 50
render_mode = "rgb_array"
# render_mode = None

env = gym.make('env/mTSP-v0', num_agents=num_agents,
                num_tasks=num_tasks,
                boundary=boundary, render_mode=render_mode)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_mtsp")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_mtsp")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()