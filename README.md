# MTSP with Reinforcement Learning Approaches

## Environments
This environment built base on the `gymnasium` environment. Examples are shown [on the environment creation documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/). The modified environment include:
- `MTSPEnv(num_agents,num_tasks,map_boundary)`: Multiple Traveling salesman Problem
  - State: is a concatenated vector of agent position and a binary vector of remaining tasks
  - Action space: $a_{ij} = i*(\text{task number})+j$ means assign task $i$ for agent $j$
  - Transition probabilities: this is a deterministic environment, so $P({s}'|s,a) = \{1, 0\}$

## Models 
- Deep-Q Network
The DQN code is implemented based on the examples that are shown [on youtube video](https://www.youtube.com/watch?v=wc-FxNENg9U&t=1697s&pp=ygULZHFuIHB5dG9yY2g%3D).
- PPO
The PPO algorithm is built based on the code of [nikhilbarhate99/PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch?tab=readme-ov-file)

## Results

The following plot visualizes the final solution obtained using 
- DQN

<p align="center">
  <img src="./data/img/dqn_final_sol.png" width="250"/>
</p>

- PPO
<p align="center">
  <img src="./data/img/ppo_final_sol.png" width="250"/>
</p>

## Using the code
Running the test of current training values

`python3 -m test.'algorithm'_test`

Change the `render_mode` in the test file for different visualization mode.