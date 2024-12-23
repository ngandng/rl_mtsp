from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from stable_baselines3.common.env_checker import check_env

# class Actions(Enum):
#     right = 0
#     up = 1
#     left = 2
#     down = 3


class MTSPEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, num_agents=2, num_tasks=5, boundary=50, render_mode=None):
        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.boundary = boundary
        self.render_mode = render_mode
        self.window_size = 512  # The size of the PyGame window

        # Observation space: agent positions and remaining tasks
        # remaning task would look like [1, 0, 1, 0, 1]
        # Total length of the flattened observation vector
        obs_length = self.num_agents * 2 + self.num_tasks
        self.observation_space = spaces.Box(
            low=0.0,  # Minimum value for all elements
            high=np.inf,  # Maximum value (you can also use a specific boundary)
            shape=(obs_length,),  # Shape of the flattened observation vector
            dtype=np.float32  # Match the dtype of _get_obs
        )

        # Action is a tupe [agent_index, next_task]
        # Flatten action space: single integer to represent [agent, task_idx] pair
        self.action_space = spaces.Discrete(self.num_agents * self.num_tasks)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all agents at zero positions
        self.agent_positions = np.zeros((self.num_agents, 2))

        # Initialize tasks at random positions
        self.task_positions = self.np_random.integers(0, self.boundary, size=(self.num_tasks, 2))
        self.tasks_remaining = np.ones(self.num_tasks, dtype=np.int8)

        if self.render_mode == "human":
            self._render_frame()

        # return observation, info
        return self._get_obs(), {}  

    def step(self, action):
        reward = 0

        # Decode the action back to [agent, task_idx]
        agent = action // self.num_tasks  # Integer division to get the agent index
        task_idx = action % self.num_tasks  # Modulus to get the task index

        if self.tasks_remaining[task_idx]:  # If the task is not completed
            distance = np.linalg.norm(self.agent_positions[agent]-self.task_positions[task_idx])
            reward = 100/distance  # Reward for completing a task
            self.tasks_remaining[task_idx] = 0  # Mark task as completed
            self.agent_positions[agent] = self.task_positions[task_idx] # update position for agent

        # Terminate if all tasks are completed
        terminated = bool(np.sum(self.tasks_remaining) == 0)

        if self.render_mode == "human":
            self._render_frame()

        # return observation, reward, terminated, False, info
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        # Flatten the observations
        agents_obs = self.agent_positions.flatten().astype(np.float32)
        tasks_obs = self.tasks_remaining.astype(np.float32)
        return np.concatenate([agents_obs, tasks_obs])
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.boundary
        )  # The size of a single grid square in pixels

        # First we draw the tasks
        for task in self.task_positions:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * task,
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agent
        for agent in self.agent_positions:
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (agent + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# env = MTSPEnv()
# check_env(env)