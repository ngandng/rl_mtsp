from gymnasium.envs.registration import register

register(
    id="env/mTSP-v0",
    entry_point="env.envs:MTSPEnv",
)