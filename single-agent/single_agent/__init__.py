from gym.envs.registration import register

register(
    id='singleAgent-v0',
    entry_point='single_agent.envs:SingleAgentEnv',
)
