from gym.envs.registration import register

register(
    id="twoAgents-v0",
    entry_point="two_agents.envs:TwoAgentsEnv",
)
