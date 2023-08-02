from gym.envs.registration import register

register(
    id="multioAgents-v0",
    entry_point="multi_agents.envs:MultiAgentsEnv",
)
