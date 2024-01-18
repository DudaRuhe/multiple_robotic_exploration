import gym


def make_env(gym_id, seed, idx, visualization, num_robots):
    def thunk():
        if visualization == "True":
            if idx == 0:
                env = gym.make(
                    gym_id, idx=idx, num_robots=num_robots, render_mode="human"
                )
        else:
                
            env = gym.make(gym_id, idx=idx, num_robots=num_robots)
        env = gym.wrappers.FlattenObservation(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
