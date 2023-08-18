import random
import time
from distutils.util import strtobool

import gym
import single_agent
from gym.spaces.utils import flatdim
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import json
from modules import agent, environment


if __name__ == "__main__":

    with open(f"parameters.json", "r") as parameters_file:
        parameters = json.load(parameters_file)

    # TRY NOT TO MODIFY: seeding
    seed = parameters["env"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = bool(parameters["env"]["torch_deterministic"])

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and bool(parameters["env"]["cuda"])
        else "cpu"
    )

    # env setup
    visualize_inference = parameters["visualization"]["visualization_window"]
    envs = gym.vector.SyncVectorEnv(
        [
            environment.make_env(
                parameters["env"]["gym_id"], seed + i, i, visualize_inference
            )
            for i in range(1)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.VectorListInfo(envs)

    agent = agent.Agent(envs).to(device)

    state_dict = torch.load(
        parameters["infer"]["trained_model_path"],
        map_location=device,
    )
    agent.load_state_dict(state_dict)
    agent.eval()

    global_step = 0
    start_time = time.time()

    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_obs = torch.reshape(next_obs, (1, flatdim(envs.single_observation_space)))

    next_done = torch.zeros(1).to(device)
    done = 0

    path_len = []
    exploration_rate = []
    collisions = 0
    inferring = 1

    while inferring:

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)

        next_obs, reward, done, _, info = envs.step(action.cpu().numpy())

        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(
            device
        )
        next_obs = torch.reshape(next_obs, (1, flatdim(envs.single_observation_space)))

        if next_done.any():
            if info[0]["collision"]:
                collisions += 1

            path_len.append(info[0]["path_lenght"])
            exploration_rate.append(info[0]["explored_rate"])
            if len(path_len) == parameters["infer"]["number_of_tests"]:
                envs.close()
                inferring = 0
                break

print(f"COLLISIONS: {collisions}")
print(f"PATH LEN MEAN: {(np.mean(path_len)*20)/100}")
print(f"PATH LEN STD: {(np.std(path_len)*20)/100}")
print(f"EXPLORATION RATE MEAN: {np.mean(exploration_rate)}")
print(f"EXPLORATION STD: {np.std(exploration_rate)}")
