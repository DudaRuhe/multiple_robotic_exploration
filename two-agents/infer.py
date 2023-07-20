import random
import time
from distutils.util import strtobool

import gym
import two_agents
from gym.spaces.utils import flatdim
import numpy as np
import torch
import json
from modules import agent, environment

if __name__ == "__main__":

    with open(f"parameters.json", "r") as parameters_file:
        parameters = json.load(parameters_file)

    num_robots = parameters["env"]["num_robots"]
    
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
                parameters["env"]["gym_id"], seed + i, i, visualize_inference, num_robots
            )
            for i in range(1)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.VectorListInfo(envs)

    # AGENT 1------------------------------------------------------
    agent_1 = agent.Agent(envs).to(device)
    print((agent_1))
    state_dict_agent_1 = torch.load(
        parameters["infer"]["agent_1_model_path"],
        map_location=device,
    )
    agent_1.load_state_dict(state_dict_agent_1)
    agent_1.eval()
    # ------------------------------------------------------------------------

    # AGENT 2------------------------------------------------------------
    agent_2 = agent.Agent(envs).to(device)
    print(np.shape(agent_2))
    state_dict_agent_2 = torch.load(
        parameters["infer"]["agent_2_model_path"],
        map_location=device,
    )
    agent_2.load_state_dict(state_dict_agent_2)
    agent_2.eval()
    # ---------------------------------------------------------------------

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    robot_obs, _ = envs.reset()
    next_obs = robot_obs[:, 0 : int(robot_obs.shape[1] / num_robots)]
    next_obs = torch.Tensor(next_obs).to(device)
    next_obs = torch.reshape(
        next_obs,
        (1, int(flatdim(envs.single_observation_space) / num_robots)),
    )

    next_obs_2 = robot_obs[
        :, int(robot_obs.shape[1] / num_robots) : int(robot_obs.shape[1])
    ]
    # print(next_obs.shape)
    next_obs_2 = torch.Tensor(next_obs_2).to(device)
    next_obs_2 = torch.reshape(
        next_obs_2,
        (1, int(flatdim(envs.single_observation_space) / num_robots)),
    )

    next_done = torch.zeros(1).to(device)
    done = 0
    inferring = 1
    path_len_robots = {
        "robot"+str(i): []
        for i in range(num_robots)
    }
    collisions_robots = {
        "robot"+str(i): 0
        for i in range(num_robots)
    }
    exploration_rate = []
    while inferring:
        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent_1.get_action_and_value(next_obs)

        action_2, logprob_2, _, value_2 = agent_2.get_action_and_value(next_obs_2)

        pass_action = np.zeros(shape=(1, num_robots))
        for i in range(1):
            pass_action[i][0] = action[i]
            for j in range(num_robots - 1):
                pass_action[i][j + 1] = action_2[i]  # second robot action

        pass_action = torch.from_numpy(pass_action)

        robot_obs, reward, done, _, info = envs.step(pass_action.cpu().numpy())

        # get first robot observation
        next_obs = robot_obs[:, 0 : int(robot_obs.shape[1] / num_robots)]
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(
            device
        )
        next_obs = torch.reshape(
            next_obs,
            (
                1,
                int(flatdim(envs.single_observation_space) / num_robots),
            ),
        )

        # get second robot observation
        next_obs_2 = robot_obs[
            :, int(robot_obs.shape[1] / num_robots) : int(robot_obs.shape[1])
        ]
        # print(next_obs.shape)
        next_obs_2 = torch.Tensor(next_obs_2).to(device)
        next_obs_2 = torch.reshape(
            next_obs_2,
            (
                1,
                int(flatdim(envs.single_observation_space) / num_robots),
            ),
        )

        if next_done.any():

            for i in range(num_robots):
                robot_name = "robot"+str(i)
                if info[0][robot_name]["collision"]:
                    collisions_robots[robot_name] += 1
                path_len_robots[robot_name].append(info[0][robot_name]["path_lenght"])

            exploration_rate.append(info[0]["robot0"]["explored_rate"])
            
            if len(path_len_robots["robot0"]) == 50:
                envs.close()
                inferring = 0

for i in range(num_robots):
    robot_name = "robot" + str(i)
    print(f"\n ROBOT {i+1}")
    print(f"COLLISIONS ROBOT {i+1}: {collisions_robots[robot_name]}")
    print(f"PATH LEN MEAN: {(np.mean(path_len_robots[robot_name])*20)/100}")
    print(f"PATH LEN STD: {(np.std(path_len_robots[robot_name])*20)/100}")
print("\n")
print(f"EXPLORATION RATE MEAN: {np.mean(exploration_rate)}")
print(f"EXPLORATION STD: {np.std(exploration_rate)}")
