import torch.nn as nn
import numpy as np
import json
from gym.spaces.utils import flatdim
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):

        with open(f"parameters.json", "r") as parameters_file:
            parameters = json.load(parameters_file)

        num_robots = parameters["env"]["num_robots"]
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(
                        (int(flatdim(envs.single_observation_space) / num_robots),)
                    ).prod(),
                    64,
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(
                        (int(flatdim(envs.single_observation_space) / num_robots),)
                    ).prod(),
                    64,
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
