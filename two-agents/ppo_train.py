import random
import time

import gym
import two_agents
from gym.spaces.utils import flatdim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
from modules import agent, environment


if __name__ == "__main__":

    num_robots = 2

    with open(f"parameters.json", "r") as parameters_file:
        parameters = json.load(parameters_file)

    if parameters["train"]["finetune_agent_1"] == 1:
        run_name = (
            f"two_robots_{parameters['map']['map_name']}_finetune_{int(time.time())}"
        )
    else:
        run_name = f"two_robots_{parameters['map']['map_name']}_from_scratch_{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")

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
    visualize_train = parameters["visualization"]["visualization_window"]
    envs = gym.vector.SyncVectorEnv(
        [
            environment.make_env(
                parameters["env"]["gym_id"],
                seed + i,
                i,
                visualize_train,
                num_robots,
            )
            for i in range(parameters["train"]["num_envs"])
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.VectorListInfo(envs)

    # AGENT 1------------------------------------------------------
    # ONLY AGENT 1 IS TRAINED! YOU CAN TRAIN FROM SRATCH, OR FINETUNE AN ALREADY TRAINED NETWORK
    agent_1 = agent.Agent(envs).to(device)
    optimizer = optim.Adam(
        agent_1.parameters(), lr=parameters["train"]["learning_rate"], eps=1e-5
    )

    if parameters["train"]["finetune_agent_1"] == 1:
        state_dict_agent_1 = torch.load(
            parameters["train"]["agent_1_state_dict_path"],
            map_location=device,
        )
        agent_1.load_state_dict(state_dict_agent_1)
        # print(str(agent.state_dict()))
        # ------------------------------------------------------------------------

    # AGENT 2------------------------------------------------------------
    # AGENT 2 ONLY PERFORM INFERENCE. YOU CAN MAKE IT RANDOM, OR INITIALIZE IT WITH A TRAINED NETWORK
    agent_2 = agent.Agent(envs).to(device)
    if parameters["train"]["initialize_trained_agent_2"] == 1:
        state_dict_agent_2 = torch.load(
            parameters["train"]["agent_2_state_dict_path"],
            map_location=device,
        )
        agent_2.load_state_dict(state_dict_agent_2)
    # ---------------------------------------------------------------------

    # ALGO Logic: Storage setup
    num_envs = parameters["train"]["num_envs"]
    num_steps = parameters["train"]["num_steps"]
    obs = torch.zeros(
        (num_steps, num_envs)
        + (int(flatdim(envs.single_observation_space) / num_robots),)
    ).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(
        device
    )
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // parameters["train"]["num_minibatches"])

    robot_obs, _ = envs.reset()
    next_obs = robot_obs[:, 0 : int(robot_obs.shape[1] / num_robots)]
    next_obs = torch.Tensor(next_obs).to(device)
    next_obs = torch.reshape(
        next_obs,
        (num_envs, int(flatdim(envs.single_observation_space) / num_robots)),
    )

    next_obs_2 = robot_obs[
        :, int(robot_obs.shape[1] / num_robots) : int(robot_obs.shape[1])
    ]
    next_obs_2 = torch.Tensor(next_obs_2).to(device)
    next_obs_2 = torch.reshape(
        next_obs_2,
        (num_envs, int(flatdim(envs.single_observation_space) / num_robots)),
    )

    next_done = torch.zeros(num_envs).to(device)
    num_updates = parameters["train"]["total_train_timesteps"] // batch_size
    n_episodes = 0

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if parameters["train"]["anneal_lr"] == 1:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * parameters["train"]["learning_rate"]
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):

            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                # GET AGENT 1 ACTION
                action, logprob, _, value = agent_1.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # GET AGENT 2 ACTION
            action_2, logprob_2, _, value_2 = agent_2.get_action_and_value(next_obs_2)
            pass_action = np.zeros(shape=(num_envs, num_robots))
            for i in range(num_envs):
                pass_action[i][0] = action[i]
                for j in range(num_robots - 1):
                    pass_action[i][j + 1] = action_2[i]  # second robot action

            pass_action = torch.from_numpy(pass_action)

            robot_obs, reward, done, _, info = envs.step(pass_action.cpu().numpy())
            next_obs = robot_obs[:, 0 : int(robot_obs.shape[1] / num_robots)]
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)
            next_obs = torch.reshape(
                next_obs,
                (
                    num_envs,
                    int(flatdim(envs.single_observation_space) / num_robots),
                ),
            )

            next_obs_2 = robot_obs[
                :, int(robot_obs.shape[1] / num_robots) : int(robot_obs.shape[1])
            ]
            # print(next_obs.shape)
            next_obs_2 = torch.Tensor(next_obs_2).to(device)
            next_obs_2 = torch.reshape(
                next_obs_2,
                (
                    num_envs,
                    int(flatdim(envs.single_observation_space) / num_robots),
                ),
            )

            for item in info:
                if type(item) == dict:
                    if "episode" in item.keys():
                        writer.add_scalar(
                            "charts/episodic_return", item["episode"]["r"], n_episodes
                        )
                        writer.add_scalar(
                            "charts/episodic_length", item["episode"]["l"], n_episodes
                        )
                        writer.add_scalar(
                            "charts/explored_rate",
                            item["robot1"]["explored_rate"],
                            n_episodes,
                        )
                        writer.add_scalar(
                            "charts/path_lenght",
                            item["robot1"]["path_lenght"],
                            n_episodes,
                        )
                        n_episodes += 1
                        break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent_1.get_value(next_obs).reshape(1, -1)
            # If gae == 1, use General Advantage Estimation for advantage computation
            # Ctes:
            #   gamma - the discount factor gamma
            #   gae_lambda - the lambda for the general advantage estimation
            gamma = parameters["train"]["gamma"]
            if parameters["train"]["gae"] == 1:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + gamma
                        * parameters["train"]["gae_lambda"]
                        * nextnonterminal
                        * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape(
            (-1,) + (int(flatdim(envs.single_observation_space) / num_robots),)
        )
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        clip_coef = parameters["train"][
            "clip_coef"
        ]  # the surrogate clipping coefficient
        entropy_coef = parameters["train"]["entropy_coef"]  # coefficient of the entropy
        value_function_coef = parameters["train"][
            "value_function_coef"
        ]  # value function coefficient

        for epoch in range(parameters["train"]["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent_1.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if (
                    parameters["train"]["norm_adv"] == 1
                ):  # Toggles advantages normalization
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if (
                    parameters["train"]["clip_vloss"] == 1
                ):  # "Toggles whether or not to use a clipped loss for the value function, as per the paper."
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss - entropy_coef * entropy_loss + v_loss * value_function_coef
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent_1.parameters(), parameters["train"]["max_grad_norm"]
                )  # max_grad_norm = the maximum norm for the gradient clipping
                optimizer.step()

            target_kl = parameters["train"][
                "target_kl"
            ]  # the target KL divergence threshold
            if target_kl != "None":
                if approx_kl > target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    # print(str(agent.state_dict()))
    torch.save(agent_1.state_dict(), f"trained_models/{run_name}_state_dict")
    print("ACABOU!")
    envs.close()
    writer.close()
