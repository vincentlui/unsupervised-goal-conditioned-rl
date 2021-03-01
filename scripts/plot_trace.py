import torch
import matplotlib.pyplot as plt
import argparse
import gym
import numpy as np
from rlkit.envs.wrappers import NormalizedBoxEnv
from envs.navigation2d.navigation2d import Navigation2d
from rlkit.envs.mujoco.ant import AntEnv
from rlkit.envs.mujoco.half_cheetah import HalfCheetahEnv
from rlkit.samplers.util import DIAYNRollout as rollout


def simulate_policy(args):
    data = torch.load(args.file, map_location='cpu')
    policy = data['evaluation/policy']
    env = NormalizedBoxEnv(Navigation2d())
    # env = NormalizedBoxEnv(AntEnv(expose_all_qpos=True))
    figure = plt.figure()
    for i in range(10):
        skill = policy.stochastic_policy.skill_space.sample()
        path = rollout(env, policy, skill, max_path_length=args.H, render=False)
        obs = path['observations']
        plt.plot(obs[:,0], obs[:,1], label=tuple(skill.numpy()))

    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend()
    plt.show()

def simulate_policy2(args):
    data = torch.load(args.file, map_location='cpu')
    policy = data['evaluation/policy']
    # envs = NormalizedBoxEnv(Navigation2d())
    # env = NormalizedBoxEnv(AntEnv(expose_all_qpos=True))
    env = NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=True))
    figure = plt.figure()
    # skills = torch.Tensor(np.vstack([np.arange(-1, 1.1, 0.2), 0.3 * np.ones(11)])).transpose(1, 0)
    # skills = torch.Tensor(np.vstack([0.8 * np.ones(6), 0.8 * np.ones(6)])).transpose(1, 0)
    skills = torch.Tensor(np.arange(-0.9, 0.99,0.2)).reshape(-1,1)
    for skill in skills:
        # skill = policy.stochastic_policy.skill_space.sample()
        policy.skill = skill
        path = DIAYNRollout(env, policy.stochastic_policy, skill, max_path_length=args.H, render=True)
        obs = path['observations']
        plt.plot(obs[:,0], obs[:,1], label=tuple(skill.numpy()))
        action = path['actions']
        # print(action)

    plt.xlim([-4,4])
    plt.ylim([-4,4])
    # plt.legend()
    plt.show()

def DIAYNRollout(env, agent, skill, max_path_length=np.inf, render=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param render:
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    images = []

    o = env.reset()
    next_o = None
    path_length = 0
    if render:
        img = env.render('rgb_array')
#        env.viewer.cam.fixedcamid = 0
#        env.viewer.cam.type = 2
        images.append(img)

    while path_length < max_path_length:
        agent.skill = skill
        a, agent_info = agent.get_action(o, return_log_prob=True)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if max_path_length == np.inf and d:
            break
        o = next_o
        if render:
            img = env.render('rgb_array')
            images.append(img)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        images=images
    )

def DIAYNRollout2(env, agent, skill, max_path_length=np.inf, render=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param render:
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    images = []

    o = env.reset()
    next_o = None
    path_length = 0
    if render:
        img = env.render('rgb_array')
#        env.viewer.cam.fixedcamid = 0
#        env.viewer.cam.type = 2
        images.append(img)

    while path_length < max_path_length:
        agent.set_skill(skill)
        a, agent_info = agent.get_action(o[1:], return_log_prob=True)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if max_path_length == np.inf and d:
            break
        o = next_o
        if render:
            img = env.render('rgb_array')
            images.append(img)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        images=images
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy2(args)