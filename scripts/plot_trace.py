import torch
import matplotlib.pyplot as plt
import argparse
import gym
import numpy as np
from rlkit.envs.wrappers import NormalizedBoxEnv, GoalToNormalEnv
from rlkit.envs.navigation2d.navigation2d import Navigation2d
from rlkit.envs.mujoco.ant import AntEnv
from rlkit.envs.mujoco.half_cheetah import HalfCheetahEnv
from rlkit.envs.mujoco.humanoid import HumanoidEnv
from rlkit.samplers.util import DIAYNRollout as rollout
from rlkit.samplers.rollout_functions import hierachical_rollout, hierachical_rollout2
import cv2

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

def simulate_policy2(args, filename='endobs.jpg'):
    data = torch.load(args.file, map_location='cpu')
    policy = data['evaluation/policy']
    envs = NormalizedBoxEnv(Navigation2d())
    # env = NormalizedBoxEnv(AntEnv(expose_all_qpos=True))
    env = NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=False))
    # env = NormalizedBoxEnv(HumanoidEnv(expose_all_qpos=False))
    # env = NormalizedBoxEnv(gym.make('Swimmer-v2'))
    # env = NormalizedBoxEnv(gym.make('MountainCarContinuous-v0'))
    # env = GoalToNormalEnv(gym.make('FetchReach-v1'))
    # env = GoalToNormalEnv(gym.make('FetchPush-v1'))
    skills = torch.Tensor(np.vstack([np.arange(-0.9, 0.91, 0.2), 0.8 * np.ones(10)])).transpose(1, 0)
    # skills = torch.Tensor(np.vstack([0.5 * np.ones(10), np.arange(-0.9, 0.91, 0.2)])).transpose(1, 0)
    # skills = torch.Tensor(np.vstack([-0.3 * np.ones(6), -0.3 * np.ones(6), -0.3 * np.ones(6)])).transpose(1, 0)
    # skills = torch.Tensor(np.arange(-0.9, 0.99,0.1)).reshape(-1,1)
    skills = 0.8*torch.ones([10, 12])
    #skills = torch.Tensor([[ 0.1080, -0.4160,  0.5176, -0.2920, -0.5953, -0.5421]])
    for skill in skills:
        skill = policy.stochastic_policy.skill_space.sample()
        # skill = torch.Tensor([ 0.10,  0.15, -0.82,  0.65]) #torch.Tensor([-0.2450, -0.0149, -0.8797,  0.5261]) tensor([ 0.0202,  0.0922, -0.8595,  0.6306])
        print(skill)
        policy.stochastic_policy.skill = skill
        path = DIAYNRollout(env, policy, max_path_length=args.H, render=True)
        obs = path['observations']
        # action = path['actions']
        # print(action)
        print(obs[-1])

    image = path['images'][-1]
    # cv2.imwrite(filename, image)


def simulate_policy3(args):
    data = torch.load(args.file, map_location='cpu')
    policy = data['evaluation/policy']
    envs = NormalizedBoxEnv(Navigation2d())
    # env = NormalizedBoxEnv(AntEnv(expose_all_qpos=False))
    env = NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=False))
    # env = NormalizedBoxEnv(gym.make('Swimmer-v2'))
    figure = plt.figure()
    skills = torch.Tensor(np.vstack([np.arange(-0.9, 0.91, 0.2), 0. * np.ones(10)])).transpose(1, 0)
    # skills = torch.Tensor(np.vstack([0.5 * np.ones(10), np.arange(-0.9, 0.91, 0.2)])).transpose(1, 0)
    # skills = torch.Tensor(np.vstack([-0.3 * np.ones(6), -0.3 * np.ones(6), -0.3 * np.ones(6)])).transpose(1, 0)
    # skills = torch.Tensor(np.arange(-0.9, 0.99,0.1)).reshape(-1,1)
    # skills = torch.ones([5, 3])
    skills = torch.Tensor([[ -1, -1,  1]])
    skills = torch.Tensor(np.vstack([0. * np.ones(10), 0. * np.ones(10), 0. * np.ones(10),0. * np.ones(10),0. * np.ones(10),np.arange(-0.9, 0.91, 0.2)])).transpose(1, 0)
    for skill in skills:
        skill = policy.stochastic_policy.skill_space.sample()
        print(skill)
        path = DIAYNRollout2(env, policy.stochastic_policy, skill, skill_horizon=3, scale=1, max_path_length=args.H, render=True)
        # obs = path['observations']
        # plt.plot(obs[:,0], obs[:,1], label=tuple(skill.numpy()))
        # action = path['actions']
        # print(action)
        # rewards = path['rewards']
        # print(rewards)

    plt.xlim([-4,4])
    plt.ylim([-4,4])
    # plt.legend()
    # plt.show()

def simulate_policy4(args):
    data = torch.load(args.file2, map_location='cpu')
    policy = data['evaluation/scheduler']
    data = torch.load(args.file, map_location='cpu')
    worker = data['evaluation/policy']
    env = NormalizedBoxEnv(Navigation2d())
    # env = NormalizedBoxEnv(AntEnv(expose_all_qpos=False))
    env = NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=False))
    # env = NormalizedBoxEnv(gym.make('Swimmer-v2'))
    figure = plt.figure()
    # for skill in skills:
    path = hierachical_rollout2(env, policy.stochastic_policy,
                                worker, skill_horizon=3, max_path_length=args.H, render=True)
    # obs = path['observations']
    # plt.plot(obs[:,0], obs[:,1], label=tuple(skill.numpy()))
    # action = path['actions']
    # print(action)

    # plt.xlim([-4,4])
    # plt.ylim([-4,4])
    # plt.legend()
    # plt.show()

# def simulate_policy_discrete_skill(args):
#     data = torch.load(args.file, map_location='cpu')
#     policy = data['evaluation/policy']
#     envs = NormalizedBoxEnv(Navigation2d())
#     # env = NormalizedBoxEnv(AntEnv(expose_all_qpos=True))
#     env = NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=True))
#     figure = plt.figure()
#     # skills = torch.Tensor(np.vstack([np.arange(-0.9, 0.91, 0.2), 0 * np.ones(10)])).transpose(1, 0)
#     # skills = torch.Tensor(np.vstack([0.5 * np.ones(10), np.arange(-0.9, 0.91, 0.2)])).transpose(1, 0)
#     skills = torch.Tensor(np.vstack([-0.3 * np.ones(6), -0.3 * np.ones(6), -0.3 * np.ones(6)])).transpose(1, 0)
#     # skills = torch.Tensor(np.arange(-0.9, 0.99,0.1)).reshape(-1,1)
#     for skill in range(50):
#         # skill = policy.stochastic_policy.skill_space.sample()
#         policy.skill = skill
#         path = DIAYNRollout(env, policy.stochastic_policy, skill, max_path_length=args.H, render=True)
#         obs = path['observations']
#         # plt.plot(obs[:,0], obs[:,1], label=tuple(skill.numpy()))
#         action = path['actions']
#         # print(action)
#
#     # plt.xlim([-4,4])
#     # plt.ylim([-4,4])
#     # plt.legend()
#     # plt.show()

# def simulate_policy_hierarchical(args):
#     data = torch.load(args.file, map_location='cpu')
#     scheduler = data['exploration/scheduler']
#     worker = data['exploration/worker']
#     env = NormalizedBoxEnv(Navigation2d())
#     # env = NormalizedBoxEnv(AntEnv(expose_all_qpos=False))
#     env = NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=False))
#     figure = plt.figure()
#     # skills = torch.Tensor(np.vstack([np.arange(-0.9, 0.91, 0.2), 0 * np.ones(10)])).transpose(1, 0)
#     # skills = torch.Tensor(np.vstack([0.5 * np.ones(10), np.arange(-0.9, 0.91, 0.2)])).transpose(1, 0)
#     skills = torch.Tensor(np.vstack([-0.3 * np.ones(6), -0.3 * np.ones(6), -0.3 * np.ones(6)])).transpose(1, 0)
#     # skills = torch.Tensor(np.arange(-0.9, 0.99,0.1)).reshape(-1,1)
#     for skill in range(5):
#         skill = scheduler.skill_space.sample()
#         scheduler.set_skill(skill)
#         # scheduler.skill = skill
#         # scheduler.skill_vec=np.zeros(50)
#         # scheduler.skill_vec[skill] += 1
#         path, goal_path = hierachical_rollout(env, scheduler, worker, 3, max_path_length=args.H, render=True)
#         obs = path['observations']
#         plt.plot(obs[:,0], obs[:,1])#, label=tuple(skill.numpy()))
#         action = path['actions']
#         # print(action)
#
#     plt.xlim([-1,1])
#     plt.ylim([-1,1])
#     plt.legend()
#     plt.show()

def DIAYNRollout(env, agent, max_path_length=np.inf, render=False):
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
    # o = o['observation']
    next_o = None
    path_length = 0
    if render:
        img = env.render('rgb_array')
        # img = env.render()
#        env.viewer.cam.fixedcamid = 0
#        env.viewer.cam.type = 2
        images.append(img)

    while path_length < max_path_length:
        # agent.stochastic_policy.skill = skill
        a, agent_info = agent.get_action(o)
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
        # next_o=next_o['observation']
        o = next_o
        if render:
            img = env.render('rgb_array')
            # img = env.render()
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

def DIAYNRollout2(env, agent, skill,skill_horizon=5,scale=1, max_path_length=np.inf, render=False):
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
    # agent.stochastic_policy.set_skill(skill)
    agent.set_skill(skill)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        print(path_length, a)
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
        randomskill = agent.stochastic_policy.skill_space.sample()
        # randomskill = agent.skill_space.sample()
        # randomskill = [0,0,0,0,0,0]
        if path_length % skill_horizon == 0:
            agent.stochastic_policy.set_skill(randomskill)
            # agent.stochastic_policy.set_skill(-scale*skill)
        if path_length % (2*skill_horizon) == 0:
            # agent.stochastic_policy.set_skill(randomskill)
            agent.stochastic_policy.set_skill(scale*skill)

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

def test(max_path_length=np.inf, render=True):
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
    skill_horizon = 20
    scale=0.3
    env = NormalizedBoxEnv(Navigation2d())
    env = NormalizedBoxEnv(AntEnv(expose_all_qpos=False))
    env = NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=False))
    # env = NormalizedBoxEnv(gym.make('Swimmer-v2'))
    ac=np.array([1, 1, 1, 1, 1, 1])
    a=ac
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
        # a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        # agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if max_path_length == np.inf and d:
            break
        o = next_o
        if render:
            img = env.render('rgb_array')
            images.append(img)
        # randomskill = agent.skill_space.sample()
        if path_length % skill_horizon == 0:
            # agent.set_skill(randomskill)
            a = -scale * ac
        if path_length % (2*skill_horizon) == 0:
            a = scale * ac

    # actions = np.array(actions)
    # if len(actions.shape) == 1:
    #     actions = np.expand_dims(actions, 1)
    # observations = np.array(observations)
    # if len(observations.shape) == 1:
    #     observations = np.expand_dims(observations, 1)
    #     next_o = np.array([next_o])
    # next_observations = np.vstack(
    #     (
    #         observations[1:, :],
    #         np.expand_dims(next_o, 0)
    #     )
    # )
    # return dict(
    #     observations=observations,
    #     actions=actions,
    #     rewards=np.array(rewards).reshape(-1, 1),
    #     next_observations=next_observations,
    #     terminals=np.array(terminals).reshape(-1, 1),
    #     agent_infos=agent_infos,
    #     env_infos=env_infos,
    #     images=images
    # )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('file2', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=200,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy2(args)
    # test()
