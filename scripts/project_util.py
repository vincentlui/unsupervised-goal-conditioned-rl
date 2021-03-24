import torch
import matplotlib.pyplot as plt
import argparse
import gym
import numpy as np
from rlkit.envs.wrappers import NormalizedBoxEnv, GoalToNormalEnv
from rlkit.envs.mujoco_image_env import ImageMujocoEnv
from rlkit.envs.navigation2d.navigation2d import Navigation2d
from rlkit.envs.mujoco.ant import AntEnv
from rlkit.envs.mujoco.half_cheetah import HalfCheetahEnv
from rlkit.envs.mujoco.swimmer import SwimmerEnv
from rlkit.envs.fetch.reach import FetchReachEnv
from rlkit.samplers.util import DIAYNRollout as rollout
from rlkit.samplers.rollout_functions import hierachical_rollout, hierachical_rollout2
import cv2

def simulate_policy_goal(args, goal, filename='goal_end.jpg'):
    data = torch.load(args.file, map_location='cpu')
    policy = data['evaluation/policy']
    df = data['trainer/df']
    # env = NormalizedBoxEnv(Navigation2d())
    # env = NormalizedBoxEnv(AntEnv(expose_all_qpos=True))
    # env = NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=False))
    env = NormalizedBoxEnv(SwimmerEnv())
    # env = NormalizedBoxEnv(gym.make('MountainCarContinuous-v0'))
    # env = GoalToNormalEnv(gym.make('FetchReach-v1'))
    env = GoalToNormalEnv(FetchReachEnv())
    path = GCSRollout(env, policy, df, goal, max_path_length=args.H, render=True)
    image = path['images'][-1]
    cv2.imwrite(filename, image)
    # write_vid(path)
    env.close()

def simulate_policy2(args, filename='endobs.jpg'):
    data = torch.load(args.file, map_location='cpu')
    policy = data['evaluation/policy']
    envs = NormalizedBoxEnv(Navigation2d())
    # env = NormalizedBoxEnv(AntEnv(expose_all_qpos=True))
    # env = NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=False))
    env = NormalizedBoxEnv(gym.make('Swimmer-v2'))
    env = NormalizedBoxEnv(SwimmerEnv())
    # env = NormalizedBoxEnv(gym.make('MountainCarContinuous-v0'))
    # env = GoalToNormalEnv(gym.make('FetchReach-v1'))
    env = GoalToNormalEnv(FetchReachEnv())
    skills = torch.Tensor(np.vstack([np.arange(-0.9, 0.91, 0.2), 0.8 * np.ones(10)])).transpose(1, 0)
    # skills = torch.Tensor(np.vstack([0.5 * np.ones(10), np.arange(-0.9, 0.91, 0.2)])).transpose(1, 0)
    # skills = torch.Tensor(np.vstack([-0.3 * np.ones(6), -0.3 * np.ones(6), -0.3 * np.ones(6)])).transpose(1, 0)
    # skills = torch.Tensor(np.arange(-0.9, 0.99,0.1)).reshape(-1,1)
    skills = 0.8*torch.ones([1, 2])
    for skill in skills:
        skill = policy.stochastic_policy.skill_space.sample()
        print(skill)
        policy.stochastic_policy.skill = skill
        path = DIAYNRollout(env, policy, max_path_length=args.H, render=True)
        obs = path['observations']
        # action = path['actions']
        # print(action)
        print(obs[-1])

    image = path['images'][-1]
    # write_vid(path)
    cv2.imwrite(filename, image)

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
        img = env.render(mode='rgb_array')
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
            img = env.render(mode='rgb_array')
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

def GCSRollout(env, agent, df, goal, skill_horizon=200, max_path_length=np.inf, render=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    images = []

    o_env = env.reset()
    # o=o_env
    o = o_env['observation']
    print(o_env['desired_goal'])
    goal = o_env['desired_goal']
    # goal = np.subtract(goal, o)
    next_o = None
    path_length = 0
    if render:
        # img = env.render('rgb_array')
        img = env.render(mode= 'rgb_array',width=1840,height=860)
#        env.viewer.cam.fixedcamid = 0
#        env.viewer.cam.type = 2
        images.append(img)

    df_input = torch.Tensor(np.concatenate([o, goal]))
    skill = df(df_input).mean
    print(skill)
    agent.stochastic_policy.skill = skill.detach().numpy()

    while path_length < max_path_length:
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
        next_o = next_o['observation']
        o = next_o
        if render:
            # img = env.render('rgb_array')
            img = env.render(mode= 'rgb_array',width=1840,height=860)
            images.append(img)
        if path_length % skill_horizon == 0:
            df_input = torch.Tensor(np.concatenate([o, goal]))
            skill = df(df_input).mean
            agent.stochastic_policy.skill = skill.detach().numpy()

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

def write_vid(path, filename='gcs.avi'):
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                            (1200, 800))

    for i, img in enumerate(path['images']):
        print(i)
        print(img.shape)
        video.write(img[:, :, ::-1].astype(np.uint8))
        # #                cv2.imwrite("frames/diayn_bipedal_walker_hardcore.avi/%06d.png" % index, img[:,:,::-1])

    video.release()
    print("wrote video")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=200,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    goal = [-0.82459948,  0.80886438,  0.36210404, -0.25644148, -0.36181184,  0.42180618,
 -0.96572935,  1.19107613]
    simulate_policy_goal(args, goal)
    # simulate_policy2(args)
    # test()
