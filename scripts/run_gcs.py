import torch
import matplotlib.pyplot as plt
import argparse
import gym
import numpy as np
from rlkit.envs.wrappers import NormalizedBoxEnv, GoalToNormalEnv
from rlkit.envs.navigation2d.navigation2d import Navigation2d
from rlkit.envs.mujoco.ant import AntEnv
from rlkit.envs.mujoco.half_cheetah import HalfCheetahEnv
from rlkit.samplers.util import DIAYNRollout as rollout
from rlkit.samplers.rollout_functions import hierachical_rollout, hierachical_rollout2
from rlkit.envs.fetch.reach import FetchReachEnv
from rlkit.envs.fetch.push import FetchPushEnv
import cv2


def simulate_policy(args, goal, filename='achieved.jpg'):
    data = torch.load(args.file, map_location='cpu')
    policy = data['evaluation/policy']
    df = data['trainer/df']
    # env = NormalizedBoxEnv(Navigation2d())
    # env = NormalizedBoxEnv(AntEnv(expose_all_qpos=True))
    env = NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=False))
    # env = NormalizedBoxEnv(gym.make('Swimmer-v2'))
    # env = NormalizedBoxEnv(gym.make('MountainCarContinuous-v0'))
    # env = GoalToNormalEnv(gym.make('FetchReach-v1'))
    # env = GoalToNormalEnv(FetchReachEnv())
    # env = GoalToNormalEnv(FetchPushEnv())
    path = GCSRollout2(env, policy, df, goal, max_path_length=args.H, render=True)
    # write_vid(path)
    image = path['images'][-1][250:850, 630:1230]#[150:550, 730:1130]
    print(image.shape)
    # [400:800, 200:600]
    cv2.imwrite(filename, image)
    env.close()

def GCSRollout(env, agent, df, goal, skill_horizon=200, max_path_length=np.inf, render=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    images = []

    o_env = env.reset()
    o = o_env
    if isinstance(o, dict):
        o = o_env['observation']
    if goal is None:
        goal = o_env['desired_goal']
    # goal = np.subtract(goal, o[:3])
    next_o = None
    path_length = 0
    if render:
        # img = env.render('rgb_array')
        img = env.render(mode= 'rgb_array',width=1900,height=860)
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
        if isinstance(next_o, dict):
            next_o = next_o['observation']
        o = next_o
        if render:
            # img = env.render('rgb_array')
            img = env.render(mode= 'rgb_array',width=1900,height=860)
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

def GCSRollout2(env, agent, df, goal, skill_horizon=200, max_path_length=np.inf, render=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    images = []

    o_env = env.reset()
    o = o_env
    if isinstance(o, dict):
        o = o_env['observation']
    if goal is None:
        goal = o_env['desired_goal']
    # goal = np.subtract(goal, o[:3])
    next_o = None
    path_length = 0
    if render:
        # img = env.render('rgb_array')
        img = env.render(mode= 'rgb_array',width=1900,height=860)
#        env.viewer.cam.fixedcamid = 0
#        env.viewer.cam.type = 2
        images.append(img)

    # df_input = torch.Tensor(np.concatenate([o, goal]))
    df_input = torch.Tensor(o[None])
    d_pred = df(df_input)
    d_pred_log_softmax = torch.nn.functional.log_softmax(d_pred, 1)
    _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
    print(pred_z)
    agent.stochastic_policy.skill = pred_z.squeeze().detach().numpy()

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
        if isinstance(next_o, dict):
            next_o = next_o['observation']
        o = next_o
        if render:
            # img = env.render('rgb_array')
            img = env.render(mode= 'rgb_array',width=1900,height=860)
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
                            (1900, 860))

    for i, img in enumerate(path['images']):
        # print(i)
        print(img.shape)
        video.write(img[:, :, ::-1].astype(np.uint8))
        #                cv2.imwrite("frames/diayn_bipedal_walker_hardcore.avi/%06d.png" % index, img[:,:,::-1])

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
    goal = [ 0.85640258, -1.51874102,  1.07693566,  0.02949487, -0.0969029,   0.13652097, -0.23478749,  0.13588686] #N
    # goal = [-1.33639694,  0.90047344,  1.22487349, -0.21544459, -0.13565433 , 0.33489104, -0.44654661, -0.05634518] #n
    # goal = [ 1.35731238, -1.74397825, -0.30489173, -0.07217514,  0.0074444,  -0.14156388, 0.07849566,  0.2556606 ] # v
    goal = [-8.07507281e-01, - 2.30535072e-01,   1.74625024e+00, - 3.63259100e-03, 1.61476225e-02, - 1.63148458e-02, - 1.14411647e-02 ,- 5.57299090e-06]
    goal = [-1., 0.3, 0.5, -0., 0., -0., 0., 0.,0.,0.]  # v
    goal = [-0.44509227,  -0.76124629,  0.01393204,  0,0,0,0,0]
    goal = [1.24,   0.63,   0.51]

    goal = [  3.,  -5.77272449e-01]

    goal = [  8.37394279e-02,   1.83033441e+00,  -5.24249608e-01,   4.26296879e-01,
  -4.19837966e-01,  -5.34730371e-01,  -4.57303263e-01,   5.09928320e-01,
  -2.70265454e-04,  -3.15888488e-06,  -3.21376419e-04,  -9.04722009e-06,
   5.96453730e-06,   7.84579038e-07,   3.04543559e-04,   3.06825160e-04,
   7.21252102e-05] #handstand

    goal = [ -5.77292727e-01,   3.30235611e+00,   3.97563350e-01,  -4.74817654e-02,
  -4.15206841e-01,  -2.40023158e-01,   6.77983316e-02,   1.71831904e-01,
   2.28047796e-03,  -1.28652406e-04,   2.71806759e-04,   4.35098728e-02,
   1.89267677e-01,  -1.42352542e-03,   1.81919403e-01,   2.01454011e-01,
   1.27394950e-01] #flip

  #   goal = [ -2.84508064e-01,   7.94366364e-01,   2.53811284e-01,  -2.68182510e-01,
  # -3.09767525e-01 ,  4.20495568e-01 ,  6.77008654e-01  , 2.76152462e-01,
  #  1.28293533e-06 , -1.82558913e-06 , -2.74220124e-06 ,  2.93166962e-05,
  # -1.86504883e-05 , -9.93102586e-06 , -1.66690749e-06 ,  1.36124378e-05,
  # -7.70153437e-06] #lean

  #   goal = [ -2.58902415e-01 ,  9.70717820e-02 , -4.42095927e-01  , 4.74764951e-01,
  # -4.19874266e-01 , -7.23166672e-01 ,  7.07282137e-03 , -3.50294717e-01,
  #  5.85242066e-05 ,  1.55446561e-05 ,  1.06914095e-04  , 2.46169442e-04,
  # -2.60413472e-04 ,  2.49826875e-05  , 3.97568677e-04 , -7.85007616e-04,
  #  3.30781803e-04]
    #goal = [0.6,  0.02949391] #mountain car
    simulate_policy(args, goal)
