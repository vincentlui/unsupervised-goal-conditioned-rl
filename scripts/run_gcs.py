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


def simulate_policy(args, goal):
    data = torch.load(args.file, map_location='cpu')
    policy = data['evaluation/policy']
    df = data['trainer/df']
    # env = NormalizedBoxEnv(Navigation2d())
    # env = NormalizedBoxEnv(AntEnv(expose_all_qpos=True))
    # env = NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=True))
    # env = NormalizedBoxEnv(gym.make('Swimmer-v2'))
    # env = NormalizedBoxEnv(gym.make('MountainCarContinuous-v0'))
    # env = GoalToNormalEnv(gym.make('FetchReach-v1'))
    env = GoalToNormalEnv(FetchReachEnv())
    # env = GoalToNormalEnv(FetchPushEnv())
    path = GCSRollout(env, policy, df, goal, max_path_length=args.H, render=True)
    # write_vid(path)
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
    o = o_env['observation']
    print(o_env['desired_goal'])
    goal = np.concatenate([o_env['desired_goal'], [0.] * 7])
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
    #goal = [0.6,  0.02949391] #mountain car
    simulate_policy(args, goal)
