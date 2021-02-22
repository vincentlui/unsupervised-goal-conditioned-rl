import torch
import matplotlib.pyplot as plt
import argparse
import gym
import numpy as np
from rlkit.envs.wrappers import NormalizedBoxEnv
from envs.navigation2d.navigation2d import Navigation2d
from envs.mujoco.ant import AntEnv
from rlkit.samplers.util import DIAYNRollout as rollout


def simulate_policy(args):
    data = torch.load(args.file, map_location='cpu')
    policy = data['evaluation/policy']
    # envs = NormalizedBoxEnv(Navigation2d())
    env = NormalizedBoxEnv(AntEnv(expose_all_qpos=True))
    figure = plt.figure()
    for _ in range(3):
        skill = policy.stochastic_policy.skill_space.sample()
        path = rollout(env, policy, skill, max_path_length=args.H, render=True)
        obs = path['observations']
        plt.plot(obs[:,0], obs[:,1], label=tuple(skill.numpy()))

    plt.xlim([0,2])
    plt.ylim([0,2])
    plt.legend()
    plt.show()

def simulate_policy2(args):
    data = torch.load(args.file, map_location='cpu')
    policy = data['evaluation/policy']
    # envs = NormalizedBoxEnv(Navigation2d())
    env = NormalizedBoxEnv(AntEnv())
    figure = plt.figure()
    for _ in range(3):
        skill = policy.stochastic_policy.skill_space.sample()
        path = rollout(env, policy, skill, max_path_length=2, render=True)
        obs = np.array([p['coordinate'] for p in path['env_infos']])
        plt.plot(obs[:,0], obs[:,1], label=tuple(skill.numpy()))
        print(obs)

    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy2(args)