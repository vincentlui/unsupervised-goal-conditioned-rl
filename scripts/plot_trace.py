import torch
import matplotlib.pyplot as plt
import argparse
from rlkit.envs.wrappers import NormalizedBoxEnv
from env.navigation2d.navigation2d import Navigation2d
from rlkit.samplers.util import DIAYNRollout as rollout


def simulate_policy(args):
    data = torch.load(args.file, map_location='cpu')
    policy = data['evaluation/policy']
    env = NormalizedBoxEnv(Navigation2d())
    figure = plt.figure()
    for skill in range(policy.stochastic_policy.skill_dim):
        path = rollout(env, policy, skill, max_path_length=20)
        obs = path['observations']
        plt.plot(obs[:,0], obs[:,1])

    plt.xlim([0,1])
    plt.ylim([0,1])

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)