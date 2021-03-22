from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger
from rlkit.envs.navigation2d.navigation2d import Navigation2d
from rlkit.envs.mujoco.ant import AntEnv
from rlkit.envs.mujoco.half_cheetah import HalfCheetahEnv
import gym

filename = str(uuid.uuid4())
import torch
from rlkit.envs.wrappers import NormalizedBoxEnv

def simulate_policy(args):
    data = torch.load(str(args.file))
    #data = joblib.load(str(args.file))
    policy = data['evaluation/policy']
    env, _ = get_env('Half-cheetah')
    #env = data['evaluation/env']
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=True,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()
def get_env(name):
    if name == 'test':
        expl_env, eval_env = Navigation2d(), Navigation2d()
        # expl_env.set_random_start_state(True)
        # eval_env.set_random_start_state(True)
        return NormalizedBoxEnv(expl_env), NormalizedBoxEnv(eval_env)
    elif name == 'Ant':
        return NormalizedBoxEnv(AntEnv(expose_all_qpos=False)), NormalizedBoxEnv(AntEnv(expose_all_qpos=False))
    elif name == 'Half-cheetah':
        return NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=False)), NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=False))

    return NormalizedBoxEnv(gym.make(name)), NormalizedBoxEnv(gym.make(name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
