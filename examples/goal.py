import gym
import argparse
#from gym.envs.mujoco import HalfCheetahEnv
from env.navigation2d.navigation2d import Navigation2d

import rlkit.torch.pytorch_util as ptu
# from rlkit.torch.sac.diayn.diayn_env_replay_buffer import DIAYNEnvReplayBuffer
from rlkit.torch.sac.gcs.gcs_env_replay_buffer import GCSEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.diayn.diayn_path_collector import DIAYNMdpPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.sac.diayn.policies import SkillTanhGaussianPolicy, MakeDeterministic
# from rlkit.torch.sac.diayn.diayn import DIAYNTrainer
from rlkit.torch.sac.gcs.gcs import GCSTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.diayn.diayn_torch_online_rl_algorithm import DIAYNTorchOnlineRLAlgorithm
from rlkit.torch.sac.gcs.skill_discriminator import SkillDiscriminator
from rlkit.torch.sac.gcs.gcs_torch_online_rl_algorithm import GCSTorchOnlineRLAlgorithm
from rlkit.torch.sac.gcs.gcs_path_collector import GCSMdpPathCollector
from rlkit.torch.sac.gcs.policies import UniformSkillTanhGaussianPolicy


def experiment(variant, args):
    # expl_env = NormalizedBoxEnv(gym.make(str(args.env)))
    # eval_env = NormalizedBoxEnv(gym.make(str(args.env)))
    # expl_env = NormalizedBoxEnv(HalfCheetahEnv())
    # eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    expl_env = NormalizedBoxEnv(Navigation2d())
    expl_env.set_random_start_state(True)
    eval_env = NormalizedBoxEnv(Navigation2d())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    skill_dim = args.skill_dim
    ends_dim = expl_env.observation_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    df = SkillDiscriminator(
        input_size=obs_dim + ends_dim,
        skill_dim=skill_dim,
        hidden_sizes=[M, M],
    )
    policy = UniformSkillTanhGaussianPolicy(
        obs_dim=obs_dim + skill_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        skill_dim=skill_dim,
        low=[-1,-1],
        high=[1,1],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = DIAYNMdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_step_collector = GCSMdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = GCSEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        skill_dim,
        ends_dim
    )
    trainer = GCSTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        df=df,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = GCSTorchOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str,
                        help='environment')
    parser.add_argument('--skill_dim', type=int, default=10,
                        help='skill dimension')
    args = parser.parse_args()

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="DIAYN-goal",
        version="normal",
        layer_size=32,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=1000, #1000
            num_eval_steps_per_epoch=20,
            num_trains_per_train_loop=32,
            num_expl_steps_per_train_loop=600,
            min_num_steps_before_training=600,
            max_path_length=20,
            batch_size=128, #256
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('GOAL_' + str(args.skill_dim) + '_' + args.env, variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, args)
