import gym
import argparse
# from gym.envs.mujoco import HalfCheetahEnv
from rlkit.envs.navigation2d.navigation2d import Navigation2d
from rlkit.envs.mujoco.ant import AntEnv
from rlkit.envs.mujoco.half_cheetah import HalfCheetahEnv

import torch
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
from rlkit.torch.sac.gcs.gcs2 import GCSTrainer2
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.diayn.diayn_torch_online_rl_algorithm import DIAYNTorchOnlineRLAlgorithm
from rlkit.torch.sac.gcs.skill_discriminator import SkillDiscriminator
from rlkit.torch.sac.gcs.gcs_torch_online_rl_algorithm import GCSTorchOnlineRLAlgorithm
from rlkit.torch.sac.gcs.gcs_torch_rl_algorithm import GCSTorchRLAlgorithm
from rlkit.torch.sac.gcs.gcs_path_collector import GCSMdpPathCollector, GCSMdpPathCollector2
from rlkit.torch.sac.gcs.policies import UniformSkillTanhGaussianPolicy
from rlkit.torch.sac.gcs.skill_dynamics import SkillDynamics
from rlkit.torch.sac.gcs.networks import FlattenBNMlp
from rlkit.torch.sac.gcs.gcs_goal_buffer import GCSGoalBuffer


def experiment(variant, args):
    expl_env, eval_env = get_env(str(args.env))
    obs_dim = expl_env.observation_space.low.size - (len(variant['exclude_obs_ind']) if variant['exclude_obs_ind'] else 0)
    action_dim = eval_env.action_space.low.size
    skill_dim = args.skill_dim
    # ends_dim = expl_env.observation_space.low.size
    ends_dim = args.ends_dim

    M = variant['layer_size']
    qf1 = FlattenBNMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    qf2 = FlattenBNMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    target_qf1 = FlattenBNMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    target_qf2 = FlattenBNMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    df = SkillDiscriminator(
        input_size=obs_dim + ends_dim,
        skill_dim=skill_dim,
        hidden_sizes=[M, M],
        output_activation=torch.tanh,
        num_components=1,
        batch_norm=variant['batch_norm'],
        # std=[0.1, 0.1]
    )
    # df2 = SkillDiscriminator(
    #     input_size=obs_dim + ends_dim,
    #     skill_dim=skill_dim,
    #     hidden_sizes=[M, M],
    #     output_activation=torch.tanh,
    #     num_components=1,
    #     batch_norm=variant['batch_norm'],
    #     # std=[0.1, 0.1]
    # )
    # skill_dynamics = SkillDynamics(
    #     input_size=obs_dim + skill_dim,
    #     state_dim=ends_dim,
    #     hidden_sizes=[M, M],
    #     num_components=4,
    #     # std=[0.1, 0.1]
    # )
    goal_buffer = GCSGoalBuffer(
        variant['goal_buffer_size'],
        goal_dim=ends_dim,
    )
    policy = UniformSkillTanhGaussianPolicy(
        obs_dim=obs_dim + skill_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        skill_dim=skill_dim,
        low=[-1] * skill_dim,
        high=[1] * skill_dim,
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = DIAYNMdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_step_collector = GCSMdpPathCollector2(
        expl_env,
        policy,
        goal_buffer,
        skill_discriminator=df,
        exclude_obs_ind=variant['exclude_obs_ind'],
        goal_ind=variant['goal_ind'],
        skill_horizon=variant['skill_horizon'],
        # render=True
    )
    replay_buffer = GCSEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        skill_dim,
        ends_dim,
    )
    trainer = GCSTrainer2(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        df=df,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        exclude_obs_ind=variant['exclude_obs_ind'],
        **variant['trainer_kwargs']
    )
    algorithm = GCSTorchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        goal_buffer=goal_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def get_env(name):
    if name == 'test':
        expl_env, eval_env = Navigation2d(), Navigation2d()
        # expl_env.set_random_start_state(True)
        # eval_env.set_random_start_state(True)
        return NormalizedBoxEnv(expl_env), NormalizedBoxEnv(eval_env)
    elif name == 'Ant':
        return NormalizedBoxEnv(AntEnv(expose_all_qpos=True)), NormalizedBoxEnv(AntEnv(expose_all_qpos=True))
    elif name == 'Half-cheetah':
        return NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=True)), NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=True))

    return NormalizedBoxEnv(gym.make('name')), NormalizedBoxEnv(gym.make('name'))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str,
                        help='environment')
    parser.add_argument('--skill_dim', type=int, default=2,
                        help='skill dimension')
    parser.add_argument('--ends_dim', type=int, default=2,
                        help='end_state dimension')
    args = parser.parse_args()

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="GCS2",
        version="normal",
        layer_size=128,
        replay_buffer_size=int(1E6),
        goal_buffer_size=int(1E4),
        exclude_obs_ind=None,#[0],
        goal_ind=None,#[0],
        skill_horizon=200,
        batch_norm=False,
        algorithm_kwargs=dict(
            num_epochs=3000, #1000
            num_eval_steps_per_epoch=0,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=2000,
            num_trains_discriminator_per_train_loop=0,
            min_num_steps_before_training=0,
            max_path_length=200,
            batch_size=128, #256
        )
        ,
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            df_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=False,
        ),
    )
    setup_logger('GCS_' + str(args.skill_dim) + '_' + args.env, variant=variant,snapshot_mode="gap_and_last",
            snapshot_gap=100,)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, args)
