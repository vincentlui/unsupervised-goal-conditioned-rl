import gym
import argparse
# from gym.envs.mujoco import HalfCheetahEnv
from rlkit.envs.navigation2d.navigation2d import Navigation2d
from rlkit.envs.mujoco.ant import AntEnv
from rlkit.envs.mujoco.half_cheetah import HalfCheetahEnv

import torch
import rlkit.torch.pytorch_util as ptu
# from rlkit.torch.sac.diayn.diayn_env_replay_buffer import DIAYNEnvReplayBuffer
# from rlkit.torch.sac.gcs.gcs_env_replay_buffer import GCSEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.diayn.diayn_path_collector import DIAYNMdpPathCollector
from rlkit.torch.sac.diayn.policies import SkillTanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.hsd.hsd import HSDSchedulerTrainer, HSDWorkerTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.hsd.hsd_torch_rl_algorithm import HSDTorchRLAlgorithm
from rlkit.torch.sac.hsd.hsd_path_collector import HSDMdpPathCollector
from rlkit.torch.sac.hsd.policies import UniformSkillTanhGaussianPolicy, SkillTanhGaussianPolicy
from rlkit.torch.sac.gcs.networks import FlattenBNMlp
# from rlkit.torch.sac.gcs.gcs_goal_buffer import GCSGoalBuffer, GCSSGoalPathBuffer
from rlkit.torch.sac.hsd.skill_discriminator import DiscreteSkillDiscriminator, SkillDiscriminator
from rlkit.torch.sac.hsd.hsd_env_replay_buffer import HSDEnvReplayBuffer, HSDSchedulerEnvReplayBuffer
from rlkit.torch.sac.hsd.skill_dynamics import SkillDynamics


def experiment(variant, args):
    expl_env, eval_env = get_env(str(args.env))
    obs_dim = expl_env.observation_space.low.size - (len(variant['exclude_obs_ind']) if variant['exclude_obs_ind'] else 0)
    action_dim = eval_env.action_space.low.size
    y_dim = variant['y_dim']
    z_dim = variant['z_dim']
    goal_dim = variant['goal_dim']

    M = variant['layer_size']
    qf1_z = FlattenBNMlp(
        input_size=obs_dim + action_dim + z_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    qf2_z = FlattenBNMlp(
        input_size=obs_dim + action_dim + z_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    target_qf1_z = FlattenBNMlp(
        input_size=obs_dim + action_dim + z_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    target_qf2_z = FlattenBNMlp(
        input_size=obs_dim + action_dim + z_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    discriminator_z = SkillDiscriminator(
        input_size=obs_dim + obs_dim,
        skill_dim=z_dim,
        hidden_sizes=[M, M],
        output_activation=torch.tanh,
        num_components=1,
        batch_norm=variant['batch_norm'],
        std=[1.] * z_dim
    )
    worker = UniformSkillTanhGaussianPolicy(
        obs_dim=obs_dim + z_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        skill_dim=z_dim,
        low=[-1] * z_dim,
        high=[1] * z_dim,
    )
    qf1_y = FlattenBNMlp(
        input_size=obs_dim + z_dim + y_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    qf2_y = FlattenBNMlp(
        input_size=obs_dim + z_dim + y_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    target_qf1_y = FlattenBNMlp(
        input_size=obs_dim + z_dim + y_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    target_qf2_y = FlattenBNMlp(
        input_size=obs_dim + z_dim + y_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    discriminator_y = DiscreteSkillDiscriminator(
        input_size=obs_dim,
        skill_dim=y_dim,
        hidden_sizes=[M, M],
    )
    dynamics = SkillDynamics(
        hidden_sizes=[M,M],
        input_size=obs_dim+y_dim,
        state_dim=obs_dim,
        num_components=1,
        std=[1.] * obs_dim,
        batch_norm=True
    )
    # scheduler = SkillTanhGaussianPolicy(
    #     obs_dim=obs_dim + y_dim,
    #     action_dim=z_dim,
    #     hidden_sizes=[M, M],
    #     skill_dim=y_dim,
    # )
    scheduler = UniformSkillTanhGaussianPolicy(
        obs_dim=obs_dim + y_dim,
        action_dim=z_dim,
        hidden_sizes=[M, M],
        skill_dim=y_dim,
        low=[-1] * y_dim,
        high=[1] * y_dim,
    )

    eval_policy = MakeDeterministic(scheduler)
    eval_path_collector = DIAYNMdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_step_collector = HSDMdpPathCollector(
        expl_env,
        scheduler,
        worker,
        exclude_obs_ind=variant['exclude_obs_ind'],
        goal_ind=variant['goal_ind'],
        skill_horizon=variant['skill_horizon'],
        # render=True
    )
    replay_buffer = HSDEnvReplayBuffer(
        variant['worker_replay_buffer_size'],
        expl_env,
        z_dim,
        goal_dim,
    )
    scheduler_replay_buffer = HSDSchedulerEnvReplayBuffer(
        variant['scheduler_replay_buffer_size'],
        expl_env,
        variant['skill_horizon'],
        y_dim,
        z_dim,
        goal_dim,
    )
    scheduler_trainer = HSDSchedulerTrainer(
        env=eval_env,
        scheduler=scheduler,
        qf1=qf1_y,
        qf2=qf2_y,
        discriminator=discriminator_y,
        skill_dynamics=dynamics,
        target_qf1=target_qf1_y,
        target_qf2=target_qf2_y,
        exclude_obs_ind=variant['exclude_obs_ind'],
        **variant['trainer_kwargs']
    )
    worker_trainer = HSDWorkerTrainer(
        env=eval_env,
        worker=worker,
        qf1=qf1_z,
        qf2=qf2_z,
        discriminator=discriminator_z,
        target_qf1=target_qf1_z,
        target_qf2=target_qf2_z,
        exclude_obs_ind=variant['exclude_obs_ind'],
        **variant['trainer_worker_kwargs']
    )
    algorithm = HSDTorchRLAlgorithm(
        trainer=scheduler_trainer,
        trainer_worker=worker_trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        scheduler_replay_buffer=scheduler_replay_buffer,
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
        return NormalizedBoxEnv(AntEnv(expose_all_qpos=False)), NormalizedBoxEnv(AntEnv(expose_all_qpos=False))
    elif name == 'Half-cheetah':
        return NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=False)), NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=False))

    return NormalizedBoxEnv(gym.make('name')), NormalizedBoxEnv(gym.make('name'))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str,
                        help='environment')
    args = parser.parse_args()

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="HSD",
        version="normal",
        layer_size=128,
        worker_replay_buffer_size=int(1E6),
        scheduler_replay_buffer_size=int(1E4),
        exclude_obs_ind=None,
        goal_ind=None,#[0, 1],
        skill_horizon=3,
        batch_norm=False,
        y_dim=3,
        z_dim=3,
        goal_dim=17,
        algorithm_kwargs=dict(
            num_epochs=3000, #1000
            num_eval_steps_per_epoch=0,
            num_trains_per_train_loop=64,
            num_expl_steps_per_train_loop=1005,
            num_trains_discriminator_per_train_loop=8,
            min_num_steps_before_training=0,
            max_path_length=201,
            scheduler_batch_size=128,
            worker_batch_size=128, #256
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
            use_automatic_entropy_tuning=True,
        ),
        trainer_worker_kwargs=dict(
            discount=0.5,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            df_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=False,
        ),
    )
    setup_logger('HSD_' + '_' + args.env, variant=variant,snapshot_mode="gap_and_last",
            snapshot_gap=100,)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, args)
