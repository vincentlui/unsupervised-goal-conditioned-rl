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
from rlkit.torch.sac.hrls.hrls import HRLSchedulerTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.hrls.hrls_torch_rl_algorithm import HRLSTorchRLAlgorithm
from rlkit.torch.sac.hrls.hrls_path_collector import HRLSMdpPathCollector
from rlkit.torch.sac.gcs.networks import FlattenBNMlp
from rlkit.torch.sac.hrls.hrls_env_replay_buffer import HRLSchedulerEnvReplayBuffer
from rlkit.torch.sac.hsd.skill_dynamics import SkillDynamics
from rlkit.torch.sac.hrls.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer

def experiment(variant, args):
    expl_env, eval_env = get_env(str(args.env))
    obs_dim = expl_env.observation_space.low.size - (len(variant['exclude_obs_ind']) if variant['exclude_obs_ind'] else 0)
    action_dim = eval_env.action_space.low.size
    goal_dim = variant['goal_dim']
    data = torch.load(args.file, map_location='cpu')
    worker = data['evaluation/policy']
    skill_dim = worker.stochastic_policy.skill_dim
    # worker = worker.stochastic_policy

    M = variant['layer_size']
    # qf1_z = FlattenBNMlp(
    #     input_size=obs_dim + action_dim + z_dim,
    #     output_size=1,
    #     hidden_sizes=[M, M],
    #     batch_norm=variant['batch_norm'],
    # )
    # qf2_z = FlattenBNMlp(
    #     input_size=obs_dim + action_dim + z_dim,
    #     output_size=1,
    #     hidden_sizes=[M, M],
    #     batch_norm=variant['batch_norm'],
    # )
    # target_qf1_z = FlattenBNMlp(
    #     input_size=obs_dim + action_dim + z_dim,
    #     output_size=1,
    #     hidden_sizes=[M, M],
    #     batch_norm=variant['batch_norm'],
    # )
    # target_qf2_z = FlattenBNMlp(
    #     input_size=obs_dim + action_dim + z_dim,
    #     output_size=1,
    #     hidden_sizes=[M, M],
    #     batch_norm=variant['batch_norm'],
    # )
    # discriminator_z = SkillDiscriminator(
    #     input_size=obs_dim + obs_dim,
    #     skill_dim=z_dim,
    #     hidden_sizes=[M, M],
    #     output_activation=torch.tanh,
    #     num_components=1,
    #     batch_norm=variant['batch_norm'],
    #     std=[1.] * z_dim
    # )
    # worker = UniformSkillTanhGaussianPolicy(
    #     obs_dim=obs_dim + z_dim,
    #     action_dim=action_dim,
    #     hidden_sizes=[M, M],
    #     skill_dim=z_dim,
    #     low=[-1] * z_dim,
    #     high=[1] * z_dim,
    # )
    qf1_scheduler = FlattenBNMlp(
        input_size=obs_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    qf2_scheduler = FlattenBNMlp(
        input_size=obs_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    target_qf1_scheduler = FlattenBNMlp(
        input_size=obs_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    target_qf2_scheduler = FlattenBNMlp(
        input_size=obs_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
        batch_norm=variant['batch_norm'],
    )
    # discriminator_y = DiscreteSkillDiscriminator(
    #     input_size=obs_dim,
    #     skill_dim=y_dim,
    #     hidden_sizes=[M, M],
    # )
    # scheduler = SkillTanhGaussianPolicy(
    #     obs_dim=obs_dim + y_dim,
    #     action_dim=z_dim,
    #     hidden_sizes=[M, M],
    #     skill_dim=y_dim,
    # )
    scheduler = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=skill_dim,
        hidden_sizes=[M, M]
    )

    eval_policy = MakeDeterministic(scheduler)
    eval_path_collector = HRLSMdpPathCollector(
        eval_env,
        eval_policy,
        worker,
        exclude_obs_ind=variant['exclude_obs_ind'],
        goal_ind=variant['goal_ind'],
        skill_horizon=variant['skill_horizon'],
        # render=True,
    )
    expl_step_collector = HRLSMdpPathCollector(
        expl_env,
        scheduler,
        worker,
        exclude_obs_ind=variant['exclude_obs_ind'],
        goal_ind=variant['goal_ind'],
        skill_horizon=variant['skill_horizon'],
        # render=True
    )
    scheduler_replay_buffer = HRLSchedulerEnvReplayBuffer(
        variant['scheduler_replay_buffer_size'],
        expl_env,
        variant['skill_horizon'],
        skill_dim,
        goal_dim,
    )
    scheduler_trainer = HRLSchedulerTrainer(
        env=eval_env,
        scheduler=scheduler,
        qf1=qf1_scheduler,
        qf2=qf2_scheduler,
        target_qf1=target_qf1_scheduler,
        target_qf2=target_qf2_scheduler,
        exclude_obs_ind=variant['exclude_obs_ind'],
        **variant['trainer_kwargs']
    )
    scheduler_trainer = SACTrainer(
        env=eval_env,
        policy=scheduler,
        qf1=qf1_scheduler,
        qf2=qf2_scheduler,
        target_qf1=target_qf1_scheduler,
        target_qf2=target_qf2_scheduler,
        # exclude_obs_ind=variant['exclude_obs_ind'],
        # **variant['trainer_kwargs']
    )

    algorithm = HRLSTorchRLAlgorithm(
        trainer=scheduler_trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=scheduler_replay_buffer,
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

    return NormalizedBoxEnv(gym.make(name)), NormalizedBoxEnv(gym.make(name))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str,
                        help='environment')
    parser.add_argument('file', type=str,
                        help='file')
    args = parser.parse_args()

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="HRLS",
        version="normal",
        layer_size=128,
        scheduler_replay_buffer_size=int(1E6),
        exclude_obs_ind=None,
        goal_ind=None,#[0, 1],
        skill_horizon=3,
        batch_norm=False,
        goal_dim=17,
        algorithm_kwargs=dict(
            num_epochs=3000, #1000
            num_eval_steps_per_epoch=201,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=1005,
            num_trains_discriminator_per_train_loop=8,
            min_num_steps_before_training=1005,
            max_path_length=201,
            batch_size=128,
        )
        ,
        trainer_kwargs=dict(
            discount=0.99 ** 3,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1.,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('HRLS_' + args.env, variant=variant,snapshot_mode="gap_and_last",
            snapshot_gap=100,)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, args)
