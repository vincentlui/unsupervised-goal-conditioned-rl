import gym
import argparse
#from gym.envs.mujoco import HalfCheetahEnv
from envs.navigation2d.navigation2d import Navigation2d
from rlkit.envs.mujoco.ant import AntEnv
from rlkit.envs.mujoco.half_cheetah import HalfCheetahEnv
from rlkit.envs.mujoco.humanoid import HumanoidEnv
from rlkit.envs.fetch.reach import FetchReachEnv


import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.diayn.diayn_env_replay_buffer import DIAYNEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv, GoalToNormalEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.diayn.diayn_path_collector import DIAYNGoalMdpStepCollector, DIAYNGoalPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.sac.diayn.policies import SkillTanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.diayn.diayn import DIAYNTrainer, DIAYNGoalTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.diayn.diayn_torch_online_rl_algorithm import DIAYNTorchOnlineRLAlgorithm
from rlkit.envs.navigation2d.navigation2d import Navigation2d
from rlkit.envs.mujoco.ant import AntEnv
from rlkit.envs.mujoco.half_cheetah import HalfCheetahEnv
from rlkit.torch.sac.diayn.diayn_cont_torch_online_rl_algorithm import DIAYNContTorchOnlineRLAlgorithm


def experiment(variant, args):
    expl_env, eval_env = get_env(str(args.env))

    # expl_env = NormalizedBoxEnv(gym.make(str(args.env)))
    # eval_env = NormalizedBoxEnv(gym.make(str(args.env)))
    # expl_env = NormalizedBoxEnv(HalfCheetahEnv())
    # eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    skill_dim = args.skill_dim
    goal_dim =3

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
    df = FlattenMlp(
        input_size=goal_dim,
        output_size=skill_dim,
        hidden_sizes=[M, M],
    )
    policy = SkillTanhGaussianPolicy(
        obs_dim=obs_dim + skill_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        skill_dim=skill_dim
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = DIAYNGoalPathCollector(
        eval_env,
        eval_policy,
        df
    )
    expl_step_collector = DIAYNGoalMdpStepCollector(
        expl_env,
        policy,
    )
    replay_buffer = DIAYNEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        skill_dim,
    )
    trainer = DIAYNGoalTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        df=df,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = DIAYNTorchOnlineRLAlgorithm(
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

def get_env(name):
    if name == 'test':
        expl_env, eval_env = Navigation2d(), Navigation2d()
        # expl_env.set_random_start_state(True)
        # eval_env.set_random_start_state(True)
        return NormalizedBoxEnv(expl_env), NormalizedBoxEnv(eval_env)
    elif name == 'Ant':
        return NormalizedBoxEnv(AntEnv(expose_all_qpos=False)), NormalizedBoxEnv(AntEnv(expose_all_qpos=True))
    elif name == 'Half-cheetah':
        return NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=True)), NormalizedBoxEnv(HalfCheetahEnv(expose_all_qpos=True))
    elif name == 'Humanoid':
        return NormalizedBoxEnv(HumanoidEnv(expose_all_qpos=False)), NormalizedBoxEnv(HumanoidEnv(expose_all_qpos=False))
    elif name == 'FetchReach-v1':
        return GoalToNormalEnv(FetchReachEnv()), GoalToNormalEnv(FetchReachEnv(reward_type='dense'))
    elif name == 'FetchPush-v1':
        return GoalToNormalEnv(gym.make(name)), GoalToNormalEnv(gym.make(name))

    return NormalizedBoxEnv(gym.make(name)), NormalizedBoxEnv(gym.make(name))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str,
                        help='environment')
    parser.add_argument('--skill_dim', type=int, default=10,
                        help='skill dimension')
    args = parser.parse_args()

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="DIAYN",
        version="normal",
        layer_size=128,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000, #1000
            num_eval_steps_per_epoch=400,
            num_trains_per_train_loop=600,
            num_expl_steps_per_train_loop=600,
            min_num_steps_before_training=2000,
            max_path_length=40,
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
    setup_logger('DIAYN_' + str(args.skill_dim) + '_' + args.env, variant=variant,snapshot_mode="gap_and_last",
            snapshot_gap=100,)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, args)
