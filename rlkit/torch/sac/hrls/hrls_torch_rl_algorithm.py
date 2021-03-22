import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import (
    PathCollector,
    StepCollector,
)
import numpy as np
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.sac.gcs import gcs_env_replay_buffer
from rlkit.core import logger, eval_util



class HRLSTorchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: StepCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_trains_discriminator_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            num_epoch_before_goal_condition_sampling=20,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_trains_discriminator_per_train_loop = num_trains_discriminator_per_train_loop
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.num_epoch_before_goal_condition_sampling = num_epoch_before_goal_condition_sampling
        self.discount=0.99

        # assert self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop, \
        #     'Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop'

    def _train(self):
        self.training_mode(False)
        if self.min_num_steps_before_training > 0:
            self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
                # goal_conditioned=False,
            )
            init_skill_goals = self.expl_data_collector.get_epoch_goal_paths()
            self.replay_buffer.add_paths(init_skill_goals)
            self.expl_data_collector.end_epoch(-1)
            gt.stamp('initial exploration', unique=True)

        num_trains_per_expl_step = self.num_trains_per_train_loop
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')


            for _ in range(self.num_train_loops_per_epoch):
                self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,  # num steps
                    discard_incomplete_paths=False,
                    # goal_condition_training=(epoch>=self.num_epoch_before_goal_condition_sampling),
                )
                gt.stamp('exploration sampling', unique=False)

                # new_expl_paths = self.expl_data_collector.get_epoch_paths()
                new_goal_paths = self.expl_data_collector.get_epoch_goal_paths()
                # self.replay_buffer.add_paths(new_expl_paths)
                self.replay_buffer.add_paths(new_goal_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(self.process_data(train_data))
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)

    def process_data(self, data):
        terminals = data['terminals'][:, -1]
        obs = data['observations']
        next_obs = data['next_observations']
        actions = data['actions']
        rewards_batch = data['rewards']
        #
        # if self.exclude_obs_ind:
        #     obs = obs[:, :, self.obs_ind]
        #     next_obs = next_obs[:, :, self.obs_ind]

        skill_horizon = obs.shape[-2]
        obs_dim = obs.shape[-1]
        gammas = self.discount ** np.arange(skill_horizon)
        data['terminals'] = terminals
        data['observations'] = obs[:, 0]
        data['next_observations'] = next_obs[:, -1]
        data['actions'] = actions[:, 0]
        data['rewards'] = np.sum(rewards_batch * gammas.reshape(-1, 1), axis=1)

        return data