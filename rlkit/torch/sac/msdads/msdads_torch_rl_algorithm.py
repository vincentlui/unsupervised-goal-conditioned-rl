import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import (
    PathCollector,
    StepCollector,
)
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.sac.gcs import gcs_env_replay_buffer


class MSDADSTorchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
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
            num_steps=1,
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
        self.num_steps=num_steps

        # assert self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop, \
        #     'Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop'

        # get policy object for assigning skill
        self.policy = trainer.policy

    def _train(self):
        self.training_mode(False)
        if self.min_num_steps_before_training > 0:
            self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
                # goal_conditioned=False,
            )
            init_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(init_expl_paths)
            # self.goal_buffer.add(skill_goals)
            self.expl_data_collector.end_epoch(-1)
            gt.stamp('initial exploration', unique=True)

        num_trains_per_expl_step = self.num_trains_per_train_loop
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            # self.eval_data_collector.collect_new_paths(
            #     self.max_path_length,
            #     self.num_eval_steps_per_epoch,
            #     discard_incomplete_paths=True,
            # )
            # gt.stamp('evaluation sampling')

            # set policy  for one epoch
            # self.policy.skill_reset()

            for _ in range(self.num_train_loops_per_epoch):
                self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,  # num steps
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                new_expl_paths = self.expl_data_collector.get_epoch_paths()
                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_discriminator_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size, num_steps=1)
                    self.trainer.train_skill_dynamics(np_to_pytorch_batch(train_data))
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size, num_steps=self.num_steps)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
