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


class GCSTorchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
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
        # self.goal_buffer = goal_buffer
        # self.sd_replay_buffer = sd_replay_buffer
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_trains_discriminator_per_train_loop = num_trains_discriminator_per_train_loop
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.num_epoch_before_goal_condition_sampling = num_epoch_before_goal_condition_sampling

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
            self.policy.skill_reset()

            for _ in range(self.num_train_loops_per_epoch):
                self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,  # num steps
                    discard_incomplete_paths=False,
                    # goal_condition_training=(epoch>=self.num_epoch_before_goal_condition_sampling),
                )
                gt.stamp('exploration sampling', unique=False)

                new_expl_paths = self.expl_data_collector.get_epoch_paths()
                self.replay_buffer.add_paths(new_expl_paths)
                # self.goal_buffer.add(skill_goals)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                # for _ in range(self.num_trains_discriminator_per_train_loop):
                #     train_data = self.replay_buffer.random_batch(
                #         self.batch_size)
                #     self.trainer.train_discriminator(np_to_pytorch_batch(train_data))
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                # train_data = self.replay_buffer.random_batch(
                #     self.batch_size)
                # self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class GCSTorchRLAlgorithm2(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: StepCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            sd_replay_buffer,
            goal_path_buffer,
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
        self.sd_replay_buffer = sd_replay_buffer
        self.goal_path_buffer = goal_path_buffer
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_trains_discriminator_per_train_loop = num_trains_discriminator_per_train_loop
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.num_epoch_before_goal_condition_sampling = num_epoch_before_goal_condition_sampling

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
            # init_start_goal_pairs = self.expl_data_collector.get_start_goal_pairs()
            self.replay_buffer.add_paths(init_expl_paths)
            # self.sd_replay_buffer.add_paths(init_expl_paths)
            # self.goal_path_buffer.add_samples(init_start_goal_pairs)
            self.expl_data_collector.end_epoch(-1)
            gt.stamp('initial exploration', unique=True)

        num_trains_per_expl_step = self.num_trains_per_train_loop
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            for _ in range(self.num_train_loops_per_epoch):
                self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,  # num steps
                    discard_incomplete_paths=False,
                    # goal_condition_training=(epoch>=self.num_epoch_before_goal_condition_sampling),
                )
                gt.stamp('exploration sampling', unique=False)

                new_expl_paths = self.expl_data_collector.get_epoch_paths()
                # new_start_goal_pairs = self.expl_data_collector.get_epoch_goal_paths()
                # self.sd_replay_buffer.add_paths(new_expl_paths)
                self.replay_buffer.add_paths(new_expl_paths)
                # self.goal_path_buffer.add_samples(new_start_goal_pairs)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_discriminator_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train_skill_dynamics(np_to_pytorch_batch(train_data))

                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                # for _ in range(self.num_trains_discriminator_per_train_loop):
                #     train_data = self.goal_path_buffer.random_batch(
                #         self.batch_size)
                #     self.trainer.train_skill_discriminator(np_to_pytorch_batch(train_data))
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


    # def create_online_replay_buffer(self, buffer_size):
    #     b = GCSEnvReplayBuffer(
    #         buffer_size,
    #         self.expl_env,
    #         self.replay_buffer.skill_dim,
    #         self.replay_buffer.goal_dim,
    #     )
    #     return b