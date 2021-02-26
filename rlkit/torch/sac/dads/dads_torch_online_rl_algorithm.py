import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import (
    PathCollector,
    StepCollector,
)
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.sac.gcs.gcs_env_replay_buffer import GCSEnvReplayBuffer


class DADSTorchOnlineRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
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
            num_trains_skill_dynamics_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
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
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.num_trains_skill_dynamics_per_train_loop = num_trains_skill_dynamics_per_train_loop

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
            )
            init_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(init_expl_paths)
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

            # set policy  for one epoch
            self.policy.skill_reset()

            for _ in range(self.num_train_loops_per_epoch):
                self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,  # num steps
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                new_expl_paths = self.expl_data_collector.get_epoch_paths()
                self.replay_buffer.add_paths(new_expl_paths)
                on_replay_buffer = self.create_online_replay_buffer(buffer_size=self.num_expl_steps_per_train_loop)
                on_replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_skill_dynamics_per_train_loop):
                    train_data = on_replay_buffer.random_batch(self.batch_size)
                    train_data = np_to_pytorch_batch(train_data)
                    self.trainer.train_skill_dynamics(train_data)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = on_replay_buffer.random_batch(
                        self.batch_size)
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

    def create_online_replay_buffer(self, buffer_size):
        b = GCSEnvReplayBuffer(
            buffer_size,
            self.expl_env,
            self.replay_buffer.skill_dim,
            self.replay_buffer.goal_dim,
        )
        return b
