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
from rlkit.core import logger, eval_util



class HSDTorchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            trainer_worker,
            exploration_env,
            evaluation_env,
            exploration_data_collector: StepCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            scheduler_replay_buffer,
            scheduler_batch_size,
            worker_batch_size,
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
        self.scheduler_batch_size = scheduler_batch_size
        self.worker_batch_size = worker_batch_size
        self.trainer_worker=trainer_worker
        self.scheduler_replay_buffer = scheduler_replay_buffer
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
            init_skill_goals = self.expl_data_collector.get_epoch_goal_paths()
            self.replay_buffer.add_paths(init_expl_paths)
            self.scheduler_replay_buffer.add_paths(init_skill_goals)
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
            self.trainer.scheduler.skill_reset()

            for _ in range(self.num_train_loops_per_epoch):
                self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,  # num steps
                    discard_incomplete_paths=False,
                    # goal_condition_training=(epoch>=self.num_epoch_before_goal_condition_sampling),
                )
                gt.stamp('exploration sampling', unique=False)

                new_expl_paths = self.expl_data_collector.get_epoch_paths()
                new_goal_paths = self.expl_data_collector.get_epoch_goal_paths()
                self.replay_buffer.add_paths(new_expl_paths)
                self.scheduler_replay_buffer.add_paths(new_goal_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_discriminator_per_train_loop):
                    train_data = self.scheduler_replay_buffer.random_batch(
                        self.scheduler_batch_size)
                    self.trainer.train_skill_dynamics(np_to_pytorch_batch(train_data))
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.scheduler_replay_buffer.random_batch(
                        self.scheduler_batch_size)
                    self.trainer.train(train_data)
                    train_data = self.replay_buffer.random_batch(
                        self.worker_batch_size)
                    self.trainer_worker.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)

    def _end_epoch(self, epoch):
        # super()._end_epoch(epoch)
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving')
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)
        self.trainer_worker.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.trainer_worker.get_snapshot().items():
            snapshot['trainer_worker/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.record_dict(self.trainer_worker.get_diagnostics(), prefix='trainer/worker/')
        super()._log_stats(epoch)

