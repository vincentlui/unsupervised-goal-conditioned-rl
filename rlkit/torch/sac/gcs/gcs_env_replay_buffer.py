from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import numpy as np

class GCSEnvReplayBuffer(EnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            skill_dim,
            end_state_dim,
            env_info_sizes=None,
            end_state_key=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self._skill = np.zeros((max_replay_buffer_size, skill_dim))
        self._end_state = np.zeros((max_replay_buffer_size, end_state_dim))
        self.end_state_key = end_state_key

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            env_info_sizes=env_info_sizes
        )

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """

        if self.end_state_key is None:
            end_state = path["next_observations"][-1]
        else:
            end_state = path["env_infos"][-1][self.end_state_key]

        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        )):
            agent_info['end_state'] = end_state
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, agent_info, **kwargs):
        self._skill[self._top] = agent_info["skill"]
        self._end_state[self._top] = agent_info["end_state"]

        return super().add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            **kwargs
        )

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            skills=self._skill[indices],
            end_state =  self._end_state[indices]
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch
