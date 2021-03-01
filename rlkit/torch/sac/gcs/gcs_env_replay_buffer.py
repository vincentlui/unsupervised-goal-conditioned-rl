from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import numpy as np

class GCSEnvReplayBuffer(EnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            skill_dim,
            goal_dim,
            env_info_sizes=None,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.skill_dim = skill_dim
        self.goal_dim = goal_dim
        self._skill = np.zeros((max_replay_buffer_size, skill_dim))
        self._cur_state = np.zeros((max_replay_buffer_size, goal_dim))
        self._next_state = np.zeros((max_replay_buffer_size, goal_dim))
        self._skill_goal = np.zeros((max_replay_buffer_size, goal_dim))
        self._log_prob = np.zeros((max_replay_buffer_size, 1))

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
        # skill_goals = path["next_observations"][-1]
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info,
                skill_goal,
                current_state,
                next_state
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
            path["skill_goals"],
            path["current_states"],
            path["next_states"]
        )):
            agent_info['cur_state'] = current_state
            agent_info['next_state'] = next_state
            agent_info['skill_goal'] = skill_goal
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
        self._cur_state[self._top] = agent_info['cur_state']
        self._next_state[self._top] = agent_info["next_state"]
        self._skill_goal[self._top] = agent_info["skill_goal"]
        self._log_prob[self._top] = agent_info["log_prob"]

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
            cur_states=self._cur_state[indices],
            next_states=self._next_state[indices],
            skill_goals=self._skill_goal[indices],
            log_probs=self._log_prob[indices]
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch
