from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import numpy as np

class MSDADSEnvReplayBuffer(EnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            skill_dim,
            goal_dim,
            max_path_length,
            env_info_sizes=None,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.max_path_length = max_path_length
        self.skill_dim = skill_dim
        self.goal_dim = goal_dim
        self._skill = np.zeros((max_replay_buffer_size, max_path_length, skill_dim))
        self._cur_state = np.zeros((max_replay_buffer_size, max_path_length, goal_dim))
        self._next_state = np.zeros((max_replay_buffer_size, max_path_length, goal_dim))
        self._skill_goal = np.zeros((max_replay_buffer_size, max_path_length, goal_dim))
        self._log_prob = np.zeros((max_replay_buffer_size, max_path_length, 1))

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            env_info_sizes=env_info_sizes
        )

        self._observations = np.zeros((max_replay_buffer_size, max_path_length, self._observations.shape[-1]))
        self._actions = np.zeros((max_replay_buffer_size, max_path_length, self._actions.shape[-1]))
        self._next_obs = np.zeros((max_replay_buffer_size, max_path_length, self._next_obs.shape[-1]))
        self._rewards = np.zeros((max_replay_buffer_size, max_path_length, self._rewards.shape[-1]))
        self._terminals = np.zeros((max_replay_buffer_size, max_path_length, self._terminals.shape[-1]))

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)
            self._advance()

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
                next_state,
                # skill_step,
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
            path["next_states"],
            # path["skill_steps"]
        )):
            agent_info['cur_state'] = current_state
            agent_info['next_state'] = next_state
            agent_info['skill_goal'] = skill_goal
            # agent_info['skill_step'] = skill_step
            self.add_sample(
                i,
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()

    def add_sample(self, i, observation, action, reward, terminal,
                   next_observation, agent_info, **kwargs):
        self._skill[self._top, i] = agent_info["skill"]
        self._cur_state[self._top, i] = agent_info['cur_state']
        self._next_state[self._top, i] = agent_info["next_state"]
        self._skill_goal[self._top, i] = agent_info["skill_goal"]
        # self._skill_steps[self._top] = agent_info["skill_step"]
        self._log_prob[self._top, i] = agent_info["log_prob"]
        self._observations[self._top, i] = observation
        self._actions[self._top, i] = action
        self._rewards[self._top, i] = reward
        self._terminals[self._top, i] = terminal
        self._next_obs[self._top, i] = next_observation


    def random_batch(self, batch_size, num_steps):
        path_indices = np.random.randint(0, self._size, batch_size).reshape(-1,1)
        length_indices = np.random.randint(0, self.max_path_length - num_steps + 1, batch_size).reshape(-1,1)
        slice_indices = np.hstack([length_indices + i for i in range(num_steps)])
        batch = dict(
            observations=self._observations[path_indices,slice_indices],
            actions=self._actions[path_indices,slice_indices],
            rewards=self._rewards[path_indices,slice_indices],
            terminals=self._terminals[path_indices,slice_indices],
            next_observations=self._next_obs[path_indices,slice_indices],
            skills=self._skill[path_indices,slice_indices],
            cur_states=self._cur_state[path_indices,slice_indices],
            next_states=self._next_state[path_indices,slice_indices],
            skill_goals=self._skill_goal[path_indices,slice_indices],
            # skill_steps = self._skill_steps[indices],
            log_probs=self._log_prob[path_indices,slice_indices],
        )
        # for key in self._env_info_keys:
        #     assert key not in batch.keys()
        #     batch[key] = self._env_infos[key][indices]
        return batch

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1
