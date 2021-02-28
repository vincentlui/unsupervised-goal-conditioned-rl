from rlkit.samplers.data_collector.path_collector import MdpPathCollector
from collections import OrderedDict
from rlkit.samplers.rollout_functions import rollout
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.envs.env_utils import get_dim
import numpy as np

class GCSMdpPathCollector(MdpPathCollector):
    def __init__(self,
                env,
                policy,
                max_num_epoch_paths_saved=None,
                render=False,
                render_kwargs=None,
                exclude_obs_ind=None,
                goal_ind=None,
                skill_horizon=1):

        super().__init__(
            env,
            policy,
            max_num_epoch_paths_saved,
            render,
            render_kwargs,
        )
        self.goal_ind = goal_ind
        self.skill_horizon = skill_horizon
        self.exclude_obs_ind = exclude_obs_ind
        if exclude_obs_ind:
            obs_len = get_dim(env.observation_space)
            self.obs_ind = get_indices(obs_len, exclude_obs_ind)

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )

            self._policy.skill_reset()

            path = self._rollout(
                max_path_length=max_path_length_this_loop,
                skill_horizon=self.skill_horizon,
                render=self._render
            )
            # path = self._rollout2(
            #     max_path_length=max_path_length_this_loop,
            #     render=self._render
            # )
            # path = rollout(
            #     env=self._env,
            #     agent=self._policy,
            #     max_path_length=max_path_length_this_loop,
            #     render=self._render
            # )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def _rollout(
            self,
            skill_horizon=1,
            max_path_length=np.inf,
            render=False,
            render_kwargs=None,
    ):
        """
        The following value for the following keys will be a 2D array, with the
        first dimension corresponding to the time dimension.
         - observations
         - actions
         - rewards
         - next_observations
         - terminals

        The next two elements will be lists of dictionaries, with the index into
        the list being the index into the time
         - agent_infos
         - env_infos
        """
        if render_kwargs is None:
            render_kwargs = {}
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        skill_goals = []
        current_states = []
        o = self._env.reset()
        o_policy = o
        if self.exclude_obs_ind:
            o_policy = o[self.obs_ind]
        self._policy.reset()
        next_o = None
        path_length = 0
        skill_step = 0
        if render:
            self._env.render(**render_kwargs)
        while path_length < max_path_length:
            a, agent_info = self._policy.get_action(o_policy, return_log_prob=True)
            next_o, r, d, env_info = self._env.step(a)
            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a[0])
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            if self.goal_ind:
                current_states.append(o[self.goal_ind])
            else:
                current_states.append(o)
            path_length += 1
            skill_step += 1

            if skill_step >= skill_horizon:
                skill_step = 0
                # self._policy.skill_reset()
                if self.goal_ind:
                    skill_goals.append(next_o[self.goal_ind])
                else:
                    skill_goals.append(next_o)

            if max_path_length == np.inf and d:
                if self.goal_ind:
                    skill_goals.append(next_o[self.goal_ind])
                else:
                    skill_goals.append(next_o)
                break
            o = next_o
            o_policy = o
            if self.exclude_obs_ind:
                o_policy = o_policy[self.obs_ind]
            if render:
                self._env.render(**render_kwargs)

        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        current_states = np.array(current_states)
        if len(actions.shape) == 1:
            current_states = np.expand_dims(current_states, 1)
        observations = np.array(observations)
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, 1)
            next_o = np.array([next_o])
        next_observations = np.vstack(
            (
                observations[1:, :],
                np.expand_dims(next_o, 0)
            )
        )
        skill_goals = np.repeat(np.array(skill_goals), skill_horizon, axis=0)[:len(observations)]
        return dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
            skill_goals=skill_goals,
            current_states=current_states
        )

    def _rollout2(
            self,
            max_path_length=np.inf,
            render=False,
            render_kwargs=None,
    ):
        """
        The following value for the following keys will be a 2D array, with the
        first dimension corresponding to the time dimension.
         - observations
         - actions
         - rewards
         - next_observations
         - terminals

        The next two elements will be lists of dictionaries, with the index into
        the list being the index into the time
         - agent_infos
         - env_infos
        """
        if render_kwargs is None:
            render_kwargs = {}
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        skill_goals = []
        current_states = []
        o = self._env.reset()
        self._policy.reset()
        next_o = None
        path_length = 0
        skill_step = 0
        if render:
            self._env.render(**render_kwargs)
        while path_length < max_path_length:
            a, agent_info = self._policy.get_action(self._policy.skill)
            next_o, r, d, env_info = self._env.step(a)
            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            agent_infos.append(agent_info)
            env_infos.append(env_info)

            path_length += 1

            if max_path_length == np.inf and d:
                break
            o = next_o
            if self.exclude_obs_ind:
                o_policy = o_policy[self.obs_ind]
            if render:
                self._env.render(**render_kwargs)

        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        observations = np.array(observations)
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, 1)
            next_o = np.array([next_o])
        next_observations = np.vstack(
            (
                observations[1:, :],
                np.expand_dims(next_o, 0)
            )
        )
        return dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
        )

    # def get_diagnostics(self):
    #     path_lens = [len(path['actions']) for path in self._epoch_paths]
    #     stats = OrderedDict([
    #         ('num steps total', self._num_steps_total),
    #         ('num paths total', self._num_paths_total),
    #     ])
    #     stats.update(create_stats_ordered_dict(
    #         "path length",
    #         path_lens,
    #         always_show_all_stats=True,
    #     ))
    #     return stats


def get_indices(length, exclude_ind):
    length = np.arange(length)
    exclude_ind = np.array(exclude_ind).reshape(-1,1)
    return np.nonzero(~np.any(length == exclude_ind, axis=0))[0]