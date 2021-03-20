from rlkit.samplers.data_collector.path_collector import MdpPathCollector
from collections import OrderedDict, deque
from rlkit.samplers.rollout_functions import rollout
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.core import eval_np
from rlkit.envs.env_utils import get_dim
from rlkit.torch import pytorch_util as ptu
import numpy as np

class HSDMdpPathCollector(MdpPathCollector):
    def __init__(self,
                env,
                scheduler,
                worker,
                max_num_epoch_paths_saved=None,
                render=False,
                render_kwargs=None,
                exclude_obs_ind=None,
                goal_ind=None,
                skill_horizon=1):

        super().__init__(
            env,
            scheduler,
            max_num_epoch_paths_saved,
            render,
            render_kwargs,
        )
        self.goal_ind = goal_ind
        self._worker = worker
        self.skill_horizon = skill_horizon
        self.exclude_obs_ind = exclude_obs_ind
        # self.mean, self.std = get_stats(skill_horizon)
        self._goal_paths = deque(maxlen=self._max_num_epoch_paths_saved)
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
        list_goal_path = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )

            # self._policy.skill_reset()

            path, goal_path = self._rollout(
                max_path_length=max_path_length_this_loop,
                skill_horizon=self.skill_horizon,
                render=self._render
            )

            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
            list_goal_path.append(goal_path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        self._goal_paths.extend(list_goal_path)
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
        observations_worker = []
        actions_worker = []
        rewards_worker = []
        terminals_worker = []
        agent_infos_worker = []
        env_infos_worker = []
        skill_goals_worker = []
        current_states_worker = []
        next_states_worker = []

        observations_scheduler = []
        actions_scheduler = []
        rewards_scheduler = []
        terminals_scheduler = []
        agent_infos_scheduler = []
        env_infos_scheduler = []
        skill_goals_scheduler = []
        current_states_scheduler = []
        next_states_scheduler = []
        # skill_start_states = []
        # skills = []
        o = self._env.reset()
        o_policy = o
        if self.exclude_obs_ind:
            o_policy = o[self.obs_ind]
        self._worker.reset()
        a_scheduler, agent_info_scheduler = self._policy.get_action(o_policy)
        self._worker.set_skill(a_scheduler)
        # skill_start_states.append(o)
        # skills.append(self._worker.skill)
        next_o = None
        path_length = 0
        skill_step = 0
        if render:
            self._env.render(**render_kwargs)
        while path_length < max_path_length:
            a, agent_info = self._worker.get_action(o_policy, return_log_prob=True)
            next_o, r, d, env_info = self._env.step(a)
            observations_worker.append(o)
            rewards_worker.append(r)
            terminals_worker.append(d)
            actions_worker.append(a)
            agent_infos_worker.append(agent_info)
            env_infos_worker.append(env_info)
            # observations_scheduler.append(o)
            # rewards_scheduler.append(r)
            # terminals_scheduler.append(d)
            actions_scheduler.append(a_scheduler)
            agent_infos_scheduler.append(agent_info_scheduler)
            if self.goal_ind:
                current_states_worker.append(o[self.goal_ind])
                next_states_worker.append(next_o[self.goal_ind])
            else:
                current_states_worker.append(o)
                next_states_worker.append(next_o)
            path_length += 1
            skill_step += 1

            if max_path_length == np.inf and d:
                break
            o = next_o
            o_policy = o
            if skill_step >= skill_horizon:
                skill_step = 0
                a_scheduler, agent_info_scheduler = self._policy.get_action(o_policy)
                self._worker.set_skill(a_scheduler)
            if self.exclude_obs_ind:
                o_policy = o_policy[self.obs_ind]
            if render:
                self._env.render(**render_kwargs)

        actions_worker = np.array(actions_worker)
        if len(actions_worker.shape) == 1:
            actions_worker = np.expand_dims(actions_worker, 1)
        current_states_worker = np.array(current_states_worker)
        if len(current_states_worker.shape) == 1:
            current_states_worker = np.expand_dims(current_states_worker, 1)
        next_states_worker = np.array(next_states_worker)
        if len(next_states_worker.shape) == 1:
            next_states_worker = np.expand_dims(next_states_worker, 1)
        observations_worker = np.array(observations_worker)
        if len(observations_worker.shape) == 1:
            observations_worker = np.expand_dims(observations_worker, 1)
            next_o = np.array([next_o])
        next_observations_worker = np.vstack(
            (
                observations_worker[1:, :],
                np.expand_dims(next_o, 0)
            )
        )
        actions_scheduler = np.array(actions_scheduler)
        if len(actions_scheduler.shape) == 1:
            actions_scheduler = np.expand_dims(actions_scheduler, 1)
        skills_scheduler = np.repeat(np.array([agent_info_scheduler['skill']]),
                                     len(actions_scheduler)*skill_horizon, axis=0)

        # For scheduler
        # skill_start_states_np = np.array(skill_start_states)[:-1]
        # if len(skill_start_states_np.shape) == 1:
        #     skill_start_states_np = np.expand_dims(skill_start_states_np, 1)
        # skill_goals = np.vstack(
        #     (
        #         skill_start_states_np[1:, :],
        #         np.expand_dims(skill_start_states[-1], 0)
        #     )
        # )
        # skills = np.array(skills)[:-1]
        # if len(skills.shape) == 1:
        #     skills = np.expand_dims(skills, 1)

        return dict(
            observations=observations_worker,
            actions=actions_worker,
            rewards=np.array(rewards_worker).reshape(-1, 1),
            next_observations=next_observations_worker,
            terminals=np.array(terminals_worker).reshape(-1, 1),
            agent_infos=agent_infos_worker,
            env_infos=env_infos_worker,
            # skill_goals=skill_goals,
            current_states=current_states_worker,
            next_states=next_states_worker,
        ), dict(
            observations=observations_worker.reshape(-1, skill_horizon, observations_worker.shape[-1]),
            actions=actions_scheduler.reshape(-1, skill_horizon, actions_scheduler.shape[-1]),
            rewards=np.array(rewards_worker).reshape(-1, skill_horizon, 1),
            next_observations=next_observations_worker.reshape(-1, skill_horizon, next_observations_worker.shape[-1]),
            terminals=np.array(terminals_worker).reshape(-1, skill_horizon, 1),
            # agent_infos=np.array(agent_infos_scheduler).reshape(-1, skill_horizon, 1),
            # env_infos=np.array(env_infos_worker).reshape(-1, skill_horizon, 1),
            skills=skills_scheduler.reshape(-1, skill_horizon, skills_scheduler.shape[-1]),
            current_states=current_states_worker.reshape(-1, skill_horizon, current_states_worker.shape[-1]),
            next_states=next_states_worker.reshape(-1, skill_horizon, next_states_worker.shape[-1]),
        )

    def end_epoch(self, epoch):
        self._goal_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        super().end_epoch(epoch)

    def get_epoch_goal_paths(self):
        return self._goal_paths

    def get_snapshot(self):
        return dict(
            #env=self._env,
            scheduler=self._policy,
            worker=self._worker
        )

def get_indices(length, exclude_ind):
    length = np.arange(length)
    exclude_ind = np.array(exclude_ind).reshape(-1,1)
    return np.nonzero(~np.any(length == exclude_ind, axis=0))[0]

def calc_reward(obs, goal):
    if len(goal.shape) > 1:
        return -np.sqrt(np.sum(np.square(obs - goal), axis=1))
    return -np.sqrt(np.sum(np.square(np.subtract(obs, goal))))

def get_stats(horizon):
    h = np.arange(horizon)
    return h.mean(), h.std()

def normalize(x, mean, std):
    return (x - mean)/(std + 1e-8)
