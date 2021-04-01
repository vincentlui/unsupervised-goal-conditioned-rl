from rlkit.samplers.data_collector.path_collector import MdpPathCollector
from collections import OrderedDict, deque
from rlkit.samplers.rollout_functions import rollout
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.core import eval_np
from rlkit.envs.env_utils import get_dim
from rlkit.torch import pytorch_util as ptu
import numpy as np
import rlkit.torch.pytorch_util as ptu

class GCSMdpPathCollector(MdpPathCollector):
    def __init__(self,
                env,
                policy,
                max_num_epoch_paths_saved=None,
                render=False,
                render_kwargs=None,
                exclude_obs_ind=None,
                goal_ind=None,
                target_obs_name=None,
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
        self.target_obs_name = target_obs_name
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

            # self._policy.skill_reset()

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
        next_states = []
        skill_steps = []
        o = self._env.reset()
        if self.target_obs_name is not None:
            o = o[self.target_obs_name]
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
            if self.target_obs_name is not None:
                next_o = next_o[self.target_obs_name]
            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            skill_steps.append(skill_step)
            if self.goal_ind:
                current_states.append(o[self.goal_ind])
                next_states.append(o[self.goal_ind])
            else:
                current_states.append(o)
                next_states.append(o)
            path_length += 1
            skill_step += 1

            if skill_step >= skill_horizon:
                skill_step = 0
                self._policy.skill_reset()
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
        if len(current_states.shape) == 1:
            current_states = np.expand_dims(current_states, 1)
        next_states = np.array(next_states)
        if len(next_states.shape) == 1:
            next_states = np.expand_dims(next_states, 1)
        # skill_steps = np.array(skill_steps)
        # if len(skill_steps.shape) == 1:
        #     skill_steps = np.expand_dims(skill_steps, 1)
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
            current_states=current_states,
            next_states=next_states,
            # skill_steps=np.array(skill_steps).reshape(-1,1),
        )

    def _rollout2(
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
        last_next_o = None
        path_length = 0
        skill_step = 0
        if render:
            self._env.render(**render_kwargs)
        while path_length < max_path_length:
            a, agent_info = self._policy.get_action(o_policy, return_log_prob=True)
            next_o, r, d, env_info = self._env.step(a)
            if path_length <= max_path_length - skill_horizon:
                observations.append(o)
                rewards.append(r)
                terminals.append(d)
                actions.append()
                agent_infos.append(agent_info)
                env_infos.append(env_info)
                last_next_o = next_o
                if self.goal_ind:
                    current_states.append(o[self.goal_ind])
                else:
                    current_states.append(o)
            path_length += 1
            if path_length >= skill_horizon:
                if self.goal_ind:
                    skill_goals.append(next_o[self.goal_ind])
                else:
                    skill_goals.append(next_o)

            if max_path_length == np.inf and d:
                raise NotImplementedError()
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
            next_o = np.array([last_next_o])
        next_observations = np.vstack(
            (
                observations[1:, :],
                np.expand_dims(last_next_o, 0)
            )
        )
        skill_goals = np.array(skill_goals)
        if len(skill_goals.shape) == 1:
            skill_goals = np.expand_dims(skill_goals, 1)
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

class GCSMdpPathCollector2(MdpPathCollector):
    def __init__(self,
                 env,
                 policy,
                 goal_buffer,
                 skill_discriminator,
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
        self.goal_condition_training = False
        self.goal_buffer = goal_buffer
        self.skill_discriminator = skill_discriminator
        # self.mean, self.std = get_stats(skill_horizon)
        if exclude_obs_ind:
            obs_len = get_dim(env.observation_space)
            self.obs_ind = get_indices(obs_len, exclude_obs_ind)

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            goal_condition_training=False,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )

            if goal_condition_training:
                goal_conditioned = np.random.choice([True,False], 1)[0]
            else:
                goal_conditioned = False

            path, skill_goals = self._rollout(
                max_path_length=max_path_length_this_loop,
                skill_horizon=self.skill_horizon,
                render=self._render,
                goal_conditioned=goal_conditioned
            )
            if not goal_conditioned:
                self.goal_buffer.add(skill_goals)

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
        return paths, skill_goals

    def _rollout(
            self,
            skill_horizon=1,
            max_path_length=np.inf,
            render=False,
            render_kwargs=None,
            goal_conditioned=False,
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
        next_states = []
        skill_steps = []
        o = self._env.reset()
        o_policy = o
        if self.exclude_obs_ind:
            o_policy = o[self.obs_ind]
        self._policy.reset()
        next_o = None
        skill = None
        path_length = 0
        skill_step = 0
        if render:
            self._env.render(**render_kwargs)
        if goal_conditioned:
            if self.goal_ind:
                sampled_goal = self.goal_buffer.pick(far_away_from=o[self.goal_ind])
                sd_input = np.array(np.concatenate((o_policy, sampled_goal[self.goal_ind] - o[self.goal_ind])))
            else:
                sampled_goal = self.goal_buffer.pick(far_away_from=o)
                sd_input = np.array(np.concatenate((o_policy, sampled_goal - o)))
            skill = ptu.get_numpy(eval_np(self.skill_discriminator, sd_input).mean)
            self._policy.set_skill(skill)
        else:
            self._policy.skill_reset()

        while path_length < max_path_length:
            a, agent_info = self._policy.get_action(o_policy, return_log_prob=False)
            next_o, r, d, env_info = self._env.step(a)
            observations.append(o)
            # rewards.append(r)
            if goal_conditioned:
                rewards.append(calc_reward(o, sampled_goal))
            terminals.append(d)
            actions.append(a)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            if self.goal_ind:
                current_states.append(o[self.goal_ind])
                next_states.append(o[self.goal_ind])
            else:
                current_states.append(o)
                next_states.append(o)
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
                # if self.goal_ind:
                #     skill_goals.append(next_o[self.goal_ind])
                # else:
                #     skill_goals.append(next_o)
                raise NotImplementedError
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
        if len(current_states.shape) == 1:
            current_states = np.expand_dims(current_states, 1)
        next_states = np.array(next_states)
        if len(next_states.shape) == 1:
            next_states = np.expand_dims(next_states, 1)
        # skill_steps = np.array(skill_steps)
        # if len(skill_steps.shape) == 1:
        #     skill_steps = np.expand_dims(skill_steps, 1)
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
        if goal_conditioned:
            skill_goals = np.array([sampled_goal])
            obs_skill_goals = np.repeat(skill_goals, len(observations), axis=0)
            rewards = np.array(rewards)
        else:
            skill_goals = np.array(skill_goals)
            if len(skill_goals.shape) == 1:
                skill_goals = np.expand_dims(skill_goals, 1)
            obs_skill_goals = np.repeat(skill_goals, skill_horizon, axis=0)[:len(observations)]
            rewards = calc_reward(observations, obs_skill_goals)
        return dict(
            observations=observations,
            actions=actions,
            rewards=rewards.reshape(-1,1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
            skill_goals=obs_skill_goals,
            current_states=current_states,
            next_states=next_states,
        ), skill_goals

class GCSMdpPathCollector3(MdpPathCollector):
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

            self._policy.skill_reset()

            path, goal_path = self._rollout(
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
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        skill_goals = []
        current_states = []
        next_states = []
        skill_start_states = []
        skills = []
        o = self._env.reset()
        o_policy = o
        if self.exclude_obs_ind:
            o_policy = o[self.obs_ind]
        self._policy.reset()
        skill_start_states.append(o)
        skills.append(self._policy.skill)
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
            actions.append(a)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            if self.goal_ind:
                current_states.append(o[self.goal_ind])
                next_states.append(next_o[self.goal_ind])
            else:
                current_states.append(o)
                next_states.append(next_o)
            path_length += 1
            skill_step += 1

            if skill_step >= skill_horizon:
                skill_step = 0
                self._policy.skill_reset()
                skill_start_states.append(next_o)
                skills.append(self._policy.skill)
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
        if len(current_states.shape) == 1:
            current_states = np.expand_dims(current_states, 1)
        next_states = np.array(next_states)
        if len(next_states.shape) == 1:
            next_states = np.expand_dims(next_states, 1)
        # skill_steps = np.array(skill_steps)
        # if len(skill_steps.shape) == 1:
        #     skill_steps = np.expand_dims(skill_steps, 1)
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

        # For discriminator
        skill_start_states = np.array(skill_start_states)
        if len(skill_start_states.shape) == 1:
            skill_start_states = np.expand_dims(skill_start_states, 1)
        final_state = next_observations[None, -1].repeat(len(skill_start_states), axis=0)
        skills = np.array(skills)
        if len(skills.shape) == 1:
            skills = np.expand_dims(skills, 1)

        return dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
            skill_goals=skill_goals,
            current_states=current_states,
            next_states=next_states,
        ), dict(
            start_states=skill_start_states,
            final_states=final_state,
            skills=skills,
        )

    def end_epoch(self, epoch):
        self._start_goal_pairs_np = None
        super().end_epoch(epoch)

    def get_epoch_goal_paths(self):
        return self._goal_paths

class GCSPathCollector(MdpPathCollector):
    def __init__(self,
                env,
                policy,
                df,
                max_num_epoch_paths_saved=None,
                render=False,
                render_kwargs=None,
                exclude_obs_ind=None,
                goal_ind=None,
                target_obs_name=None,
                skill_horizon=1):

        super().__init__(
            env,
            policy,
            max_num_epoch_paths_saved,
            render,
            render_kwargs,
        )
        self.df = df
        self.goal_ind = goal_ind
        self.skill_horizon = skill_horizon
        self.exclude_obs_ind = exclude_obs_ind
        self.target_obs_name = target_obs_name
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

            path = GCSRollout(
                env = self._env,
                agent=self._policy,
                df=self.df,
                max_path_length=max_path_length_this_loop,
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
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_diagnostics(self):
        total = 0.
        success_count = 0.
        dist_to_goal = 0.
        for path in self._epoch_paths:
            success_count += path['env_infos'][-1]['is_success']
            total += 1
            dist_to_goal += path['rewards'][-1][0] / path['rewards'][0][0]

        return {
            'success rate': success_count/total,
            'distance to goal': dist_to_goal/total,
        }



def GCSRollout(env, agent, df, max_path_length=np.inf, render=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    images = []

    o_env = env.reset()
    o = o_env['observation']
    goal = o_env['desired_goal']
    next_o = None
    path_length = 0
    if render:
        img = env.render('rgb_array')
        # img = env.render(mode= 'rgb_array',width=1900,height=860)
#        env.viewer.cam.fixedcamid = 0
#        env.viewer.cam.type = 2
        images.append(img)

    df_input = ptu.FloatTensor(np.concatenate([o, goal]))
    skill = df(df_input).mean
    agent.set_skill(ptu.get_numpy(skill))

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if max_path_length == np.inf and d:
            break
        next_o = next_o['observation']
        o = next_o
        if render:
            img = env.render('rgb_array')
            # img = env.render(mode= 'rgb_array',width=1900,height=860)
            images.append(img)

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
        images=images
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
