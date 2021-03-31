from rlkit.samplers.data_collector.path_collector import MdpPathCollector
from rlkit.samplers.data_collector.step_collector import GoalConditionedStepCollector
from rlkit.samplers.rollout_functions import rollout
import numpy as np
import rlkit.torch.pytorch_util as ptu
import torch


class DIAYNMdpPathCollector(MdpPathCollector):
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

            self._policy.stochastic_policy.skill_reset()

            path = rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
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

class DIAYNGoalMdpStepCollector(GoalConditionedStepCollector):
    def collect_one_step(
            self,
            max_path_length,
            discard_incomplete_paths,
    ):
        if self._obs is None:
            self._start_new_rollout()

        new_obs = self._obs
        action, agent_info = self._policy.get_action(new_obs)
        next_ob, reward, terminal, env_info = (
            self._env.step(action)
        )
        next_ob = next_ob['observation']
        if self._render:
            self._env.render(**self._render_kwargs)
        terminal = np.array([terminal])
        reward = np.array([reward])
        # store path obs
        self._current_path_builder.add_all(
            observations=self._obs,
            actions=action,
            rewards=reward,
            next_observations=next_ob,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        if terminal or len(self._current_path_builder) >= max_path_length:
            self._handle_rollout_ending(max_path_length,
                                        discard_incomplete_paths)
            self._start_new_rollout()
        else:
            self._obs = next_ob

    def _start_new_rollout(self):
        super()._start_new_rollout()
        self._obs = self._obs['observation']
        self._policy.skill_reset()



class DIAYNGoalPathCollector(MdpPathCollector):
    def __init__(
            self,
            env,
            policy,
            df,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        super().__init__(env, policy, max_num_epoch_paths_saved, render, render_kwargs)
        self.df = df

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

            self._policy.stochastic_policy.skill_reset()

            path = multitask_rollout(
                self._env,
                self._policy,
                self.df,
                goal_conditioned=True,
                max_path_length=max_path_length_this_loop,
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
            dist_to_goal += -path['rewards'][-1][0]

        return {
            'success rate': success_count/total,
            'distance to goal': dist_to_goal/total,
        }


def multitask_rollout(
        env,
        agent,
        df,
        goal_conditioned=False,
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
    o_env = env.reset()
    o = o_env['observation']
    goal = o_env['desired_goal']
    agent.reset()
    if goal_conditioned:
        df_input = ptu.FloatTensor([o[:3]])
        skill = df(df_input)
        z_hat = torch.argmax(skill, dim=1)
        agent.stochastic_policy.skill = ptu.get_numpy(z_hat)
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
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
            env.render(**render_kwargs)

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