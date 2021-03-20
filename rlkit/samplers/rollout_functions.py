import numpy as np


def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    goal = o[desired_goal_key]
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            o = o[observation_key]
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=dict_obs,
    )


def rollout(
        env,
        agent,
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
    o = env.reset()
    agent.reset()
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


def hierachical_rollout(
        env,
        scheduler,
        worker,
        skill_horizon=1,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        exclude_obs_ind=None,
        obs_ind=None,
        goal_ind=None,
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
        current_states_worker = []
        next_states_worker = []
        actions_scheduler = []
        agent_infos_scheduler = []
        o = env.reset()
        o_policy = o
        if exclude_obs_ind:
            o_policy = o[obs_ind]
        worker.reset()
        a_scheduler, agent_info_scheduler = scheduler.get_action(o_policy)
        worker.set_skill(a_scheduler)
        # skill_start_states.append(o)
        # skills.append(self._worker.skill)
        next_o = None
        path_length = 0
        skill_step = 0
        if render:
            env.render(**render_kwargs)
        while path_length < max_path_length:
            a, agent_info = worker.get_action(o_policy, return_log_prob=True)
            next_o, r, d, env_info = env.step(a)
            observations_worker.append(o)
            rewards_worker.append(r)
            terminals_worker.append(d)
            actions_worker.append(a)
            agent_infos_worker.append(agent_info)
            env_infos_worker.append(env_info)
            actions_scheduler.append(a_scheduler)
            agent_infos_scheduler.append(agent_info_scheduler)
            if goal_ind:
                current_states_worker.append(o[goal_ind])
                next_states_worker.append(next_o[goal_ind])
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
                a_scheduler, agent_info_scheduler = scheduler.get_action(o_policy)
                worker.set_skill(a_scheduler)
            if exclude_obs_ind:
                o_policy = o_policy[obs_ind]
            if render:
                env.render(**render_kwargs)

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