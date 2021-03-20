from collections import OrderedDict

import math
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.envs.env_utils import get_dim

class MSDADSTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            skill_dynamics,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            dyn_lr=1e-3,
            gamma=0.9,
            alpha_is=10,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            exclude_obs_ind=None,
            goal_ind=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.skill_dynamics = skill_dynamics
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.gamma = gamma

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        # self.sf_criterion = nn.CrossEntropyLoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.dyn_optimizer = optimizer_class(
            self.skill_dynamics.parameters(),
            lr=dyn_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.alpha_is = alpha_is
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.exclude_obs_ind = exclude_obs_ind
        if exclude_obs_ind:
            obs_len = get_dim(env.observation_space)
            self.obs_ind = get_indices(obs_len, exclude_obs_ind)

    def train_skill_dynamics(self, batch):
        skills = batch['skills'][:,0]
        actions = batch['actions'][:,0]
        cur_states = batch['cur_states'][:,0]
        next_states = batch['next_states'][:,0]
        log_probs_old = batch['log_probs'][:,0]
        obs = batch['observations'][:,0]


        if self.exclude_obs_ind:
            obs = obs[:, self.obs_ind]

        state_change = next_states - cur_states
        sf_input = torch.cat([obs, skills], dim=1)
        # sf_distribution = self.skill_dynamics(sf_input)
        # log_likelihood = sf_distribution.log_prob(skill_goals-cur_states)
        log_likelihood = self.skill_dynamics.log_prob(sf_input, state_change)# - cur_states)
        importance_weight = self._calc_importance_weight(self.policy.log_prob(obs,skills,actions), log_probs_old)
        # sf_loss = -torch.mean(log_likelihood * importance_weight)
        dyn_loss = -torch.mean(log_likelihood * importance_weight.view(-1))

        self.dyn_optimizer.zero_grad()
        dyn_loss.backward()
        self.dyn_optimizer.step()

        self.eval_statistics['Dyn Loss'] = np.mean(ptu.get_numpy(dyn_loss))


    def train_from_torch(self, batch):
        rewards_batch_steps = batch['rewards']
        terminals_steps = batch['terminals']
        obs_steps = batch['observations']
        actions_steps = batch['actions']
        next_obs_steps = batch['next_observations']
        skills_steps = batch['skills']
        cur_states = batch['cur_states']
        next_states = batch['next_states']

        obs = obs_steps[:, 0]
        actions = actions_steps[:, 0]
        next_obs = next_obs_steps[:, 0]
        skills = skills_steps[:, 0]
        rewards_batch = rewards_batch_steps[:, 0]
        terminals = terminals_steps[:, 0]

        if self.exclude_obs_ind:
            obs_steps = obs_steps[:, :, self.obs_ind]
            next_obs_steps = next_obs_steps[:,:,self.obs_ind]
            obs = obs[:, self.obs_ind]
            next_obs = next_obs[:, self.obs_ind]

        """
        DF Loss and Intrinsic Reward
        """
        state_change = next_states - cur_states

        # dyn_input = torch.cat([obs, skills], dim=1)
        # dyn_distribution = self.skill_dynamics(dyn_input)
        # log_likelihood = dyn_distribution.log_prob(state_change)
        # rewards = log_likelihood.view(-1, 1)
        rewards = self._calc_reward(obs_steps, state_change, skills_steps)
        # dyn_loss = -log_likelihood.mean()

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, skill_vec=skills, reparameterize=True, return_log_prob=True,
        )
        obs_skills = torch.cat((obs, skills), dim=1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = .1

        q_new_actions = torch.min(
            self.qf1(obs_skills, new_obs_actions),
            self.qf2(obs_skills, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs_skills, actions)
        q2_pred = self.qf2(obs_skills, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, skill_vec = skills, reparameterize=True, return_log_prob=True,
        )
        next_obs_skills = torch.cat((next_obs, skills), dim=1)
        target_q_values = torch.min(
            self.target_qf1(next_obs_skills, new_next_actions),
            self.target_qf2(next_obs_skills, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """

        # self.dyn_optimizer.zero_grad()
        # dyn_loss.backward()
        # self.dyn_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        # df_accuracy = torch.sum(torch.eq(z_hat, pred_z.reshape(1, list(pred_z.size())[0])[0])).float()/list(pred_z.size())[0]

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['Intrinsic Rewards'] = np.mean(ptu.get_numpy(rewards))
            # self.eval_statistics['Dyn Loss'] = np.mean(ptu.get_numpy(dyn_loss))
            # self.eval_statistics['Dis Loss'] = np.mean(ptu.get_numpy(dis_loss))
            # self.eval_statistics['DF Accuracy'] = np.mean(ptu.get_numpy(df_accuracy))
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'D Predictions',
            #     ptu.get_numpy(pred_z),
            # ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.skill_dynamics
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
            skill_dynamics=self.skill_dynamics,
        )

    # def _calc_reward(self, cur_states, next_states, skills):
    #     num_sample_skills = 50
    #     state_dim = cur_states.shape[-1]
    #     skill_dim = skills.shape[-1]
    #     num_steps = cur_states.shape[-2]
    #     next_state_dim = next_states.shape[-1]
    #     sf_input = torch.cat([cur_states, skills], dim=-1).view(-1, state_dim+skill_dim)
    #     logp = self.skill_dynamics.log_prob(sf_input, next_states.view(-1, next_state_dim)).view(-1, num_steps)
    #
    #     #Other skill
    #     cur_states = ptu.get_numpy(cur_states)
    #     num_rows =len(cur_states)
    #     skills_sampled = ptu.get_numpy(self.policy.skill_space.sample(torch.Size([num_sample_skills])))
    #     a = np.repeat(cur_states, num_sample_skills, axis=0).reshape(-1, state_dim)
    #     b = np.tile(skills_sampled, (num_rows,num_steps)).reshape(-1, skill_dim)
    #     c = torch.Tensor(np.concatenate([a,b], axis=1))
    #     # c = torch.Tensor(b-a)
    #     x = torch.Tensor(np.repeat(ptu.get_numpy(next_states), num_sample_skills, axis=0)).view(-1, next_state_dim)
    #     log_prob_sample_goal = self.skill_dynamics.log_prob(c, x).view(-1,num_sample_skills, num_steps)
    #     # denom = torch.sum(torch.exp(log_prob), dim=1)
    #     diff = torch.clamp(log_prob_sample_goal - logp,-20, 20)
    #     rewards = -torch.log(1 + torch.exp(diff).sum(dim=1)).view(num_rows, -1) + np.log(num_sample_goal+1)
    #     return rewards

    def _calc_reward(self, cur_states, next_states, skills):
        num_sample_skills = 50
        num_steps = cur_states.shape[-2]
        num_rows = len(cur_states)
        diffs = None
        skills_sampled = ptu.get_numpy(self.policy.skill_space.sample(torch.Size([num_sample_skills])))
        b = np.tile(skills_sampled, (num_rows, 1))
        for i in range(num_steps):
            cur_step_states = cur_states[:, i]
            next_step_states = next_states[:, i]
            step_skills = skills[:, i]
            sf_input = torch.cat([cur_step_states, step_skills], dim=1)
            logp = self.skill_dynamics.log_prob(sf_input, next_step_states).view(-1, 1)
            # df_distribution = self.skill_dynamics(sf_input)
            # logp = df_distribution.log_prob(skill_goal).view(-1, 1)
            #Other goal
            cur_step_states = ptu.get_numpy(cur_step_states)
            a = np.repeat(cur_step_states, num_sample_skills, axis=0)
            # b = np.tile(skills_sampled, (num_rows,1))
            c = torch.Tensor(np.concatenate([a,b], axis=1))
            # c = torch.Tensor(b-a)
            x = torch.Tensor(np.repeat(ptu.get_numpy(next_step_states), num_sample_skills, axis=0))
            log_prob_sample_goal = self.skill_dynamics.log_prob(c, x).view(-1,num_sample_skills)
            # denom = torch.sum(torch.exp(log_prob), dim=1)
            diff = torch.clamp(log_prob_sample_goal - logp,-20, 20)
            if diffs is not None:
                diffs += self.gamma ** i * diff
            else:
                diffs = diff
        rewards = -torch.log(1 + torch.exp(diffs).sum(dim=1)).view(num_rows, -1) + np.log(num_sample_skills+1)
        return rewards

    def _calc_importance_weight(self, log_prob_new, log_prob_old):
        ratio = torch.exp(log_prob_new - log_prob_old)
        return torch.clamp(ratio, 1/self.alpha_is, self.alpha_is)
#
#
def get_indices(length, exclude_ind):
    length = np.arange(length)
    exclude_ind = np.array(exclude_ind).reshape(-1,1)
    return np.nonzero(~np.any(length == exclude_ind, axis=0))[0]