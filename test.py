#Importing OpenAI gym package and MuJoCo engine
import gym
import mujoco_py
import numpy as np
from envs.mujoco.ant import AntEnv
from rlkit.samplers.util import DIAYNRollout as rollout
import torch

#Setting MountainCar-v0 as the environment
env = AntEnv(expose_all_qpos=True)
data = torch.load('data/test/params.pkl', map_location='cpu')
policy = data['evaluation/policy']
#Sets an initial state
o = env.reset()
# print(o[:3])
# Rendering our instance 300 times
action = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
env = AntEnv(expose_all_qpos=True)
# figure = plt.figure()
skills = torch.Tensor(np.vstack([np.arange(-1, 1.1, 0.2), 0 * np.ones(11)])).transpose(1, 0)
for skill in skills:
    # skill = policy.stochastic_policy.skill_space.sample()
    # print(skill)
    path = rollout(env, policy, skill, max_path_length=100, render=True)
    obs = path['observations']
    # plt.plot(obs[:,0], obs[:,1])#, label=tuple(skill.numpy()))
    action = path['actions']
    # print(action)