#Importing OpenAI gym package and MuJoCo engine
import gym
import mujoco_py
from envs.mujoco.ant import AntEnv
#Setting MountainCar-v0 as the environment
env = AntEnv(expose_all_qpos=True)
#Sets an initial state
o = env.reset()
print(o[:3])
# Rendering our instance 300 times
for _ in range(100):
  #renders the environment
  env.render()
  #Takes a random action from its action space 
  # aka the number of unique actions an agent can perform
  no,_,_,_ = env.step(env.action_space.sample())
  print(no[:3])
env.close()
