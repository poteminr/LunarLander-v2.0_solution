import gym
from gym import spaces, utils
from gym.utils import seeding
import numpy as np

class FooEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
      """
      Action space:
      0 - left
      1 - up
      2 - right
      3 - down
      """
      self.action_space = spaces.Discrete(4)
      self.observation_space = spaces.Box(0, 3, shape=(2,), dtype=np.float32)
      self.reset()

  def step(self, action):
      obs = 
      return obs, rew, done, info


  def reset(self):
      self.actual_obs = 

  def render(self, mode='human', close=False):
