'''
Environment to calculate the Whittle index values as a deep reinforcement
learning environment modelled after the OpenAi Gym API.
This is a line setting
There are N states, numbered as {0, 1, ..., N-1}
There is an "optimal state" = OptX
The reward of state (x,y) = 1 - (x-OptX)^2/(max(OptX, N-OptX)^2)
Transition prob is given by:
If activated: move to min{x+1,N-1} w.p. p; stay in the current state w.p. 1-p
If not activated: move max{x-1,0} w.p. q; stay in the current state w.p. 1-q
'''

import gym
import math
import time
import torch
import random
import datetime
import numpy as np
import pandas as pd
from gym import spaces



class lineEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    '''
    Custom Gym environment modelled after "deadline scheduling as restless bandits" paper RMAB description.
    The environment represents one position in the N-length queue. 
    '''

    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc...
    """

    def __init__(self, seed, N, OptX, p,q):
        super(lineEnv, self).__init__()
        self.seed = seed
        self.N = N
        self.OptX = OptX
        self.p = p
        self.q = q
        self.myRandomPRNG = np.random.RandomState(self.seed)

        self.observationSize = 1
        self.X = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(N)

    def _calRewardAndState(self, action):
        ''' function to calculate the reward and next state. '''
        reward = 1 - ((self.X - self.OptX)*(self.X - self.OptX) / (max(self.OptX, self.N-1-self.OptX)*max(self.OptX, self.N-1-self.OptX)))
        nextX = self.X
        if action == 1:
            if (self.myRandomPRNG.choice([1,0], p=[self.p, 1 - self.p]) == 1):
                nextX = min(self.N-1, self.X+1)
        elif action == 0:
            if (self.myRandomPRNG.choice([1,0], p=[self.q, 1 - self.q]) == 1):
                nextX = max(0, self.X-1)
        nextState = [nextX]
        return nextState, reward

    def step(self, action):
        ''' standard Gym function for taking an action. Provides the next state, reward, and episode termination signal.'''
        assert action in [0, 1]

        nextState, reward = self._calRewardAndState(action)
        self.X = nextState[0]
        done = False
        info = {}

        return nextState, reward, done, info

    def reset(self):
        ''' standard Gym function for reseting the state for a new episode.'''
        self.X = 0
        initialState = np.array([self.X], dtype=np.intc)

        return initialState

