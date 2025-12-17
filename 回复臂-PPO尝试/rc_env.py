import gym
import math
import time
import torch 
import random
import datetime 
import numpy as np
import pandas as pd
from gym import spaces
from scipy.stats import norm 
import matplotlib.pyplot as plt


class recoveringBanditsEnv(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, seed, thetaVals, noiseVar, maxWait = 20):
        super(recoveringBanditsEnv, self).__init__()

        self.seed = seed
        self.myRandomPRNG = random.Random(self.seed)
        self.G = np.random.RandomState(self.seed) # create a special PRNG for a class instantiation

        self.observationSize = 1 # state size

        self.maxWait = maxWait
        self.noiseVar = noiseVar
        self.myRandomPRNG = np.random.RandomState(self.seed)

        self.arm = {0:1} # initial state of the arm.
        
        self.noiseVector = self.G.normal(0, np.sqrt(self.noiseVar), self.maxWait*2)  

        
        val = sum([2**x for x in np.arange(1,self.maxWait+1)])
        self.stateProbs = [2**(x)/(val) for x in np.arange(1,self.maxWait+1)]

        self.theta0 = thetaVals[0]
        self.theta1 = thetaVals[1]
        self.theta2 = thetaVals[2]

        lowState = np.zeros(self.observationSize, dtype=np.float32)
        highState = np.full(self.observationSize, self.maxWait, dtype=np.float32)

        self.action_space = spaces.Discrete(2) 
        self.state_space = spaces.Box(lowState, highState, dtype=np.float32)

    def _calReward(self, action, stateVal):
        '''Function to calculate recovery function's reward based on supplied state.'''
        if action == 1:
            reward = self.theta0 * (1 - np.exp(-1*self.theta1 * stateVal + self.theta2))
            val = int(stateVal-1)
            noise = self.noiseVector[val]
            
        else:
            reward = 0.0
            val = int(stateVal-1 + self.maxWait)
            noise = self.noiseVector[val]

        return reward + (noise*reward)

    def step(self, action):
        ''' Standard Gym function for taking an action. Supplies nextstate, reward, and episode termination signal.'''
        assert self.action_space.contains(action)

        reward = self._calReward(action, self.arm[0])

        self.currentState = self.arm[0]

        if action == 1:
            self.arm[0] = 1
        elif action == 0:
            self.arm[0] = min(self.arm[0]+1, self.maxWait) 

        nextState = np.array([self.arm[0]], dtype=np.float32)

        done = False

        info = {}

        return nextState, reward, done, info

    def reset(self):
        ''' Standard Gym function for supplying initial episode state.'''

        self.arm[0] = 1 
        initialState = np.array([self.arm[0]], dtype=np.float32)

        return initialState