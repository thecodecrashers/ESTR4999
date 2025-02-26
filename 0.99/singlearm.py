import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SingleArmAoIEnv(gym.Env):
    """
    Single-Arm Age of Information (AoI) Minimization Environment.
    
    状态空间 (a, d) 包含两个维度：
    - a: 更新包到达以来的时间间隔
    - d: 当前成功传输更新后能够改进的 AoI 量
    
    动作空间：
    - 0: 不传输（被动）
    - 1: 传输（主动）
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, max_a=10, max_d=10, pg=0.5, ps=0.7, discount=0.99):
        super(SingleArmAoIEnv, self).__init__()
        
        # 参数设置
        self.max_a = max_a  # 最大等待时间，防止状态空间无限增长
        self.max_d = max_d  # 最大改进量
        self.pg = pg        # 更新包到达的概率
        self.ps = ps        # 传输成功的概率
        self.discount = discount  # 折扣因子
        
        # 状态空间 (a, d) 的范围
        self.observation_space = spaces.MultiDiscrete([self.max_a + 1, self.max_d + 1])
        
        # 动作空间：0表示不传输，1表示传输
        self.action_space = spaces.Discrete(2)
        
        # 初始状态
        self.state = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 初始化状态为 (a=1, d=1)
        self.state = np.array([1, 1], dtype=int)
        return self.state, self.discount,self.max_a,self.max_d,self.pg,self.ps,{}

    def step(self, action):
        a, d = self.state
        
        # 新的更新包到达的概率
        new_packet_arrival = np.random.rand() < self.pg
        transmission_success = action == 1 and np.random.rand() < self.ps

        if action == 1 and transmission_success:
            # 传输成功后，AoI 被重置，并且状态 d 更新
            a = 1  # 成功传输后，a 重置
            d = 1 if new_packet_arrival else min(d + a, self.max_d)  # 有新包则d=1，否则更新d
        elif action == 1 and not transmission_success:
            # 传输失败，则继续增加 a 和 d
            a = min(a + 1, self.max_a)
            d = min(d + a, self.max_d) if new_packet_arrival else d
        else:
            # 没有传输，a 和 d 继续增加
            a = min(a + 1, self.max_a)
            d = min(d + a, self.max_d) if new_packet_arrival else d

        # 更新状态
        self.state = np.array([a, d], dtype=int)
        
        # 计算 AoI 的惩罚（越大越差）
        reward = - (a + d)

        # 环境终止条件：状态空间达到最大值
        done = a == self.max_a and d == self.max_d

        return self.state, reward, done, False, {}

    def render(self, mode='human'):
        print(f"State (a, d): {self.state}")


