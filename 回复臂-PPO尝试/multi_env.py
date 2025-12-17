# multi_arm_envs.py

import json
import random
import numpy as np
from rc_env import recoveringBanditsEnv


def load_theta_pairs(json_path="theta_pairs_recovering.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["theta_pairs"]


def set_global_seed(seed=114514):
    random.seed(seed)
    np.random.seed(seed)


class MultiArmEnvDict:
    """
    提供字典形式访问多个单臂环境，如 env["arm3"]
    """

    def __init__(self, arm_count, theta_list):
        set_global_seed(114514)  # 统一设定种子
        self.arms = {}
        for i in range(arm_count):
            theta = theta_list[i]
            env = recoveringBanditsEnv(thetaVals=theta,seed=114514,noiseVar=0.05,maxWait=50)
            self.arms[f"arm{i}"] = env

    def __getitem__(self, key):
        return self.arms[key]

    def keys(self):
        return self.arms.keys()

    def values(self):
        return self.arms.values()

    def items(self):
        return self.arms.items()

    def reset_all(self):
        return {k: env.reset() for k, env in self.arms.items()}

    def step(self, action_dict):
        """
        action_dict: { "arm3": 1, "arm5": 0, ... }
        返回: { "arm3": (s, r, d, i), ... }
        """
        result = {}
        for k, a in action_dict.items():
            result[k] = self.arms[k].step(a)
        return result


# === 供外部使用的工厂函数 ===

def get_10_arm_env():
    theta_pairs = load_theta_pairs()
    return MultiArmEnvDict(10, theta_pairs)


def get_20_arm_env():
    theta_pairs = load_theta_pairs()
    return MultiArmEnvDict(20, theta_pairs)
