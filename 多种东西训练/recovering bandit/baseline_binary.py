#!/usr/bin/env python
# whittle_recovering_bandit.py
# ----------------------------------------------------------
# 计算 Recovering Bandit (单臂) 的 Whittle 指数（折扣 γ）
# 二分搜索 + 价值迭代
# ----------------------------------------------------------
import json
import math
import numpy as np
from tqdm import tqdm

# ---------- 可调超参数 ----------
GAMMA       = 0.99      # 折扣因子
TOLERANCE   = 1e-2      # 二分搜索精度
MAX_ITERS   = 1000      # 价值迭代上限
INPUT_F     = "theta_pairs_recovering.json"   # 概率（θ）文件
OUTPUT_F    = "whittle_index_recovering.json" # 输出文件
SEED        = 42

# ---------- 基础函数 ----------
def reward_pull(state: int, theta0: float, theta1: float, theta2: float = 0.0) -> float:
    """
    当 action=1（拉臂）时的期望奖励。
    根据 recoveringBanditsEnv._calReward 中无噪声部分推导：  
    r(s) = θ₀ · (1 - exp(-θ₁·s + θ₂))  :contentReference[oaicite:0]{index=0}
    """
    return theta0 * (1.0 - math.exp(-theta1 * state + theta2))

def value_iteration(max_state: int,
                    cost: float,
                    theta0: float,
                    theta1: float,
                    theta2: float,
                    gamma: float = GAMMA,
                    max_iters: int = MAX_ITERS) -> np.ndarray:
    """
    给定补贴 cost（λ），在 (max_state) 个状态上做价值迭代，
    返回最优 value-function V[s], s∈[1,max_state]
    """
    V = np.zeros(max_state + 1, dtype=np.float64)  # 1-based 存储
    for _ in range(max_iters):
        V_new = V.copy()
        delta = 0.0
        for s in range(1, max_state + 1):
            # 被动：不拉 -> 状态 +1（封顶）
            next_passive = min(s + 1, max_state)
            passive = gamma * V[next_passive]

            # 主动：拉臂 -> 状态重置为 1，获得即时奖励 r(s) - λ
            active = reward_pull(s, theta0, theta1, theta2) - cost + gamma * V[1]

            V_new[s] = max(passive, active)
            delta = max(delta, abs(V_new[s] - V[s]))

        V[:] = V_new
        if delta < 1e-3:
            break
    return V

def whittle_single_theta(max_state: int,
                         theta0: float,
                         theta1: float,
                         theta2: float,
                         tol: float = TOLERANCE) -> np.ndarray:
    """
    对固定 (θ₀,θ₁,θ₂) 计算所有状态的 Whittle 指数。
    """
    lower = np.full(max_state + 1, -1e3, dtype=np.float64)   # λ 下界
    upper = np.full(max_state + 1,  1e3, dtype=np.float64)   # λ 上界
    whittle = np.zeros(max_state + 1, dtype=np.float64)

    for s in range(1, max_state + 1):
        while upper[s] - lower[s] > tol:
            mid = 0.5 * (lower[s] + upper[s])

            V = value_iteration(max_state, mid, theta0, theta1, theta2)

            # 比较主动 / 被动 Q 值 (不含 γ，因为两边都有)
            passive_Q = GAMMA * V[min(s + 1, max_state)]
            active_Q  = reward_pull(s, theta0, theta1, theta2) - mid + GAMMA * V[1]

            if active_Q > passive_Q:
                lower[s] = mid
            else:
                upper[s] = mid

        whittle[s] = 0.5 * (lower[s] + upper[s])

    return whittle[1:]   # 丢弃索引 0，返回长度 = max_state

# ---------- 主流程 ----------
def main():
    np.random.seed(SEED)

    # 1) 加载 θ-对与 maxWait
    with open(INPUT_F, "r", encoding="utf-8") as f:
        data = json.load(f)
        max_state   = int(data["maxWait"])                 # e.g. 100 :contentReference[oaicite:1]{index=1}
        theta_pairs = data["theta_pairs"]                  # List[[θ0,θ1,θ2], ...]

    results = []
    with tqdm(total=len(theta_pairs), desc="Computing Whittle Indices") as pbar:
        for t0, t1, t2 in theta_pairs:
            w_table = whittle_single_theta(max_state, t0, t1, t2)
            for state, w in enumerate(w_table, start=1):
                results.append({
                    "theta0":  t0,
                    "theta1":  t1,
                    "theta2":  t2,
                    "state":   state,
                    "whittle": float(w)
                })
            pbar.update(1)

    # 2) 写出 JSON
    with open(OUTPUT_F, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✔ Done! Saved {len(results)} entries to {OUTPUT_F}")

if __name__ == "__main__":
    main()
