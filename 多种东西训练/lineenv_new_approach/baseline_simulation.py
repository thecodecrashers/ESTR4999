#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate long-run average return under a Whittle-index policy
for several (num_arms, budget) configurations of lineEnv.
2025-06-13
"""

import os, json, argparse, random, pathlib
from typing import List, Tuple, Dict
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lineEnv import lineEnv   # 确保同级或 PYTHONPATH 中能找到

# ------------------------- 命令行参数 -------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="Whittle-index long-run average-return simulator")
    parser.add_argument("--wi_json", type=str, default='baseline\\whittle_index_baseline.json',
                        help="Path to the Whittle-index table (JSON)")
    parser.add_argument("--prob_json", type=str, default="prob_pairs.json",
                        help="p/q 参数文件；默认同目录下 prob_pairs.json")
    parser.add_argument("--seed", type=int, default=42,
                        help="全局随机种子")
    return parser.parse_args()

args = get_args()
ROOT = pathlib.Path(__file__).parent.resolve()

# ------------------------- 读配置 -----------------------------
with open(args.prob_json, "r", encoding="utf-8") as f:
    PROB_PAIRS = json.load(f)        # list[dict{p,q}]

with open(args.wi_json, "r", encoding="utf-8") as f:
    wi_raw = json.load(f)

# {(p,q,state) : index}
WI_TABLE: Dict[Tuple[float, float, int], float] = {
    (round(r["p"], 3), round(r["q"], 3), r["state"]): r["whittle"]
    for r in wi_raw
}

def whittle_value(p: float, q: float, state: int) -> float:
    try:
        return WI_TABLE[(round(p, 3), round(q, 3), state)]
    except KeyError:
        raise KeyError(f"Missing Whittle index for (p={p}, q={q}, state={state})")

# ------------------------- 常量设置 ---------------------------
CONFIGS = [(10, 3), (20, 5), (30, 6)]   # (arm 数, budget)
TOTAL_STEPS = 20_000
WINDOW = 200

N_STATES = 100      # lineEnv 参数
OPT_X = 100

# ------------------------- 主模拟函数 -------------------------
def run_sim(num_arms: int, budget: int, seed: int):
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    envs: List[lineEnv] = []
    for i in range(num_arms):
        p, q = PROB_PAIRS[i]["p"], PROB_PAIRS[i]["q"]
        env = lineEnv(seed=i + seed * 100,
                      N=N_STATES, OptX=OPT_X, p=p, q=q)
        env.reset()
        envs.append(env)

    step_rewards = []

    for t in range(TOTAL_STEPS):
        wi_vals = [whittle_value(env.p, env.q, env.X) for env in envs]
        chosen_idx = np.argsort(wi_vals)[-budget:]
        chosen = set(chosen_idx)

        total_r = 0.0
        for idx, env in enumerate(envs):
            action = 1 if idx in chosen else 0
            _, r, _, _ = env.step(action)
            total_r += r
        step_rewards.append(total_r / num_arms)

    step_arr = np.array(step_rewards)
    window_avg = step_arr.reshape(-1, WINDOW).mean(axis=1)
    return step_arr, window_avg

# ------------------------- 执行并保存 -------------------------
for n_arms, B in CONFIGS:
    print(f"→ ({n_arms} arms, budget {B})")
    steps, win_avg = run_sim(n_arms, B, seed=args.seed)

    out_dir = ROOT / "results" / f"{n_arms}_{B}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "step_rewards.npy", steps)
    np.save(out_dir / "window_avg.npy", win_avg)

    x = np.arange(1, len(win_avg)+1) * WINDOW
    plt.figure(figsize=(8, 4))
    plt.plot(x, win_avg, linewidth=1.8)
    plt.title(f"Avg return – {n_arms} arms, B={B}")
    plt.xlabel("Env step")
    plt.ylabel("Avg reward (window=200)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "avg_return.png", dpi=300)
    plt.close()

    print(f"  数据已保存到 {out_dir}")

print("全部模拟完成")

