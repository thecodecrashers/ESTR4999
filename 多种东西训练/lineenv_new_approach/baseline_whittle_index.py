import json
import numpy as np
from tqdm import tqdm
from lineEnv import lineEnv

def calculate_whittle_index_lineenv(env, tolerance=0.01, max_iterations=1000):
    max_state = env.N - 1
    opt_state = env.OptX
    min_w = np.full(env.N, -1000., dtype=np.float64)
    max_w = np.full(env.N,  1000., dtype=np.float64)
    whittle = np.zeros(env.N, dtype=np.float64)

    def reward(x):
        return 1 - ((x - opt_state)**2 / max(opt_state, max_state - opt_state)**2)

    def value_iteration(cost):
        V = np.zeros(env.N, dtype=np.float64)
        for _ in range(max_iterations):
            V_new = V.copy()
            delta = 0.0
            for x in range(env.N):
                passive = (
                    reward(x)
                    + 0.99 * (env.q * V[max(0, x-1)] + (1-env.q) * V[x])
                )
                active = (
                    reward(x) - cost
                    + 0.99 * (env.p * V[min(max_state, x+1)] + (1-env.p) * V[x])
                )
                V_new[x] = max(passive, active)
                delta = max(delta, abs(V_new[x] - V[x]))
            V[:] = V_new
            if delta < 1e-3:
                break
        return V

    for x in range(env.N):
        while max_w[x] - min_w[x] > tolerance:
            mid = 0.5 * (min_w[x] + max_w[x])
            V = value_iteration(mid)
            passive = reward(x) + env.q * V[max(0, x-1)] + (1-env.q) * V[x]
            active  = reward(x) - mid + env.p * V[min(max_state, x+1)] + (1-env.p) * V[x]
            if active > passive:
                min_w[x] = mid
            else:
                max_w[x] = mid
        whittle[x] = 0.5 * (min_w[x] + max_w[x])

    return whittle

if __name__ == "__main__":
    N        = 100
    OPTX     = 99
    SEED     = 42
    INPUT_F  = "prob_pairs.json"
    OUTPUT_F = "whittle_index_baseline.json"

    np.random.seed(SEED)

    # 1) 从 JSON 里加载概率对
    with open(INPUT_F, "r", encoding="utf-8") as f:
        prob_pairs = json.load(f)
        # prob_pairs 应当是 [{ "p": ..., "q": ... }, ...]

    results = []
    with tqdm(total=len(prob_pairs), desc="Computing Whittle Indices") as pbar:
        for entry in prob_pairs:
            p, q = float(entry["p"]), float(entry["q"])
            env = lineEnv(seed=SEED, N=N, OptX=OPTX, p=p, q=q)
            w_table = calculate_whittle_index_lineenv(env)
            for state, w in enumerate(w_table):
                results.append({
                    "p": p,
                    "q": q,
                    "state": state,
                    "whittle": float(w)
                })
            pbar.update(1)

    # 2) 写入结果
    with open(OUTPUT_F, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved {len(results)} entries to {OUTPUT_F}")
