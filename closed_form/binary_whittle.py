import numpy as np
import json
from tqdm import tqdm

def bellman_equation(a_max, d_max, m, lambd, tol=1e-6, max_iter=100000):
    f = np.zeros((a_max + 1, d_max + 1))  # 初始化 f(a, d) 的表
    f[1, 0] = 0  # 强制设定 f(1,0) = 0
    J = 0  # 初始化长期平均收益 J
    
    #reference_point = (1, 0)  # 选择 (1,0) 作为参考点
    reference_point = (a_max // 2, d_max // 2)
    
    for _ in range(max_iter):
        f_old = f.copy()
        for a in range(1, a_max + 1):
            for d in range(d_max + 1):
                if a == 1 and d == 0:
                    continue
                term1 = d + a - J + (1 - lambd) * f[min(a + 1, a_max), d] + lambd * f[1, min(d + a, d_max)]
                term2 = a + m - J + (1 - lambd) * f[min(a + 1, a_max), 0] + lambd * f[1, min(a, d_max)]
                f[a, d] = min(term1, term2)
        
        J_new = f[reference_point]  # 选定参考点 (1,0) 计算 J
        if np.abs(J_new - J) < tol and np.max(np.abs(f - f_old)) < tol:
            break
        J = J_new
    
    return f, J

def whittle_index(a_max, d_max, lambd, tol=1e-6, max_iter=100000, w_tol=1e-4, w_iter=50):
    whittle_indices = np.zeros((a_max + 1, d_max + 1))
    
    with tqdm(total=a_max * d_max, desc="Computing Whittle Index", unit="points") as pbar:
        for a in range(1, (a_max + 1)//5):
            for d in range((d_max + 1)//5):
                low, high = 0, 100 # 设定初始搜索范围
                while high - low > w_tol:
                    mid = (low + high) / 2
                    f_mid, _ = bellman_equation(a_max, d_max, mid, lambd, tol, max_iter)
                    
                    term1 = d + a + (1 - lambd) * f_mid[min(a + 1, a_max), d] + lambd * f_mid[1, min(d + a, d_max)]
                    term2 = a + mid + (1 - lambd) * f_mid[min(a + 1, a_max), 0] + lambd * f_mid[1, min(a, d_max)]
                    
                    if np.abs(term1 - term2) < w_tol:
                        whittle_indices[a, d] = mid
                        break
                    elif term1 < term2:
                        high = mid
                    else:
                        low = mid
                pbar.update(1)
    
    whittle_indices_dict = {str((a, d)): whittle_indices[a, d] for a in range(a_max + 1) for d in range(d_max + 1)}
    with open("binary_whittle_indices.json", "w") as f:
        json.dump(whittle_indices_dict, f)
    
    return whittle_indices

# 示例调用
whittle_indices = whittle_index(50, 50, 0.5)
print("Whittle Index Table Saved as JSON.")

