import json

def halton_single(index, base):
    """Compute a single Halton sequence value for a given index and base."""
    result = 0.0
    f = 1.0
    i = index
    while i > 0:
        f = f / base
        result += f * (i % base)
        i //= base
    return result

def generate_halton_pairs(n=100, start=0.1, end=0.9):
    """
    生成 n 个 Halton 序列点（dimension=2），并映射到 [start, end] 区间。
    Halton 基数选用 [2, 3]。
    返回键为 {"p": ..., "q": ...} 的字典列表。
    """
    p_vals = [halton_single(i, 2) for i in range(1, n+1)]
    q_vals = [halton_single(i, 3) for i in range(1, n+1)]
    pairs = []
    for x, y in zip(p_vals, q_vals):
        p = round(start + x * (end - start), 3)
        q = round(start + y * (end - start), 3)
        pairs.append({"p": p, "q": q})
    return pairs

if __name__ == "__main__":
    pairs = generate_halton_pairs(100)
    
    # 写入 JSON 文件
    with open("prob_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    
    # 打印前 10 对
    print("前 10 对均匀分布 (低差异 Halton 序列)：")
    for idx, pair in enumerate(pairs[:10], start=1):
        print(f"{idx}: p={pair['p']}, q={pair['q']}")
