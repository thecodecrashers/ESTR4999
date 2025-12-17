import os
import numpy as np
import matplotlib.pyplot as plt

def plot_and_save_whittle_index(npy_path):
    # 1. 检查文件是否存在
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"找不到文件：{npy_path}")
    
    # 2. 加载数据
    data = np.load(npy_path)
    print(f"加载数据 shape: {data.shape}")

    # 3. squeeze 到合适维度
    data = np.squeeze(data)  # (50, 1) → (50,)

    # 4. 判断维度并绘图
    plt.figure(figsize=(10, 6))
    if data.ndim == 1:
        x = np.arange(len(data))
        plt.plot(x, data, label="Whittle Index")
    elif data.ndim == 2:
        x = np.arange(data.shape[1])
        for i in range(data.shape[0]):
            plt.plot(x, data[i], label=f"Curve {i}")
        plt.legend()
    else:
        raise ValueError("只支持1维或2维 numpy 数组")

    plt.xlabel("State")
    plt.ylabel("Whittle Index")
    plt.title("Whittle Index Curve")
    plt.grid(True)

    # 5. 保存图片
    folder = os.path.dirname(os.path.abspath(npy_path))
    output_path = os.path.join(folder, "whittle_index_plot.png")
    plt.savefig(output_path)
    print(f"图像已保存到：{output_path}")
    plt.close()

if __name__ == "__main__":
    # ✅ 修改为你的文件路径
    npy_file_path = r"D:\回复臂-PPO尝试\actor_critic\20arms_arm19\whittle_index.npy"  # 可改为绝对路径
    plot_and_save_whittle_index(npy_file_path)

