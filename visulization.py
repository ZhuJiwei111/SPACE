import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

# 示例数据
corr_human = pickle.load(open("./temp/corr_human_space.pkl", "rb"))
corr_mouse = pickle.load(open("./temp/corr_mouse_space.pkl", "rb"))
corr_human_baseline = pickle.load(open("./temp//corr_human_baseline.pkl", "rb"))
corr_mouse_baseline = pickle.load(open("./temp//corr_mouse_baseline.pkl", "rb"))

# 实验类型
experiments = list(corr_human.keys())

# 创建画布和子图 (2行4列)
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)

# 绘制每个子图
for i, exp in enumerate(experiments):
    # 获取 Human 和 Mouse 的数据
    human_data = corr_human[exp]
    mouse_data = corr_mouse[exp]
    human_data_baseline = corr_human_baseline[exp]
    mouse_data_baseline = corr_mouse_baseline[exp]

    # 绘制 Human 子图
    print(f"{exp} Human shape: {human_data.shape}")
    ax_human = axes[0, i]  # 调整索引以适应2行4列
    ax_human.scatter(
        human_data_baseline, 
        human_data, 
        alpha=0.6, 
        label="Human"
    )
    ax_human.plot([0, 1], [0, 1], "r--", label="y = x")  # 参考线
    human_avg_model1 = torch.mean(human_data_baseline)
    human_avg_model2 = torch.mean(human_data)
    ax_human.text(0.05, 0.95, f'{human_avg_model2:.8f}', transform=ax_human.transAxes, fontsize=10, verticalalignment='top')
    ax_human.text(0.70, 0.05, f'{human_avg_model1:.8f}', transform=ax_human.transAxes, fontsize=10, verticalalignment='bottom')
    ax_human.set_title(f"{exp} (Human)")

    # 绘制 Mouse 子图
    print(f"{exp} Mouse shape: {mouse_data.shape}")
    ax_mouse = axes[1, i]  # 调整索引以适应2行4列
    ax_mouse.scatter(
        mouse_data_baseline,
        mouse_data,
        alpha=0.6,
        label="Mouse",
        color="orange",
    )
    ax_mouse.plot([0, 1], [0, 1], "r--", label="y = x")  # 参考线
    mouse_avg_model1 = torch.mean(mouse_data_baseline)
    mouse_avg_model2 = torch.mean(mouse_data)
    ax_mouse.text(0.05, 0.95, f'{mouse_avg_model2:.8f}', transform=ax_mouse.transAxes, fontsize=10, verticalalignment='top')
    ax_mouse.text(0.70, 0.05, f'{mouse_avg_model1:.8f}', transform=ax_mouse.transAxes, fontsize=10, verticalalignment='bottom')
    ax_mouse.set_title(f"{exp} (Mouse)")

# 设置共享的 xlabel 和 ylabel
fig.text(0.5, 0.03, "Enformer", ha="center", fontsize=14)  # 共享的 xlabel
fig.text(0.05, 0.5, "SPACE", va="center", rotation="vertical", fontsize=14)  # 共享的 ylabel

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.1, left=0.1, hspace=0.2)  # 减小 hspace 以缩短子图之间的距离
plt.savefig("model_comparison_human_mouse_new.png", dpi=300)  # 保存图像
plt.show()