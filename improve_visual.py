import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 输入标签和指标名称
inputs = ['VR-only', 'VR+2IMU', 'VR+3IMU']
metrics = ['MPJRE', 'MPJPE', 'MPJVE', 'Jitter', 'H-PE', 'U-PE', 'L-PE', 'R-PE']

# HMD-Poser指标数据
hmd_poser = np.array([
    [2.32, 3.20, 18.23, 7.13, 1.37, 1.58, 5.54, 2.86],
    [1.89, 2.32, 13.69, 6.88, 1.36, 1.53, 3.47, 2.69],
    [1.80, 2.03, 13.00, 7.13, 1.40, 1.50, 2.81, 2.34],
])

# Ours指标数据
ours = np.array([
    [2.73, 3.08, 19.61, 8.48, 0.85, 1.44, 5.46, 2.68],
    [2.29, 2.17, 14.30, 7.16, 0.82, 1.36, 3.35, 2.40],
    [2.19, 1.90, 13.22, 7.26, 0.83, 1.31, 2.73, 2.05],
])

# 计算增益百分比
gain_percent = (hmd_poser - ours) / hmd_poser * 100

# 创建绘图
fig, ax = plt.subplots(figsize=(12, 6))

width = 0.2
x = np.arange(len(metrics))

for i in range(len(inputs)):
    ax.bar(x + i*width, gain_percent[i], width, label=inputs[i])

ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=16)  # 横轴刻度字体
ax.set_ylabel('Performance Gain (%)', fontsize=18)  # Y轴标签字体
ax.set_title('Percentage Performance Gain of Our Model over HMD-Poser', fontsize=20)  # 标题字体

plt.tight_layout()

# 保存为PDF
with PdfPages('performance_gain.pdf') as pdf:
    pdf.savefig(fig)

plt.show()
