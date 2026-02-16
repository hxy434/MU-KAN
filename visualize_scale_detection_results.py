import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("正在读取数据...")
df = pd.read_csv('epoch80_test_results_cleaned.csv')

# 清理数据，移除空行
df = df.dropna()

print(f"数据总行数: {len(df)}")
print(f"列名: {df.columns.tolist()}")

# 创建图形
fig = plt.figure(figsize=(20, 16))

# 第一个图：relative_error vs filename
ax1 = plt.subplot(2, 2, 1)
ax1.plot(range(len(df)), df['relative_error'], 'b-', linewidth=1, alpha=0.7)
ax1.set_xlabel('图片序号')
ax1.set_ylabel('相对误差 (relative_error)')
ax1.set_title('相对误差随图片序号变化')
ax1.grid(True, alpha=0.3)

# 添加统计信息
mean_error = df['relative_error'].mean()
std_error = df['relative_error'].std()
ax1.axhline(y=mean_error, color='r', linestyle='--', alpha=0.7, label=f'平均误差: {mean_error:.2f}')
ax1.legend()

# 第二个图：真实坐标(x1,y1)和预测坐标(x1,y1)的连线
ax2 = plt.subplot(2, 2, 2)

# 为了可视化效果，我们只显示前100个点
n_points = min(100, len(df))
sample_indices = np.linspace(0, len(df)-1, n_points, dtype=int)

for i in sample_indices:
    # 真实坐标
    gt_x1, gt_y1 = df.iloc[i]['gt_x1'], df.iloc[i]['gt_y1']
    # 预测坐标
    pred_x1, pred_y1 = df.iloc[i]['pred_x1'], df.iloc[i]['pred_y1']
    
    # 绘制连线
    ax2.plot([gt_x1, pred_x1], [gt_y1, pred_y1], 'b-', alpha=0.6, linewidth=0.8)

# 绘制真实坐标点
ax2.scatter(df.iloc[sample_indices]['gt_x1'], df.iloc[sample_indices]['gt_y1'], 
           c='red', s=20, alpha=0.7, label='真实坐标 (x1,y1)')
# 绘制预测坐标点
ax2.scatter(df.iloc[sample_indices]['pred_x1'], df.iloc[sample_indices]['pred_y1'], 
           c='blue', s=20, alpha=0.7, label='预测坐标 (x1,y1)')

ax2.set_xlabel('X坐标')
ax2.set_ylabel('Y坐标')
ax2.set_title(f'真实坐标(x1,y1)和预测坐标(x1,y1)的连线 (显示前{n_points}个点)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 第三个图：真实坐标(x2,y2)和预测坐标(x2,y2)的连线
ax3 = plt.subplot(2, 2, 3)

for i in sample_indices:
    # 真实坐标
    gt_x2, gt_y2 = df.iloc[i]['gt_x2'], df.iloc[i]['gt_y2']
    # 预测坐标
    pred_x2, pred_y2 = df.iloc[i]['pred_x2'], df.iloc[i]['pred_y2']
    
    # 绘制连线
    ax3.plot([gt_x2, pred_x2], [gt_y2, pred_y2], 'g-', alpha=0.6, linewidth=0.8)

# 绘制真实坐标点
ax3.scatter(df.iloc[sample_indices]['gt_x2'], df.iloc[sample_indices]['gt_y2'], 
           c='red', s=20, alpha=0.7, label='真实坐标 (x2,y2)')
# 绘制预测坐标点
ax3.scatter(df.iloc[sample_indices]['pred_x2'], df.iloc[sample_indices]['pred_y2'], 
           c='green', s=20, alpha=0.7, label='预测坐标 (x2,y2)')

ax3.set_xlabel('X坐标')
ax3.set_ylabel('Y坐标')
ax3.set_title(f'真实坐标(x2,y2)和预测坐标(x2,y2)的连线 (显示前{n_points}个点)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 第四个图：真实长度和预测长度的对比
ax4 = plt.subplot(2, 2, 4)

# 计算真实长度（像素）
gt_length_pixels = np.sqrt((df['gt_x2'] - df['gt_x1'])**2 + (df['gt_y2'] - df['gt_y1'])**2)
pred_length_pixels = np.sqrt((df['pred_x2'] - df['pred_x1'])**2 + (df['pred_y2'] - df['pred_y1'])**2)

# 绘制折线图
ax4.plot(range(len(df)), gt_length_pixels, 'r-', linewidth=1, alpha=0.8, label='真实长度')
ax4.plot(range(len(df)), pred_length_pixels, 'b-', linewidth=1, alpha=0.8, label='预测长度')

ax4.set_xlabel('图片序号')
ax4.set_ylabel('比例尺长度 (像素)')
ax4.set_title('真实长度和预测长度的对比')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 添加统计信息
length_error = np.abs(gt_length_pixels - pred_length_pixels)
mean_length_error = length_error.mean()
ax4.text(0.02, 0.98, f'平均长度误差: {mean_length_error:.2f}像素', 
         transform=ax4.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('scale_detection_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印一些统计信息
print("\n=== 统计信息 ===")
print(f"总检测数量: {len(df)}")
print(f"平均相对误差: {df['relative_error'].mean():.4f}")
print(f"相对误差标准差: {df['relative_error'].std():.4f}")
print(f"最大相对误差: {df['relative_error'].max():.4f}")
print(f"最小相对误差: {df['relative_error'].min():.4f}")

print(f"\n长度检测统计:")
print(f"平均真实长度: {gt_length_pixels.mean():.2f}像素")
print(f"平均预测长度: {pred_length_pixels.mean():.2f}像素")
print(f"平均长度误差: {length_error.mean():.2f}像素")
print(f"长度误差标准差: {length_error.std():.2f}像素")

# 检测成功率
detection_rate = df['detected'].sum() / len(df) * 100
print(f"\n检测成功率: {detection_rate:.2f}%")
