import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

# Set font to support special characters
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Read data
print("Reading data...")
df = pd.read_csv('epoch80_test_results_cleaned.csv')

# Clean data - remove empty rows
df = df.dropna()

print(f"Total data rows: {len(df)}")
print(f"Column names: {df.columns.tolist()}")

# Create figure
fig = plt.figure(figsize=(20, 16))

# Plot 1: relative_error vs filename
ax1 = plt.subplot(2, 2, 1)
ax1.plot(range(len(df)), df['relative_error'], 'b-', linewidth=1, alpha=0.7)
ax1.set_xlabel('Image Index')
ax1.set_ylabel('Relative Error')
ax1.set_title('Relative Error vs Image Index')
ax1.grid(True, alpha=0.3)

# Add statistical information
mean_error = df['relative_error'].mean()
std_error = df['relative_error'].std()
ax1.axhline(y=mean_error, color='r', linestyle='--', alpha=0.7, label=f'Mean Error: {mean_error:.2f}')
ax1.legend()

# Plot 2: Connection lines between ground truth (x1,y1) and predicted (x1,y1) coordinates
ax2 = plt.subplot(2, 2, 2)

# For visualization clarity, only show first 100 points
n_points = min(100, len(df))
sample_indices = np.linspace(0, len(df)-1, n_points, dtype=int)

for i in sample_indices:
    # Ground truth coordinates
    gt_x1, gt_y1 = df.iloc[i]['gt_x1'], df.iloc[i]['gt_y1']
    # Predicted coordinates
    pred_x1, pred_y1 = df.iloc[i]['pred_x1'], df.iloc[i]['pred_y1']
    
    # Draw connection line
    ax2.plot([gt_x1, pred_x1], [gt_y1, pred_y1], 'b-', alpha=0.6, linewidth=0.8)

# Plot ground truth points
ax2.scatter(df.iloc[sample_indices]['gt_x1'], df.iloc[sample_indices]['gt_y1'], 
           c='red', s=20, alpha=0.7, label='Ground Truth (x1,y1)')
# Plot predicted points
ax2.scatter(df.iloc[sample_indices]['pred_x1'], df.iloc[sample_indices]['pred_y1'], 
           c='blue', s=20, alpha=0.7, label='Predicted (x1,y1)')

ax2.set_xlabel('X Coordinate')
ax2.set_ylabel('Y Coordinate')
ax2.set_title(f'Connection Lines: Ground Truth vs Predicted (x1,y1) (First {n_points} Points)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Connection lines between ground truth (x2,y2) and predicted (x2,y2) coordinates
ax3 = plt.subplot(2, 2, 3)

for i in sample_indices:
    # Ground truth coordinates
    gt_x2, gt_y2 = df.iloc[i]['gt_x2'], df.iloc[i]['gt_y2']
    # Predicted coordinates
    pred_x2, pred_y2 = df.iloc[i]['pred_x2'], df.iloc[i]['pred_y2']
    
    # Draw connection line
    ax3.plot([gt_x2, pred_x2], [gt_y2, pred_y2], 'g-', alpha=0.6, linewidth=0.8)

# Plot ground truth points
ax3.scatter(df.iloc[sample_indices]['gt_x2'], df.iloc[sample_indices]['gt_y2'], 
           c='red', s=20, alpha=0.7, label='Ground Truth (x2,y2)')
# Plot predicted points
ax3.scatter(df.iloc[sample_indices]['pred_x2'], df.iloc[sample_indices]['pred_y2'], 
           c='green', s=20, alpha=0.7, label='Predicted (x2,y2)')

ax3.set_xlabel('X Coordinate')
ax3.set_ylabel('Y Coordinate')
ax3.set_title(f'Connection Lines: Ground Truth vs Predicted (x2,y2) (First {n_points} Points)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Comparison between ground truth length and predicted length
ax4 = plt.subplot(2, 2, 4)

# Calculate actual length (pixels)
gt_length_pixels = np.sqrt((df['gt_x2'] - df['gt_x1'])**2 + (df['gt_y2'] - df['gt_y1'])**2)
pred_length_pixels = np.sqrt((df['pred_x2'] - df['pred_x1'])**2 + (df['pred_y2'] - df['pred_y1'])**2)

# Plot line chart
ax4.plot(range(len(df)), gt_length_pixels, 'r-', linewidth=1, alpha=0.8, label='Ground Truth Length')
ax4.plot(range(len(df)), pred_length_pixels, 'b-', linewidth=1, alpha=0.8, label='Predicted Length')

ax4.set_xlabel('Image Index')
ax4.set_ylabel('Scale Bar Length (Pixels)')
ax4.set_title('Comparison of Ground Truth Length vs Predicted Length')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add statistical information
length_error = np.abs(gt_length_pixels - pred_length_pixels)
mean_length_error = length_error.mean()
ax4.text(0.02, 0.98, f'Mean Length Error: {mean_length_error:.2f} pixels', 
         transform=ax4.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('scale_detection_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistical information
print("\n=== Statistical Information ===")
print(f"Total detections: {len(df)}")
print(f"Mean relative error: {df['relative_error'].mean():.4f}")
print(f"Relative error standard deviation: {df['relative_error'].std():.4f}")
print(f"Maximum relative error: {df['relative_error'].max():.4f}")
print(f"Minimum relative error: {df['relative_error'].min():.4f}")

print(f"\nLength detection statistics:")
print(f"Mean ground truth length: {gt_length_pixels.mean():.2f} pixels")
print(f"Mean predicted length: {pred_length_pixels.mean():.2f} pixels")
print(f"Mean length error: {length_error.mean():.2f} pixels")
print(f"Length error standard deviation: {length_error.std():.2f} pixels")

# Detection success rate
detection_rate = df['detected'].sum() / len(df) * 100
print(f"\nDetection success rate: {detection_rate:.2f}%")
