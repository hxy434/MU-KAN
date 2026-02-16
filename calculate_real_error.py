#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate true relative error
Using ground truth coordinates from scale_bar_labels_fixed.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_real_error():
    """Calculate true relative error"""
    
    print(" Starting to calculate true relative error...")
    
    # Read prediction results
    pred_df = pd.read_csv('scale_endpoints_data.csv')
    print(f" Prediction data: {len(pred_df)} records")
    
    # Read ground truth labels
    gt_df = pd.read_csv('scale_bar_labels_fixed.csv')
    print(f" Ground truth labels: {len(gt_df)} records")
    
    # Calculate actual length from ground truth labels
    gt_df['true_length'] = np.sqrt((gt_df['x2'] - gt_df['x1'])**2 + (gt_df['y2'] - gt_df['y1'])**2)
    
    # Create results list
    error_results = []
    
    for idx, pred_row in pred_df.iterrows():
        image_name = pred_row['image_name']
        
        # Find corresponding image in ground truth labels
        # Image names in prediction file are numbers, while in ground truth are with .png suffix
        gt_match = gt_df[gt_df['filename'] == f"{int(image_name)}.png"]
        
        if len(gt_match) > 0:
            gt_row = gt_match.iloc[0]
            true_length = gt_row['true_length']
            pred_length = pred_row['predicted_length']
            
            # Calculate relative error
            if true_length > 0:
                relative_error = abs(pred_length - true_length) / true_length * 100
            else:
                relative_error = 0
            
            error_results.append({
                'image_name': image_name,
                'predicted_length': pred_length,
                'true_length': true_length,
                'relative_error_percent': relative_error,
                'absolute_error': abs(pred_length - true_length),
                'pred_endpoint1': (pred_row['endpoint1_x'], pred_row['endpoint1_y']),
                'pred_endpoint2': (pred_row['endpoint2_x'], pred_row['endpoint2_y']),
                'true_coords': (gt_row['x1'], gt_row['y1'], gt_row['x2'], gt_row['y2'])
            })
        else:
            print(f" Ground truth label not found for image {image_name}")
    
    # Convert to DataFrame
    error_df = pd.DataFrame(error_results)
    
    if len(error_df) == 0:
        print(" No matching data found")
        return
    
    print(f" Successfully matched: {len(error_df)} records")
    
    # Calculate statistical information
    avg_error = error_df['relative_error_percent'].mean()
    median_error = error_df['relative_error_percent'].median()
    max_error = error_df['relative_error_percent'].max()
    min_error = error_df['relative_error_percent'].min()
    std_error = error_df['relative_error_percent'].std()
    
    # Error distribution statistics
    error_under_5 = len(error_df[error_df['relative_error_percent'] <= 5])
    error_under_10 = len(error_df[error_df['relative_error_percent'] <= 10])
    error_under_20 = len(error_df[error_df['relative_error_percent'] <= 20])
    
    print("\n True Relative Error Statistical Report")
    print("="*60)
    print(f" Basic Statistics:")
    print(f"   • Number of valid samples: {len(error_df)}")
    print(f"   • Average relative error: {avg_error:.2f}%")
    print(f"   • Median relative error: {median_error:.2f}%")
    print(f"   • Maximum relative error: {max_error:.2f}%")
    print(f"   • Minimum relative error: {min_error:.2f}%")
    print(f"   • Standard deviation of error: {std_error:.2f}%")
    print("")
    print(f" Error Distribution:")
    print(f"   • ≤5% error: {error_under_5}/{len(error_df)} ({error_under_5/len(error_df)*100:.1f}%)")
    print(f"   • ≤10% error: {error_under_10}/{len(error_df)} ({error_under_10/len(error_df)*100:.1f}%)")
    print(f"   • ≤20% error: {error_under_20/len(error_df)*100:.1f}%)")
    print("="*60)
    
    # Save detailed results
    output_file = 'real_error_analysis.csv'
    error_df.to_csv(output_file, index=False)
    print(f" Detailed error analysis saved to: {output_file}")
    
    # Create visualization
    create_error_visualization(error_df)
    
    # Sort by error from high to low
    error_df_sorted = error_df.sort_values('relative_error_percent', ascending=False)
    
    print("\n Errors Sorted from Highest to Lowest (Top 20):")
    print("="*90)
    print(f"{'Rank':<6} {'Image Name':<10} {'Pred Length':<12} {'True Length':<12} {'Rel Error':<12} {'Abs Error':<12}")
    print("-"*90)
    
    for i, (idx, row) in enumerate(error_df_sorted.head(20).iterrows(), 1):
        print(f"{i:<6} {row['image_name']:<10} {row['predicted_length']:<12.1f} {row['true_length']:<12.1f} "
              f"{row['relative_error_percent']:<12.2f}% {row['absolute_error']:<12.1f}")
    
    print("-"*90)
    
    # Show best and worst results
    print("\n Best Prediction (Minimum Error):")
    best_idx = error_df['relative_error_percent'].idxmin()
    best_row = error_df.loc[best_idx]
    print(f"   Image: {best_row['image_name']}")
    print(f"   Predicted length: {best_row['predicted_length']:.1f}px")
    print(f"   True length: {best_row['true_length']:.1f}px")
    print(f"   Relative error: {best_row['relative_error_percent']:.2f}%")
    
    print("\n Worst Prediction (Maximum Error):")
    worst_idx = error_df['relative_error_percent'].idxmax()
    worst_row = error_df.loc[worst_idx]
    print(f"   Image: {worst_row['image_name']}")
    print(f"   Predicted length: {worst_row['predicted_length']:.1f}px")
    print(f"   True length: {worst_row['true_length']:.1f}px")
    print(f"   Relative error: {worst_row['relative_error_percent']:.2f}%")
    
    # Save sorted results
    sorted_output_file = 'real_error_analysis_sorted.csv'
    error_df_sorted.to_csv(sorted_output_file, index=False)
    print(f"\n Sorted error analysis saved to: {sorted_output_file}")
    
    # List top 10 images with largest relative error
    print("\n Top 10 Images with Largest Relative Error:")
    print("="*90)
    print(f"{'Rank':<6} {'Image Name':<10} {'Pred Length':<12} {'True Length':<12} {'Rel Error':<12} {'Abs Error':<12}")
    print("-"*90)
    
    # Sort by relative error from high to low to find top 10 largest errors
    largest_error_sorted = error_df.sort_values('relative_error_percent', ascending=False)
    
    for i, (idx, row) in enumerate(largest_error_sorted.head(10).iterrows(), 1):
        print(f"{i:<6} {row['image_name']:<10} {row['predicted_length']:<12.1f} {row['true_length']:<12.1f} "
              f"{row['relative_error_percent']:<12.2f}% {row['absolute_error']:<12.1f}")
    
    print("-"*90)
    print(f" Maximum error: {largest_error_sorted['relative_error_percent'].max():.2f}%")
    print(f" Minimum error: {largest_error_sorted['relative_error_percent'].min():.2f}%")
    
    # Save largest error analysis
    largest_error_output_file = 'largest_error_analysis.csv'
    largest_error_sorted.to_csv(largest_error_output_file, index=False)
    print(f" Largest error analysis saved to: {largest_error_output_file}")
    
    # List top 10 files with largest error difference (vs average error)
    print("\n Top 10 Files with Largest Error Difference (vs Average Error):")
    print("="*90)
    print(f"{'Rank':<6} {'Image Name':<10} {'Pred Length':<12} {'True Length':<12} {'Rel Error':<12} {'Error Diff':<12}")
    print("-"*90)
    
    # Calculate error difference (difference from average error)
    mean_error = error_df['relative_error_percent'].mean()
    error_df_sorted['error_diff'] = abs(error_df_sorted['relative_error_percent'] - mean_error)
    
    # Sort by error difference to find largest deviations from average
    error_diff_sorted = error_df_sorted.sort_values('error_diff', ascending=False)
    
    for i, (idx, row) in enumerate(error_diff_sorted.head(10).iterrows(), 1):
        error_diff = row['error_diff']
        print(f"{i:<6} {row['image_name']:<10} {row['predicted_length']:<12.1f} {row['true_length']:<12.1f} "
              f"{row['relative_error_percent']:<12.2f}% {error_diff:<12.2f}%")
    
    print("-"*90)
    print(f" Average error: {mean_error:.2f}%")
    print(f" Error difference range: {error_diff_sorted['error_diff'].min():.2f}% - {error_diff_sorted['error_diff'].max():.2f}%")
    
    # Save error difference analysis
    error_diff_output_file = 'error_difference_analysis.csv'
    error_diff_sorted.to_csv(error_diff_output_file, index=False)
    print(f" Error difference analysis saved to: {error_diff_output_file}")
    
    return error_df

def create_error_visualization(error_df):
    """Create error visualization charts"""
    
    # Set font (removed Chinese font config as all labels are now English)
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Scale Bar Length Prediction Error Analysis', fontsize=16, fontweight='bold')
    
    # Subplot 1: Predicted vs True Length scatter plot
    ax1.scatter(error_df['true_length'], error_df['predicted_length'], alpha=0.6, color='blue')
    ax1.plot([error_df['true_length'].min(), error_df['true_length'].max()], 
             [error_df['true_length'].min(), error_df['true_length'].max()], 
             'r--', label='Perfect Prediction Line')
    ax1.set_xlabel('True Length (pixels)')
    ax1.set_ylabel('Predicted Length (pixels)')
    ax1.set_title('Predicted Length vs True Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Relative Error distribution histogram
    ax2.hist(error_df['relative_error_percent'], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(error_df['relative_error_percent'].mean(), color='red', linestyle='--', 
                label=f'Average Error: {error_df["relative_error_percent"].mean():.1f}%')
    ax2.set_xlabel('Relative Error (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Relative Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Relative Error by image index
    image_indices = range(1, len(error_df) + 1)
    ax3.plot(image_indices, error_df['relative_error_percent'], 'b-o', markersize=4)
    ax3.axhline(y=5, color='orange', linestyle='--', label='5% Error Line', alpha=0.7)
    ax3.axhline(y=10, color='red', linestyle='--', label='10% Error Line', alpha=0.7)
    ax3.set_xlabel('Image Index')
    ax3.set_ylabel('Relative Error (%)')
    ax3.set_title('Relative Error Variation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Error distribution pie chart
    error_ranges = [
        len(error_df[error_df['relative_error_percent'] <= 5]),
        len(error_df[(error_df['relative_error_percent'] > 5) & (error_df['relative_error_percent'] <= 10)]),
        len(error_df[(error_df['relative_error_percent'] > 10) & (error_df['relative_error_percent'] <= 20)]),
        len(error_df[error_df['relative_error_percent'] > 20])
    ]
    labels = ['≤5%', '5-10%', '10-20%', '>20%']
    colors = ['green', 'yellow', 'orange', 'red']
    
    ax4.pie(error_ranges, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Error Distribution Ratio')
    
    plt.tight_layout()
    
    # Save chart
    plt.savefig('real_error_analysis.png', dpi=300, bbox_inches='tight')
    print(" Error analysis chart saved to: real_error_analysis.png")
    plt.show()

if __name__ == '__main__':
    calculate_real_error()
