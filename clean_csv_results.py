#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean CSV file and remove records with failed detections
"""

import pandas as pd
import os

def clean_csv_results():
    """Clean CSV file and remove records with failed detections"""
    print("🧹 Cleaning CSV file and removing records with failed detections")
    print("="*50)
    
    # Read original CSV file
    input_csv = "epoch80_test_results.csv"
    if not os.path.exists(input_csv):
        print(f"❌ File does not exist: {input_csv}")
        return
    
    # Read data
    df = pd.read_csv(input_csv)
    print(f"📊 Original data: {len(df)} records")
    
    # Statistics of detection results
    detected_count = len(df[df['detected'] == True])
    undetected_count = len(df[df['detected'] == False])
    
    print(f"✅ Successfully detected: {detected_count} records")
    print(f"❌ Failed detection: {undetected_count} records")
    
    # Keep only successfully detected records
    cleaned_df = df[df['detected'] == True].copy()
    
    print(f"🧹 Cleaned data: {len(cleaned_df)} records")
    
    # Save cleaned file
    output_csv = "epoch80_test_results_cleaned.csv"
    cleaned_df.to_csv(output_csv, index=False)
    print(f"✅ Cleaned file saved to: {output_csv}")
    
    # Recalculate statistical information
    if len(cleaned_df) > 0:
        avg_relative_error = cleaned_df['relative_error'].mean()
        avg_absolute_error = cleaned_df['absolute_error'].mean()
        min_error = cleaned_df['relative_error'].min()
        max_error = cleaned_df['relative_error'].max()
        
        print(f"\n📊 Statistical Information after Cleaning:")
        print(f"Average relative error: {avg_relative_error:.2f}%")
        print(f"Average absolute error: {avg_absolute_error:.1f} pixels")
        print(f"Minimum relative error: {min_error:.2f}%")
        print(f"Maximum relative error: {max_error:.2f}%")
        
        # Error distribution
        error_ranges = [
            (0, 5, "0-5%"),
            (5, 10, "5-10%"),
            (10, 20, "10-20%"),
            (20, 50, "20-50%"),
            (50, float('inf'), ">50%")
        ]
        
        print(f"\nError Distribution:")
        for min_err, max_err, label in error_ranges:
            if max_err == float('inf'):
                count = len(cleaned_df[cleaned_df['relative_error'] >= min_err])
            else:
                count = len(cleaned_df[(cleaned_df['relative_error'] >= min_err) & (cleaned_df['relative_error'] < max_err)])
            percentage = count / len(cleaned_df) * 100
            print(f"  {label}: {count} images ({percentage:.1f}%)")
    
    # Update statistics file
    stats_csv = "epoch80_test_stats_cleaned.csv"
    stats_data = {
        'metric': [
            'total_images',
            'detected_count',
            'undetected_count',
            'detection_rate',
            'avg_relative_error',
            'avg_absolute_error',
            'min_relative_error',
            'max_relative_error'
        ],
        'value': [
            len(df),  # Original total number of images
            len(cleaned_df),  # Number of successfully detected
            undetected_count,  # Number of undetected
            len(cleaned_df)/len(df)*100 if len(df) > 0 else 0,  # Detection rate
            cleaned_df['relative_error'].mean() if len(cleaned_df) > 0 else None,
            cleaned_df['absolute_error'].mean() if len(cleaned_df) > 0 else None,
            cleaned_df['relative_error'].min() if len(cleaned_df) > 0 else None,
            cleaned_df['relative_error'].max() if len(cleaned_df) > 0 else None
        ]
    }
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(stats_csv, index=False)
    print(f"✅ Updated statistical information saved to: {stats_csv}")
    
    return cleaned_df

def show_cleaned_summary():
    """Display summary information after cleaning"""
    print("\n" + "="*50)
    print("📋 Cleaning Summary")
    print("="*50)
    
    # Check cleaned file
    cleaned_csv = "epoch80_test_results_cleaned.csv"
    if os.path.exists(cleaned_csv):
        df = pd.read_csv(cleaned_csv)
        print(f"✅ Cleaned file contains {len(df)} records with successful detections")
        print(f"📁 File path: {cleaned_csv}")
        
        # Show first few rows
        print(f"\n📄 Preview of first 5 records:")
        print(df.head().to_string(index=False))
    else:
        print("❌ Cleaned file does not exist")

if __name__ == '__main__':
    # Clean CSV file
    cleaned_df = clean_csv_results()
    
    # Show summary
    show_cleaned_summary()
