import os
import yaml
from pathlib import Path
import cv2
import numpy as np
from collections import Counter

def analyze_dataset():
    print("=== DATASET ANALYSIS ===")
    
    # Check for existing dataset
    dataset_paths = [
        "falldataset-3",
        "falldataset-3/train",
        "falldataset-3/valid",
        "falldataset-3/test"
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"\nFound dataset at: {path}")
            
            # Analyze images
            images_path = os.path.join(path, "images")
            labels_path = os.path.join(path, "labels")
            
            if os.path.exists(images_path):
                image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                print(f"  Images: {len(image_files)}")
                
                # Analyze image sizes
                if image_files:
                    sample_image = cv2.imread(os.path.join(images_path, image_files[0]))
                    if sample_image is not None:
                        print(f"  Sample image size: {sample_image.shape[1]}x{sample_image.shape[0]}")
            
            if os.path.exists(labels_path):
                label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
                print(f"  Labels: {len(label_files)}")
                
                # Analyze class distribution
                class_counts = Counter()
                total_annotations = 0
                
                for label_file in label_files[:100]:  # Sample first 100 files
                    label_path = os.path.join(labels_path, label_file)
                    try:
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    class_counts[class_id] += 1
                                    total_annotations += 1
                    except:
                        continue
                
                print(f"  Total annotations (sample): {total_annotations}")
                print(f"  Class distribution: {dict(class_counts)}")
    
    # Check for data.yaml
    yaml_paths = ["falldataset-3/data.yaml", "data.yaml"]
    for yaml_path in yaml_paths:
        if os.path.exists(yaml_path):
            print(f"\nFound configuration at: {yaml_path}")
            try:
                with open(yaml_path, 'r') as f:
                    config = yaml.safe_load(f)
                    print("  Configuration:")
                    for key, value in config.items():
                        print(f"    {key}: {value}")
            except Exception as e:
                print(f"  Error reading YAML: {e}")
    
    # Check for existing training results
    if os.path.exists("runs/detect"):
        print(f"\nExisting training runs:")
        for folder in os.listdir("runs/detect"):
            folder_path = os.path.join("runs/detect", folder)
            if os.path.isdir(folder_path):
                weights_path = os.path.join(folder_path, "weights")
                if os.path.exists(weights_path):
                    weight_files = [f for f in os.listdir(weights_path) if f.endswith('.pt')]
                    print(f"  {folder}: {len(weight_files)} model files")
                    
                    # Check for results.csv
                    results_csv = os.path.join(folder_path, "results.csv")
                    if os.path.exists(results_csv):
                        print(f"    Has training results")
    
    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    analyze_dataset() 