from roboflow import Roboflow
from ultralytics import YOLO
import csv
import os
import shutil
from pathlib import Path

print("=== COMPREHENSIVE FALL DETECTION MODEL TRAINING (5-EPOCH SAVES) ===")

# Download dataset from Roboflow
print("\n1. Downloading dataset...")
rf = Roboflow(api_key="QFkOxliBmlCDsQQvPrzy")
project = rf.workspace("labels-acgyh").project("falldataset-53lvn")
dataset = project.version(3).download("yolov8")

data_yaml_path = dataset.location + "/data.yaml"

# Analyze dataset
print("\n2. Analyzing dataset...")
if os.path.exists(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        print("Dataset configuration:")
        print(f.read())

# Check dataset structure
dataset_path = dataset.location
train_images = len(os.listdir(os.path.join(dataset_path, "train/images")))
val_images = len(os.listdir(os.path.join(dataset_path, "val/images")))
print(f"\nDataset statistics:")
print(f"Training images: {train_images}")
print(f"Validation images: {val_images}")

# Train multiple models for ensemble approach
models_to_train = [
    ("yolov8n.pt", "nano"),
    ("yolov8s.pt", "small"), 
    ("yolov8m.pt", "medium"),
    ("yolov8l.pt", "large")
]

best_models = []

for model_file, model_name in models_to_train:
    print(f"\n3. Training {model_name.upper()} model...")
    
    # Load model
    model = YOLO(model_file)
    
    # Comprehensive training parameters with 5-epoch saves
    results = model.train(
        data=data_yaml_path,
        epochs=300,              # More epochs for better convergence
        batch=4,                 # Optimized batch size
        imgsz=640,              # Standard image size
        patience=50,            # More patience
        save_period=5,          # Save every 5 epochs (like original)
        verbose=True,
        
        # Advanced learning parameters
        lr0=0.001,              # Lower initial learning rate
        lrf=0.01,               # Lower final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        
        # Comprehensive data augmentation
        augment=True,
        degrees=15.0,           # Rotation
        translate=0.2,          # Translation
        scale=0.8,              # Scale
        shear=5.0,              # Shear
        perspective=0.001,      # Perspective
        flipud=0.0,             # No vertical flip for fall detection
        fliplr=0.5,             # Horizontal flip
        mosaic=1.0,             # Mosaic
        mixup=0.2,              # Mixup
        copy_paste=0.1,         # Copy-paste augmentation
        
        # Advanced training settings
        cos_lr=True,            # Cosine learning rate scheduler
        close_mosaic=10,        # Close mosaic in final epochs
        warmup_epochs=5,        # Warmup epochs
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Validation and monitoring
        val=True,
        plots=True,
        save=True,
        save_conf=True,         # Save confidence scores
        save_txt=True,          # Save predictions as text
    )
    
    # Validate the model
    print(f"\n4. Validating {model_name} model...")
    val_results = model.val()
    
    # Save model info
    model_info = {
        'name': model_name,
        'model_path': model.ckpt_path,
        'mAP50': val_results.box.map50,
        'mAP50-95': val_results.box.map,
        'precision': val_results.box.mp,
        'recall': val_results.box.mr
    }
    
    best_models.append(model_info)
    
    print(f"{model_name.upper()} Results:")
    print(f"  mAP50: {val_results.box.map50:.3f}")
    print(f"  mAP50-95: {val_results.box.map:.3f}")
    print(f"  Precision: {val_results.box.mp:.3f}")
    print(f"  Recall: {val_results.box.mr:.3f}")

# Find the best model
print("\n5. Analyzing results...")
best_model = max(best_models, key=lambda x: x['mAP50'])
print(f"\nBEST MODEL: {best_model['name'].upper()}")
print(f"mAP50: {best_model['mAP50']:.3f}")
print(f"Model path: {best_model['model_path']}")

# Save comprehensive results with 5-epoch metrics
print("\n6. Saving comprehensive results with 5-epoch metrics...")
results_file = "comprehensive_training_results_5epoch.txt"
with open(results_file, "w", encoding="utf-8") as f:
    f.write("=== COMPREHENSIVE FALL DETECTION TRAINING RESULTS (5-EPOCH SAVES) ===\n\n")
    f.write(f"Dataset: {dataset_path}\n")
    f.write(f"Training images: {train_images}\n")
    f.write(f"Validation images: {val_images}\n")
    f.write(f"Save period: Every 5 epochs\n\n")
    
    f.write("MODEL COMPARISON:\n")
    f.write("-" * 80 + "\n")
    for model_info in best_models:
        f.write(f"{model_info['name'].upper()}:\n")
        f.write(f"  mAP50: {model_info['mAP50']:.3f}\n")
        f.write(f"  mAP50-95: {model_info['mAP50-95']:.3f}\n")
        f.write(f"  Precision: {model_info['precision']:.3f}\n")
        f.write(f"  Recall: {model_info['recall']:.3f}\n")
        f.write(f"  Model path: {model_info['model_path']}\n\n")
    
    f.write(f"BEST MODEL: {best_model['name'].upper()}\n")
    f.write(f"Best mAP50: {best_model['mAP50']:.3f}\n")

# Save metrics for each 5-epoch checkpoint
for model_info in best_models:
    model_name = model_info['name']
    runs_dir = f"runs/detect/train"  # Update based on actual folder name
    results_csv = os.path.join(runs_dir, "results.csv")
    
    if os.path.exists(results_csv):
        print(f"\n7. Saving 5-epoch metrics for {model_name}...")
        with open(results_csv, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            for epoch in range(4, 300, 5):  # Every 5 epochs (0-based index)
                if epoch < len(rows):
                    row = rows[epoch]
                    metrics_path = os.path.join(runs_dir, f"metrics_epoch_{epoch+1}_{model_name}.txt")
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        f.write(f"Epoch: {epoch+1} - {model_name.upper()}\n")
                        f.write("=" * 50 + "\n")
                        for k, v in row.items():
                            f.write(f"{k}: {v}\n")

# Export best model
print(f"\n8. Exporting best model ({best_model['name']})...")
best_model_instance = YOLO(best_model['model_path'])
best_model_instance.export(format="onnx")

print("\n=== TRAINING COMPLETED ===")
print(f"Best model: {best_model['name']} with mAP50: {best_model['mAP50']:.3f}")
print(f"Results saved to: {results_file}")
print("Models saved every 5 epochs as requested!")
print("You can now use the best model for real-time detection!") 