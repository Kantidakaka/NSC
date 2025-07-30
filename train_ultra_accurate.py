from roboflow import Roboflow
from ultralytics import YOLO
import csv
import os
import shutil

print("=== ULTRA-ACCURATE FALL DETECTION MODEL TRAINING ===")
print("Optimized for maximum sensitivity to falling people in accidents")

# Download dataset from Roboflow
print("\n1. Downloading dataset...")
rf = Roboflow(api_key="QFkOxliBmlCDsQQvPrzy")
project = rf.workspace("labels-acgyh").project("falldataset-53lvn")
dataset = project.version(3).download("yolov8")

data_yaml_path = dataset.location + "/data.yaml"

# Analyze dataset for fall detection optimization
print("\n2. Analyzing dataset for fall detection...")
dataset_path = dataset.location

# Check for both "val" and "valid" folders
train_path = os.path.join(dataset_path, "train/images")
val_path = os.path.join(dataset_path, "val/images")
valid_path = os.path.join(dataset_path, "valid/images")

if os.path.exists(train_path):
    train_images = len(os.listdir(train_path))
    print(f"Training images: {train_images}")
else:
    print("Training images folder not found")
    train_images = 0

if os.path.exists(val_path):
    val_images = len(os.listdir(val_path))
    print(f"Validation images: {val_images}")
elif os.path.exists(valid_path):
    val_images = len(os.listdir(valid_path))
    print(f"Validation images: {val_images}")
else:
    print("Validation images folder not found")
    val_images = 0

# Train multiple models with fall-optimized settings
models_to_train = [
    ("yolov8s.pt", "small"),   # Good balance of speed and accuracy
    ("yolov8m.pt", "medium"),  # Better accuracy for fall detection
    ("yolov8l.pt", "large")    # Maximum accuracy for critical fall detection
]

best_models = []

for model_file, model_name in models_to_train:
    print(f"\n3. Training {model_name.upper()} model for ultra-accurate fall detection...")
    
    # Load model
    model = YOLO(model_file)
    
    # Ultra-accurate training parameters specifically for fall detection
    results = model.train(
        data=data_yaml_path,
        epochs=500,              # More epochs for maximum learning
        batch=4,                 # Optimized batch size
        imgsz=640,              # Standard size
        patience=100,           # Much more patience to avoid early stopping
        save_period=10,         # Save every 10 epochs
        verbose=True,
        
        # Advanced learning parameters for fall detection
        lr0=0.0005,             # Lower learning rate for precise learning
        lrf=0.005,              # Lower final learning rate
        momentum=0.95,          # Higher momentum
        weight_decay=0.001,     # Higher weight decay for regularization
        
        # Fall-specific data augmentation
        augment=True,
        degrees=20.0,           # More rotation for different fall angles
        translate=0.3,          # More translation
        scale=0.9,              # More scale variation
        shear=10.0,             # More shear for different body positions
        perspective=0.002,      # Perspective for different viewpoints
        flipud=0.0,             # NO vertical flip (important for fall detection)
        fliplr=0.5,             # Horizontal flip
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.3,              # More mixup
        copy_paste=0.2,         # Copy-paste for more data
        
        # Advanced training settings for fall detection
        cos_lr=True,            # Cosine learning rate
        close_mosaic=15,        # Close mosaic later
        warmup_epochs=10,       # More warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Fall detection specific settings
        overlap_mask=True,      # Better mask overlap
        mask_ratio=4,           # Mask ratio
        dropout=0.1,            # Dropout for regularization
        
        # Validation and monitoring
        val=True,
        plots=True,
        save=True,
        save_conf=True,         # Save confidence scores
        save_txt=True,          # Save predictions
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

# Find the best model for fall detection
print("\n5. Selecting best model for fall detection...")
best_model = max(best_models, key=lambda x: x['mAP50'])
print(f"\nBEST FALL DETECTION MODEL: {best_model['name'].upper()}")
print(f"mAP50: {best_model['mAP50']:.3f}")
print(f"Model path: {best_model['model_path']}")

# Save comprehensive results
print("\n6. Saving ultra-accurate training results...")
results_file = "ultra_accurate_fall_detection_results.txt"
with open(results_file, "w", encoding="utf-8") as f:
    f.write("=== ULTRA-ACCURATE FALL DETECTION TRAINING RESULTS ===\n\n")
    f.write("SPECIFICALLY OPTIMIZED FOR FALLING PEOPLE DETECTION\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset: {dataset_path}\n")
    f.write(f"Training images: {train_images}\n")
    f.write(f"Validation images: {val_images}\n")
    f.write(f"Training epochs: 500\n")
    f.write(f"Save period: Every 10 epochs\n\n")
    
    f.write("MODEL COMPARISON FOR FALL DETECTION:\n")
    f.write("-" * 80 + "\n")
    for model_info in best_models:
        f.write(f"{model_info['name'].upper()}:\n")
        f.write(f"  mAP50: {model_info['mAP50']:.3f}\n")
        f.write(f"  mAP50-95: {model_info['mAP50-95']:.3f}\n")
        f.write(f"  Precision: {model_info['precision']:.3f}\n")
        f.write(f"  Recall: {model_info['recall']:.3f}\n")
        f.write(f"  Model path: {model_info['model_path']}\n\n")
    
    f.write(f"BEST MODEL FOR FALL DETECTION: {best_model['name'].upper()}\n")
    f.write(f"Best mAP50: {best_model['mAP50']:.3f}\n")
    f.write(f"Recommended for accident detection\n")

# Export best model
print(f"\n7. Exporting best fall detection model ({best_model['name']})...")
best_model_instance = YOLO(best_model['model_path'])
best_model_instance.export(format="onnx")

print("\n=== ULTRA-ACCURATE TRAINING COMPLETED ===")
print(f"Best fall detection model: {best_model['name']} with mAP50: {best_model['mAP50']:.3f}")
print(f"Results saved to: {results_file}")
print("This model is optimized for maximum sensitivity to falling people!")
print("Ready for real-time accident detection!") 