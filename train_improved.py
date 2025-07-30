from roboflow import Roboflow
from ultralytics import YOLO
import csv
import os

# Download dataset from Roboflow
print("Downloading dataset...")
rf = Roboflow(api_key="QFkOxliBmlCDsQQvPrzy")
project = rf.workspace("labels-acgyh").project("falldataset-53lvn")
dataset = project.version(3).download("yolov8")

data_yaml_path = dataset.location + "/data.yaml"

# Load a larger model for better accuracy
print("Loading YOLOv8 medium model for better accuracy...")
model = YOLO("yolov8m.pt")  # Using medium model instead of nano

# Improved training parameters for better accuracy
print("Starting improved training...")
results = model.train(
    data=data_yaml_path,
    epochs=200,              # More epochs for better learning
    batch=4,                 # Slightly larger batch size
    imgsz=640,              # Keep image size
    patience=30,            # More patience to avoid early stopping
    save_period=10,         # Save every 10 epochs
    verbose=True,
    
    # Improved learning parameters
    lr0=0.001,              # Lower initial learning rate
    lrf=0.01,               # Lower final learning rate
    momentum=0.937,         # Standard momentum
    weight_decay=0.0005,    # Weight decay for regularization
    
    # Data augmentation for better generalization
    augment=True,           # Enable augmentation
    degrees=10.0,           # Rotation augmentation
    translate=0.1,          # Translation augmentation
    scale=0.5,              # Scale augmentation
    shear=2.0,              # Shear augmentation
    perspective=0.0,        # Perspective augmentation
    flipud=0.0,             # Vertical flip (usually not good for fall detection)
    fliplr=0.5,             # Horizontal flip
    mosaic=1.0,             # Mosaic augmentation
    mixup=0.1,              # Mixup augmentation
    
    # Validation settings
    val=True,               # Enable validation
    plots=True,             # Generate plots
)

print("Training completed!")

# Save detailed metrics for each saved epoch
runs_dir = "runs/detect/train"  # This will be the new training folder
results_csv = os.path.join(runs_dir, "results.csv")

if os.path.exists(results_csv):
    print("Saving detailed metrics...")
    with open(results_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        for epoch in range(9, 200, 10):  # Every 10 epochs (0-based index)
            if epoch < len(rows):
                row = rows[epoch]
                metrics_path = os.path.join(runs_dir, f"metrics_epoch_{epoch+1}.txt")
                with open(metrics_path, "w", encoding="utf-8") as f:
                    f.write(f"Epoch: {epoch+1}\n")
                    f.write("=" * 50 + "\n")
                    for k, v in row.items():
                        f.write(f"{k}: {v}\n")
    print("Metrics saved successfully!")

# Final validation
print("\nPerforming final validation...")
final_results = model.val()
print(f"Final mAP50: {final_results.box.map50:.3f}")
print(f"Final mAP50-95: {final_results.box.map:.3f}")
print(f"Final Precision: {final_results.box.mp:.3f}")
print(f"Final Recall: {final_results.box.mr:.3f}")

# Export model
print("Exporting model...")
model.export(format="onnx")
print("Training and export completed!") 