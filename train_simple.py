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

# Load YOLOv8 model (like original)
print("Loading YOLOv8 nano model...")
model = YOLO("yolov8n.pt")

# Train the model (simple settings like original)
print("Starting training...")
results = model.train(
    data=data_yaml_path,
    epochs=100,              # Like original
    batch=2,                 # Like original
    imgsz=640,              # Like original
    patience=20,            # Like original
    save_period=5,          # Save every 5 epochs (like original)
    verbose=True
)

print("Training completed!")

# Save metrics for each saved epoch (like original)
runs_dir = "runs/detect/train"  # This will be the new training folder
results_csv = os.path.join(runs_dir, "results.csv")

if os.path.exists(results_csv):
    print("Saving metrics for each 5-epoch checkpoint...")
    with open(results_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        for epoch in range(4, 100, 5):  # Every 5 epochs (0-based index)
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