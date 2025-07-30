from ultralytics import YOLO
import os

# Check available models
print("Available trained models:")
if os.path.exists("runs/detect"):
    for folder in os.listdir("runs/detect"):
        weights_path = f"runs/detect/{folder}/weights"
        if os.path.exists(weights_path):
            print(f"\nFolder: {folder}")
            for file in os.listdir(weights_path):
                if file.endswith('.pt'):
                    print(f"  - {file}")

# Load the best model
model_path = "runs/detect/train3/weights/best.pt"  # Update this path
if os.path.exists(model_path):
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    # Validate the model
    print("\nValidating model...")
    results = model.val()
    print(f"Validation Results:")
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")
    print(f"Precision: {results.box.mp:.3f}")
    print(f"Recall: {results.box.mr:.3f}")
    
    # Show class-wise metrics
    print(f"\nClass-wise metrics:")
    for i, name in enumerate(model.names):
        print(f"  {name}: mAP50 = {results.box.map50:.3f}")
else:
    print(f"Model not found at: {model_path}") 