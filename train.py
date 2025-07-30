from roboflow import Roboflow
from ultralytics import YOLO
import shutil
import os
import csv

# ฟังก์ชั่นในการดาวน์โหลดชุดข้อมูลจาก Roboflow
def download_dataset():
    try:
        rf = Roboflow(api_key="QFkOxliBmlCDsQQvPrzy")
        project = rf.workspace("labels-acgyh").project("falldataset-53lvn")
        dataset = project.version(3).download("yolov8")
        return dataset.location + "/data.yaml"
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

# ฟังก์ชั่นสำหรับการฝึกโมเดล YOLOv8
def train_model(data_yaml_path):
    try:
        # โหลด YOLOv8 โมเดล
        model = YOLO("yolov8s.pt")  # ใช้โมเดลขนาดกลางหรือใหญ่
        total_epochs = 100

        # ฝึกโมเดล
        results = model.train(
            data=data_yaml_path,
            epochs=100,         # 50-100 กำลังดี
            batch=4,           # CPU แนะนำ 2 หรือ 4
            imgsz=640,         # ไม่ควรเกินนี้
            patience=20,
            save_period=5,
            verbose=True,
            lr0=0.01,
            lrf=0.1,
            warmup_epochs=3,
            # loss='focal',    # ถ้าใช้ได้
        )
        return results
    except Exception as e:
        print(f"Error during training: {e}")
        return None

# ฟังก์ชั่นสำหรับการบันทึกเมตริกส์หลังการฝึก
def save_metrics(runs_dir, total_epochs):
    results_csv = os.path.join(runs_dir, "results.csv")

    if os.path.exists(results_csv):
        try:
            with open(results_csv, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                
                for epoch in range(4, total_epochs, 5):  # 0-based index, so 4, 9, 14, ...
                    if epoch < len(rows):
                        row = rows[epoch]
                        metrics_path = os.path.join(runs_dir, f"metrics_epoch_{epoch+1}.txt")
                        
                        # เขียนข้อมูลเมตริกส์
                        with open(metrics_path, "w", encoding="utf-8") as f:
                            f.write(f"Epoch: {epoch+1}\n")
                            for k, v in row.items():
                                f.write(f"{k}: {v}\n")
        except Exception as e:
            print(f"Error reading or saving metrics: {e}")
    else:
        print(f"results.csv not found in {runs_dir}. Metrics not saved.")

def main():
    data_yaml_path = download_dataset()
    if not data_yaml_path:
        return

    results = train_model(data_yaml_path)
    if not results:
        return

    runs_dir = "runs/detect/train2"
    save_metrics(runs_dir, total_epochs=100)

if __name__ == "__main__":
    main()
