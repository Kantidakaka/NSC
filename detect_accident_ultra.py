from ultralytics import YOLO
import cv2
import os

# ตรวจสอบโมเดล
model_path = "runs/detect/train3/weights/best.pt"
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()

print(f"Loading model from: {model_path}")
model = YOLO(model_path)
print("Model loaded successfully!\n")

# เปิดกล้อง
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully!\n")

# ตั้งชื่อหน้าต่าง
window_name = "Fall Detection - Real-time"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame from camera")
        break

    # รัน inference
    results = model(frame)  # สามารถใส่ conf=0.xx ได้ตามต้องการ

    # ดึงผลการตรวจจับจากเฟรมแรก
    for box in results[0].boxes:
        cls  = int(box.cls[0])
        name = model.names[cls]
        conf = float(box.conf[0])

        # ถ้าไม่ใช่ fall → ข้าม
        if name != "fall":
            continue

        # ถ้าใช่ fall → วาดกรอบแดง + label
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"fall {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # แสดงผลลัพธ์
    cv2.imshow(window_name, frame)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")

