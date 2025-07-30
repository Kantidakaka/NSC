from ultralytics import YOLO
import cv2
import os

# ตรวจสอบว่ามีไฟล์โมเดลหรือไม่
model_path = "runs/detect/train3/weights/best.pt"  # เส้นทางของไฟล์โมเดลที่ฝึกแล้ว
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()

# โหลดโมเดลที่ฝึกแล้ว
print(f"Loading model from: {model_path}")
model = YOLO(model_path)
print("Model loaded successfully!")

# เปิดกล้อง
cap = cv2.VideoCapture(0)  # ใช้กล้องตัวแรก ถ้าไม่สามารถเปิดได้ ให้ลองใช้กล้องที่สอง (index 1)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully!")

# อ่านกรอบจากกล้องและทำการตรวจจับการล้ม
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame from camera")
        break

    # Run inference กับโมเดล YOLO
    results = model(frame)  # ทำการประมวลผลภาพด้วยโมเดล

    # แสดงผลลัพธ์การตรวจจับ
    annotated_frame = results[0].plot()  # แสดงกรอบและข้อความบนภาพ

    # แสดงกรอบภาพที่มีการตรวจจับ
    cv2.imshow('Fall Detection - Real-time', annotated_frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")


