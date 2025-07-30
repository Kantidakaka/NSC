from ultralytics import YOLO
import cv2
import os
import telegram  # ไลบรารีสำหรับส่งข้อความผ่าน Telegram bot
import asyncio

# ตั้งค่า Telegram bot
bot_token = "8203830707:AAFuv19itIA-eH5_WxWo9B5h1fzkE0TKQsQ"  # ใส่ API Token ที่ได้จาก BotFather
chat_id = "8097087173"  # ใส่ chat_id ของผู้ที่ต้องการรับข้อความแจ้งเตือน

# ฟังก์ชั่นส่งข้อความแจ้งเตือนผ่าน Telegram
def send_telegram_alert(message):
    bot = telegram.Bot(token=bot_token)
    try:
        bot.send_message(chat_id=chat_id, text=message)
        print(f"Sent alert: {message}")
    except Exception as e:
        print(f"Error: {e}")

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

    # ตรวจสอบการล้มและค่า confidence
    fall_detected = False
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                print(f"Class: {model.names[cls]}, Confidence: {conf:.2f}")  # ดูค่า class และ confidence

                # เงื่อนไขแจ้งเตือนเมื่อความเชื่อมั่น >= 0.6 และตรวจจับเป็น "fall"
                if conf >= 0.67:
                    class_name = model.names[cls]
                    if class_name == "Fall":  # ตรวจสอบว่าคลาสคือ "fall"
                        message = f"ALERT: FALL DETECTED! Confidence: {conf:.2f}"

                        async def send_telegram_alert(message):
                            bot = telegram.Bot(token=bot_token)
                            try:
                                # ใช้ await เพื่อรอผลการส่งข้อความ
                                await bot.send_message(chat_id=chat_id, text=message)
                                print(f"Sent alert: {message}")
                            except Exception as e:
                                print(f"Error: {e}")
                            # เรียกใช้ฟังก์ชั่น
                        async def main():
                            await send_telegram_alert("Fall Detected")

                        if __name__ == "__main__":
                            asyncio.run(main())  # ใช้ asyncio.run ในการเรียกใช้งานฟังก์ชั่น async


    # แสดงกรอบภาพที่มีการตรวจจับ
    cv2.imshow('Fall Detection - Real-time', annotated_frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")






