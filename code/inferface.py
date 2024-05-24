from ultralytics import YOLO

# โหลดโมเดลที่เทรนแล้ว
model = YOLO(r"C:\Users\Keen\Desktop\YOLOv8\facedetect/yolov8l-face.pt")

# ทดสอบโมเดลกับภาพ
results = model.predict(source='testface', save=True)

# แสดงผลลัพธ์
#results.show()  # เปิดภาพที่มีการตรวจจับ
