# 🧠 Title: Real-Time Object Detection using YOLOv8 with Webcam Integration

This project demonstrates a real-time object detection system using the **YOLOv8** model, integrated with a webcam for live video feed analysis. It utilizes **Ultralytics' YOLOv8n** model for efficient and accurate detection on standard computing hardware.

# 👨‍💻 Team leader 
## Name : Apurba sarkar 
## Roll no.: 2401010280

**Video link** : https://drive.google.com/file/d/1-MMzk_2IcPGN7Kldj9D20gkfYHZyBona/view?usp=drivesdk

# 🛠️ Tech Stack

- **Model**: YOLOv8n (Ultralytics)
- **Language**: Python 3.10
- **Libraries**: Ultralytics, OpenCV, NumPy, Matplotlib
- **Platform**: Google Colab / VS Code (Local Machine)
- 
# 🚀 Features

- Real-time object detection on webcam feed
- Lightweight YOLOv8n model for high FPS on low-resource systems
- Bounding boxes with confidence scores
- Easily extensible for custom models and edge devices

# 📊 Results
FPS: ~20–30 FPS on i5, 8GB RAM laptop

Accuracy: High confidence detection of people, phones, plants, etc.

Sample Detection:

Person: 96.3%

Chair: 86.5%

Mobile Phone: 96%

Potted Plant: 82.4%

🧪 Comparative Analysis
Model	FPS	Accuracy	Hardware Requirement
YOLOv8n	✅ High	✅ High	✅ Low
YOLOv5	Medium	High	Medium
SSD	High	Low	Low
Faster R-CNN	Low	Very High	High

🔍 Limitations
Reduced accuracy in poor lighting

Dependent on webcam quality

Larger models (YOLOv8m/l/x) need better hardware

💡 Future Work
Edge deployment (e.g., Jetson Nano)

Integrating object tracking

Training with custom datasets
