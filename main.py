import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer
from ultralytics import YOLO

class YOLODesktopApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎯 YOLOv8 Object Detection App")
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #ffffff;
                font-family: Segoe UI, sans-serif;
                font-size: 15px;
            }
            QPushButton {
                background-color: #1E88E5;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QLabel {
                border: 2px solid #1E88E5;
                border-radius: 10px;
                margin: 10px;
            }
        """)

        self.model = YOLO('yolov8n.pt')

        self.image_label = QLabel(self)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Buttons
        self.webcam_button = QPushButton("Start Webcam")
        self.stop_button = QPushButton("Stop")
        self.upload_image_button = QPushButton("Detect from Image")
        self.upload_video_button = QPushButton("Detect from Video")

        # Layouts
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.webcam_button)
        btn_layout.addWidget(self.stop_button)
        btn_layout.addWidget(self.upload_image_button)
        btn_layout.addWidget(self.upload_video_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # Connect buttons
        self.webcam_button.clicked.connect(self.start_webcam)
        self.stop_button.clicked.connect(self.stop)
        self.upload_image_button.clicked.connect(self.detect_from_image)
        self.upload_video_button.clicked.connect(self.detect_from_video)

        # Variables
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.image_label.clear()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        results = self.model(frame)
        annotated_frame = results[0].plot()
        self.show_frame(annotated_frame)

    def detect_from_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.jpeg *.png)")
        if file_path:
            image = cv2.imread(file_path)
            results = self.model(image)
            annotated = results[0].plot()
            self.show_frame(annotated)

    def detect_from_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi *.mov)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.timer.start(30)

    def show_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size())
        self.image_label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    from PyQt5.QtCore import Qt
    app = QApplication(sys.argv)
    window = YOLODesktopApp()
    window.resize(1000, 800)
    window.show()
    sys.exit(app.exec_())
