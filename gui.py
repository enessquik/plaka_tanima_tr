import sys
import cv2
import easyocr
import json
import os
from datetime import datetime
from ultralytics import YOLO
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
import numpy as np

# Parametreler
YOLO_MODEL_PATH = "license_plate_detector.pt"
JSON_FILE = "plaka_kayitlari.json"

class PlateDetectionThread(QThread):
    """Plaka tespiti için ayrı thread"""
    frame_processed = pyqtSignal(object, str)  # frame, detected_plate
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.cap = None
        
        # OCR ve YOLO modellerini yükle
        os.environ['MPLBACKEND'] = 'Agg'
        self.reader = easyocr.Reader(['tr'])
        self.model = YOLO(YOLO_MODEL_PATH)
        
    def run(self):
        """Kamerayı aç ve frame'leri işle"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Kamera açılamadı")
            return
            
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Frame'i işle
            detected_plate = self.process_frame(frame)
            
            # Signal gönder
            self.frame_processed.emit(frame, detected_plate)
            
            # 30 FPS
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
    
    def process_frame(self, frame):
        """Plakayı tespit et ve oku"""
        detected_plate = ""
        
        results = self.model(frame)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                
                plate_roi = frame[y1:y2, x1:x2]
                
                try:
                    ocr_result = self.reader.readtext(plate_roi)
                    if ocr_result:
                        texts = []
                        for detection in ocr_result:
                            text = detection[1].upper().strip()
                            confidence = detection[2]
                            if confidence > 0.1:
                                if len(text) == 1:
                                    if text == "I" or text == "l":
                                        text = "I"
                                texts.append(text)
                        
                        plate_text = " ".join(texts).replace(" ", "").replace(".", "").replace("-", "")
                        
                        # I -> T düzeltmesi
                        plate_text = plate_text.replace("I", "T")
                        plate_text = plate_text.replace("l", "T")
                        
                        if self.is_valid_plate(plate_text):
                            detected_plate = plate_text
                            self.save_to_json(plate_text)
                except:
                    pass
        
        return detected_plate
    
    def is_valid_plate(self, text):
        """Plaka formatını kontrol et"""
        text = text.replace(" ", "").strip()
        return (len(text) >= 5 and len(text) <= 9 and 
                any(c.isalpha() for c in text) and any(c.isdigit() for c in text))
    
    def save_to_json(self, plate_text):
        """Plakayı JSON'a kaydet"""
        if not os.path.exists(JSON_FILE) or os.stat(JSON_FILE).st_size == 0:
            data = []
        else:
            try:
                with open(JSON_FILE, 'r') as f:
                    data = json.load(f)
            except:
                data = []
        
        # Aynı plakayı 5 saniye içinde iki kez kaydetme
        if data:
            last_record = data[-1]
            if last_record['plaka_no'] == plate_text:
                last_time = datetime.strptime(last_record['zaman'], "%Y-%m-%d %H:%M:%S")
                if (datetime.now() - last_time).total_seconds() < 5:
                    return
        
        new_record = {
            "plaka_no": plate_text,
            "zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        data.append(new_record)
        
        with open(JSON_FILE, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    
    def stop(self):
        """Thread'i durdur"""
        self.running = False

class PlakaTanimaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        # Detection thread'ini başlat
        self.detection_thread = PlateDetectionThread()
        self.detection_thread.frame_processed.connect(self.update_frame)
        self.detection_thread.start()
    
    def init_ui(self):
        """GUI öğelerini oluştur"""
        self.setWindowTitle("Plaka Tanıma Sistemi")
        self.setGeometry(100, 100, 1200, 700)
        
        # Ana widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Tab widget
        self.tabs = QTabWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        main_widget.setLayout(main_layout)
        
        # Kamera sekmesi
        self.create_camera_tab()
        
        # Geçmiş sekmesi
        self.create_history_tab()
        
        # Stilini ayarla
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QTableWidget {
                background-color: white;
                gridline-color: #cccccc;
            }
        """)
    
    def create_camera_tab(self):
        """Kamera sekmesini oluştur"""
        camera_widget = QWidget()
        layout = QVBoxLayout()
        
        # Video label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        layout.addWidget(self.video_label)
        
        # Tespit edilen plaka
        info_layout = QHBoxLayout()
        info_label = QLabel("Tespit Edilen Plaka:")
        info_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.plate_display = QLabel("---")
        self.plate_display.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.plate_display.setStyleSheet("color: #4CAF50;")
        info_layout.addWidget(info_label)
        info_layout.addWidget(self.plate_display)
        info_layout.addStretch()
        layout.addLayout(info_layout)
        
        camera_widget.setLayout(layout)
        self.tabs.addTab(camera_widget, "Kamera")
    
    def create_history_tab(self):
        """Geçmiş sekmesini oluştur"""
        history_widget = QWidget()
        layout = QVBoxLayout()
        
        # Tablo
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(3)
        self.history_table.setHorizontalHeaderLabels(["Tarih", "Saat", "Plaka Numarası"])
        self.history_table.setColumnWidth(0, 150)
        self.history_table.setColumnWidth(1, 150)
        self.history_table.setColumnWidth(2, 200)
        layout.addWidget(self.history_table)
        
        # Yenile butonu
        refresh_button = QPushButton("Yenile")
        refresh_button.clicked.connect(self.refresh_history)
        layout.addWidget(refresh_button)
        
        history_widget.setLayout(layout)
        self.tabs.addTab(history_widget, "Geçmiş")
        
        # İlk yükleme
        self.refresh_history()
    
    def update_frame(self, frame, detected_plate):
        """Kamera frame'ini güncelle"""
        if detected_plate:
            self.plate_display.setText(detected_plate)
            self.plate_display.setStyleSheet("color: #4CAF50; background-color: #e8f5e9; padding: 5px;")
            # Geçmiş sekmesini yenile
            self.refresh_history()
        else:
            self.plate_display.setText("---")
            self.plate_display.setStyleSheet("color: #999; background-color: transparent;")
        
        # Frame'i QPixmap'e çevir
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Ölçekle
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaledToWidth(800, Qt.TransformationMode.SmoothTransformation)
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def refresh_history(self):
        """Geçmiş tablosunu yenile"""
        self.history_table.setRowCount(0)
        
        if not os.path.exists(JSON_FILE):
            return
        
        try:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)
            
            # Tersten sırala (en yeni en üstte)
            for record in reversed(data):
                row = self.history_table.rowCount()
                self.history_table.insertRow(row)
                
                # Tarih ve saati ayır
                datetime_str = record['zaman']
                date_part, time_part = datetime_str.split(' ')
                
                self.history_table.setItem(row, 0, QTableWidgetItem(date_part))
                self.history_table.setItem(row, 1, QTableWidgetItem(time_part))
                self.history_table.setItem(row, 2, QTableWidgetItem(record['plaka_no']))
        except:
            pass
    
    def closeEvent(self, event):
        """Uygulamayı kapat"""
        self.detection_thread.stop()
        self.detection_thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = PlakaTanimaGUI()
    gui.show()
    sys.exit(app.exec())
