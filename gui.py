import sys
import cv2
import easyocr
import json
import os
from datetime import datetime
from ultralytics import YOLO
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QDialog, QSpinBox)
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
        self.last_detected_plate = None  # Son algılanan plaka
        self.last_plate_time = None  # Son algılanan plakanın zamanı
        
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
                            # Aynı plakayı tekrar okumamış mı kontrol et
                            if plate_text != self.last_detected_plate:
                                detected_plate = plate_text
                                self.last_detected_plate = plate_text
                                self.last_plate_time = datetime.now()
                                self.save_to_json(plate_text)
                            else:
                                # Aynı plaka 2 dakikadan fazla süredir algılanıyorsa sıfırla
                                if self.last_plate_time and (datetime.now() - self.last_plate_time).total_seconds() > 120:
                                    self.last_detected_plate = None
                                    self.last_plate_time = None
                except:
                    pass
        
        return detected_plate
    
    def is_valid_plate(self, text):
        """Türk plaka formatını kontrol et"""
        text = text.replace(" ", "").strip()
        # Yalnızca harfler ve rakamları kabul et, semboller kabul etme
        cleaned_text = ''.join(c for c in text if c.isalnum())
        
        if cleaned_text != text:  # Sembol var kontrolü
            return False
        
        # Türk plaka şablonları: SSHHSS, SSHHHSS, SSHHHSSS, SSHHSSS, SSHHSSSS, SSHHHSSSS, SSHSS, SSHSSS, SSHSSSS
        # S: Sayı (0-9), H: Harf (A-Z)
        valid_patterns = [
            (2, 2, 2),  # SSHHSS (2 sayı, 2 harf, 2 sayı)
            (2, 3, 2),  # SSHHHSS (2 sayı, 3 harf, 2 sayı)
            (2, 3, 3),  # SSHHHSSS (2 sayı, 3 harf, 3 sayı)
            (2, 2, 3),  # SSHHSSS (2 sayı, 2 harf, 3 sayı)
            (2, 2, 4),  # SSHHSSSS (2 sayı, 2 harf, 4 sayı)
            (2, 3, 4),  # SSHHHSSSS (2 sayı, 3 harf, 4 sayı)
            (2, 1, 2),  # SSHSS (2 sayı, 1 harf, 2 sayı)
            (2, 1, 3),  # SSHSSS (2 sayı, 1 harf, 3 sayı)
            (2, 1, 4),  # SSHSSSS (2 sayı, 1 harf, 4 sayı)
        ]
        
        for num1, letters, num2 in valid_patterns:
            total_length = num1 + letters + num2
            if len(cleaned_text) == total_length:
                # İlk kısım rakam
                if not cleaned_text[:num1].isdigit():
                    continue
                # Orta kısım harf
                if not cleaned_text[num1:num1+letters].isalpha():
                    continue
                # Son kısım rakam
                if not cleaned_text[num1+letters:].isdigit():
                    continue
                return True
        
        return False
    
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
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["Tarih", "Saat", "Plaka Numarası", "Düzenle", "Sil"])
        self.history_table.setColumnWidth(0, 120)
        self.history_table.setColumnWidth(1, 120)
        self.history_table.setColumnWidth(2, 150)
        self.history_table.setColumnWidth(3, 100)
        self.history_table.setColumnWidth(4, 100)
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
            for idx, record in enumerate(reversed(data)):
                row = self.history_table.rowCount()
                self.history_table.insertRow(row)
                
                # Tarih ve saati ayır
                datetime_str = record['zaman']
                date_part, time_part = datetime_str.split(' ')
                
                self.history_table.setItem(row, 0, QTableWidgetItem(date_part))
                self.history_table.setItem(row, 1, QTableWidgetItem(time_part))
                self.history_table.setItem(row, 2, QTableWidgetItem(record['plaka_no']))
                
                # Orijinal indeks hesapla (reversed olduğu için)
                original_idx = len(data) - 1 - idx
                
                # Düzenle butonu
                edit_button = QPushButton("Düzenle")
                edit_button.clicked.connect(lambda checked, i=original_idx: self.edit_plate_record(i))
                self.history_table.setCellWidget(row, 3, edit_button)
                
                # Sil butonu
                delete_button = QPushButton("Sil")
                delete_button.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
                delete_button.clicked.connect(lambda checked, i=original_idx: self.delete_plate_record(i))
                self.history_table.setCellWidget(row, 4, delete_button)
        except:
            pass
    
    def edit_plate_record(self, index):
        """Belirtilen indeksteki plaka kaydını düzenle"""
        if not os.path.exists(JSON_FILE):
            return
        
        try:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)
            
            # İndeksi kontrol et
            if 0 <= index < len(data):
                old_plate = data[index]['plaka_no']
                
                # Düzenleme dialogunu aç
                dialog = QDialog(self)
                dialog.setWindowTitle("Plaka Düzenle")
                dialog.setGeometry(100, 100, 300, 150)
                
                layout = QVBoxLayout()
                
                # Etiket ve input
                label = QLabel(f"Yeni plaka numarasını girin:")
                layout.addWidget(label)
                
                from PyQt6.QtWidgets import QLineEdit
                input_field = QLineEdit()
                input_field.setText(old_plate)
                layout.addWidget(input_field)
                
                # Kaydet ve İptal butonları
                button_layout = QHBoxLayout()
                save_button = QPushButton("Kaydet")
                cancel_button = QPushButton("İptal")
                
                def save_changes():
                    new_plate = input_field.text().strip().upper()
                    
                    # Plaka formatını kontrol et
                    if not self.detection_thread.is_valid_plate(new_plate):
                        from PyQt6.QtWidgets import QMessageBox
                        QMessageBox.warning(dialog, "Hata", "Geçersiz plaka formatı!")
                        return
                    
                    # JSON'u güncelle
                    data[index]['plaka_no'] = new_plate
                    with open(JSON_FILE, 'w') as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    
                    # Tabloyu yenile
                    self.refresh_history()
                    dialog.close()
                
                save_button.clicked.connect(save_changes)
                cancel_button.clicked.connect(dialog.close)
                
                button_layout.addWidget(save_button)
                button_layout.addWidget(cancel_button)
                layout.addLayout(button_layout)
                
                dialog.setLayout(layout)
                dialog.exec()
        except Exception as e:
            print(f"Hata: {e}")
    
    def delete_plate_record(self, index):
        """Belirtilen indeksteki plaka kaydını sil"""
        if not os.path.exists(JSON_FILE):
            return
        
        try:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)
            
            # İndeksi kontrol et
            if 0 <= index < len(data):
                deleted_plate = data[index]['plaka_no']
                data.pop(index)
                
                # JSON dosyasını güncelle
                with open(JSON_FILE, 'w') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                
                # Tabloyu yenile
                self.refresh_history()
        except Exception as e:
            print(f"Hata: {e}")
    
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
