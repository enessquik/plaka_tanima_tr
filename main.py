import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime
import os

# ----------------------------------------------------
# PARAMETRELERİ AYARLA
# ----------------------------------------------------

# Plaka tespiti için önceden eğitilmiş YOLOv8 modelini yükle
YOLO_MODEL_PATH = "best.pt"

# Karakter tanıma (OCR) için EasyOCR'ı başlat
reader = easyocr.Reader(['en'])

# Tanıma işleminin yapılacağı kaynak (resim veya kamera)
SOURCE_TYPE = "webcam"
IMAGE_PATH = "plaka_ornek.jpg" # Sadece SOURCE_TYPE "image" ise kullanılır
WEBCAM_ID = 0 # Sadece SOURCE_TYPE "webcam" ise kullanılır

# JSON kayıt dosyası
JSON_FILE = "plaka_kayitlari.json"

# ----------------------------------------------------
# FONKSİYONLAR
# ----------------------------------------------------

def is_valid_plate(text):
    """
    Türkiye plaka formatına (örn: 06 ABC 123) göre kontrol yapar.
    Basit bir kontrol, daha gelişmiş kurallar eklenebilir.
    """
    text = text.replace(" ", "").strip()
    return len(text) >= 5 and len(text) <= 9 and text.isalnum()

def save_to_json(plate_text):
    """
    Tespit edilen plakayı, tarih ve saat bilgisiyle JSON dosyasına kaydeder.
    """
    if not os.path.exists(JSON_FILE) or os.stat(JSON_FILE).st_size == 0:
        data = []
    else:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)

    new_record = {
        "plaka_no": plate_text,
        "zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    data.append(new_record)

    with open(JSON_FILE, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def process_frame(frame, model):
    """
    Tek bir kareyi işler: Plakayı tespit eder, okur, gösterir ve kaydeder.
    """
    results = model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            
            plate_roi = frame[y1:y2, x1:x2]
            ocr_result = reader.readtext(plate_roi)

            plate_text = ""
            if ocr_result:
                best_match = max(ocr_result, key=lambda x: x[2])
                plate_text = best_match[1].upper().replace(" ", "").replace(".", "").replace("-", "")

                if is_valid_plate(plate_text):
                    print(f"Tespit Edilen Plaka: {plate_text}")
                    save_to_json(plate_text)
                    
                    # Tespit edilen plakayı pembe bir kare içine al
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # (255, 0, 255) R, G, B değeriyle pembe
                    cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                else:
                    # Geçersiz plaka ise farklı bir renkte çiz
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
    return frame

# ----------------------------------------------------
# ANA ÇALIŞTIRMA KISMI
# ----------------------------------------------------

model = YOLO(YOLO_MODEL_PATH)

if SOURCE_TYPE == "image":
    frame = cv2.imread(IMAGE_PATH)
    if frame is not None:
        processed_frame = process_frame(frame, model)
        # Resim modunda sonucu göster
        cv2.imshow("Plaka Tanima", processed_frame)
        cv2.waitKey(0)
    else:
        print(f"Hata: {IMAGE_PATH} dosyası bulunamadı.")
        
elif SOURCE_TYPE == "webcam":
    # Kamera akışını başlat
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print("Hata: Kamera açılamadı.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = process_frame(frame, model)
            
            # Canlı akışı göster
            cv2.imshow("Plaka Tanima - Canli Yayin", processed_frame)
            
            # 'q' tuşuna basıldığında döngüyü sonlandır
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.release()
    
cv2.destroyAllWindows()
