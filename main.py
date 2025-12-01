import cv2
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime
import os

# easyocr özel hata düzeltmesi: GUI kütüphaneleri yüklü olmadığında çalışmasını sağla
os.environ['MPLBACKEND'] = 'Agg'

import easyocr
print("EasyOCR yükleniyor...")
reader = easyocr.Reader(['tr'])
print("EasyOCR hazır!")

# ----------------------------------------------------
# PARAMETRELERİ AYARLA
# ----------------------------------------------------

# Plaka tespiti için önceden eğitilmiş YOLOv8 modelini yükle
YOLO_MODEL_PATH = "license_plate_detector.pt"

# Tanıma işleminin yapılacağı kaynak (resim veya kamera)
SOURCE_TYPE = "webcam"
IMAGE_PATH = "plaka.jpg" # Sadece SOURCE_TYPE "image" ise kullanılır
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
    # En az 5 karakter (örn: 06ABC) en fazla 9 karakter (örn: 06ABC1234)
    # Karakter kombinasyonu: 2 rakam + harf + harf + harf + 1-4 rakam
    return len(text) >= 5 and len(text) <= 9 and any(c.isalpha() for c in text) and any(c.isdigit() for c in text)

def correct_ocr_text(text):
    """
    OCR hataları için post-processing düzeltmeleri yapar.
    Plaka numarası formatına göre karakterleri düzeltir.
    """
    # Yaygın OCR karışıklıkları - plakalarda I genellikle T olabilir
    corrections = {
        "I": "T",  # Dikey çizgi T olabilir
        "l": "T",  # Küçük L de T olabilir
    }
    
    result = ""
    for char in text:
        # Sadece alfabetik karakterler için düzelt
        if char.isalpha() and char in corrections:
            result += corrections[char]
        else:
            result += char
    
    return result

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
            
            plate_text = ""
            if reader is not None:
                try:
                    ocr_result = reader.readtext(plate_roi)
                    if ocr_result:
                        # Tüm OCR sonuçlarını birleştir (confidence 0.1'den yüksek olanlar)
                        texts = []
                        for detection in ocr_result:
                            text = detection[1].upper().strip()
                            confidence = detection[2]
                            if confidence > 0.1:  # Düşük confidence sonuçları filtrele
                                # Tek karakteri kontrol et: rakamsa bırak, harf-gibi görünüyorsa düzelt
                                if len(text) == 1:
                                    if text == "I" or text == "l":
                                        text = "I"
                                    elif text == "O":
                                        # Bağlamda rakam olabilir
                                        if any(c.isdigit() for c in " ".join(texts)):
                                            text = "0"
                                texts.append(text)
                        
                        plate_text = " ".join(texts).replace(" ", "").replace(".", "").replace("-", "")
                        plate_text = correct_ocr_text(plate_text)  # OCR hatalarını düzelt
                        print(f"OCR detections: {texts}")
                        print(f"Combined text: {plate_text}")
                except Exception as e:
                    print(f"OCR hatası: {e}")
                    plate_text = ""

            if plate_text and is_valid_plate(plate_text):
                print(f"Tespit Edilen Plaka: {plate_text}")
                save_to_json(plate_text)
                
                # Tespit edilen plakayı pembe bir kare içine al
                try:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # (255, 0, 255) R, G, B değeriyle pembe
                    cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                except Exception as e:
                    print(f"Resim çizim hatası: {e}")
            else:
                # Geçersiz plaka ise farklı bir renkte çiz
                try:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                except Exception as e:
                    print(f"Resim çizim hatası: {e}")
                    
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
        try:
            cv2.imshow("Plaka Tanima", processed_frame)
            cv2.waitKey(0)
        except Exception as e:
            print(f"GUI açılamadı, sonuç kaydediliyor... ({e})")
            output_path = "processed_" + IMAGE_PATH
            cv2.imwrite(output_path, processed_frame)
            print(f"İşlenen resim kaydedildi: {output_path}")
    else:
        print(f"Hata: {IMAGE_PATH} dosyası bulunamadı.")
        
elif SOURCE_TYPE == "webcam":
    # Kamera akışını başlat
    try:
        cap = cv2.VideoCapture(WEBCAM_ID)
        if not cap.isOpened():
            print(f"Uyarı: Kamera {WEBCAM_ID} açılamadı. Video dosyasını deniyorum...")
            # Fallback: video dosyasını dene
            if not os.path.exists("test_video.mp4"):
                print("Hata: Kamera ve video dosyası bulunamadı.")
                cap = None
            else:
                cap = cv2.VideoCapture("test_video.mp4")
                print("Video dosyası açıldı: test_video.mp4")
        
        if cap and cap.isOpened():
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Video sonlandı. {frame_count} kare işlendi.")
                    break
                
                processed_frame = process_frame(frame, model)
                frame_count += 1
                
                # Canlı akışı göster (GUI varsa)
                try:
                    cv2.imshow("Plaka Tanima - Canli Yayin", processed_frame)
                except:
                    pass
                
                # 'q' tuşuna basıldığında döngüyü sonlandır
                try:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print(f"Kullanıcı tarafından durduruldu. {frame_count} kare işlendi.")
                        break
                except:
                    pass
            
            cap.release()
    except Exception as e:
        print(f"Webcam/Video hatası: {e}")

try:
    cv2.destroyAllWindows()
except:
    pass
