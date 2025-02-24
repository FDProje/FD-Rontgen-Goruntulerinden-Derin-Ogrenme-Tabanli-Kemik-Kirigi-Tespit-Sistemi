import os
import json
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# Rastgele tohum ayarı (tekrarlanabilirlik ve güvenilirlik için)
random.seed(42)

# Dosya dizinleri
TRAIN_DIR = r"C:\Users\asyao\PycharmProjects\MICROFRACTURES\My-First-Project-1\train"
MASK_FILE = os.path.join(TRAIN_DIR, "_annotations.coco.json")
HEALTHY_DIR = r"C:\Users\asyao\Downloads\Femur_Normal\Femur_Normal"
OUT_DIR = r"C:\Users\asyao\PycharmProjects\MICROFRACTURES\My-First-Project-1\synetetics_final"
os.makedirs(OUT_DIR, exist_ok=True)

def load_annotations(mask_file):
    """Anotasyon dosyasındaki segmentasyon verilerini yükler."""
    try:
        with open(mask_file, 'r') as f:
            data = json.load(f)
        # Segmentasyonları çıkarıyoruz
        return [ann['segmentation'] for ann in data.get('annotations', []) if 'segmentation' in ann]
    except Exception as e:
        print(f"Hata: Anotasyon dosyası okunamadı - {e}")
        return []

def create_fracture_mask(image_shape, annotation, max_brightness):
    """Kırık maskesi oluşturur."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    for polygon in annotation[:5]:  # En fazla 5 koordinat
        points = np.array(polygon, dtype=np.int32).reshape((-1, 2))
        cv2.fillPoly(mask, [points], color=255)

    # Maskenin yoğunluğunu ayarlıyoruz, doğal olması için
    fracture_intensity = int(max_brightness * random.uniform(0.15, 0.25))
    mask = cv2.GaussianBlur(mask, (5,5), 0) * fracture_intensity // 255
    return cv2.subtract(mask, np.random.randint(0, 15, mask.shape, dtype=np.uint8))

def calculate_biomechanical_stress(bone_mask):
    """Kemik üzerindeki stres kırığı için uygun olabilecek bölgelerin dağılımını hesaplar."""
    distance_map = cv2.distanceTransform(bone_mask, cv2.DIST_L2, 5) # Mesafeleri normalize ederek
    return cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX)

def select_optimal_fracture_spot(stress_map):
    """En uygun kırık noktasını seçer."""
    y, x = np.unravel_index(np.argmax(stress_map), stress_map.shape)
    # Stres haritası üzerinden en yüksek stres noktasına odaklanıyoruz
    return y - stress_map.shape[0]//4, x - stress_map.shape[1]//4

def apply_fracture_to_bone(healthy_img, fracture_mask):
    """Kırık maskesini kemiğe uygular."""
    # Maskeyi kemik görüntüsüne entegre ediyoruz
    blended = cv2.addWeighted(healthy_img, 0.7, fracture_mask, 0.3, 0)

    # Trabeküler yapıyı koruyarak maskeyi daha doğal hale getiriyoruz
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    return cv2.morphologyEx(blended, cv2.MORPH_OPEN, kernel)

def process_images():
    annotations = load_annotations(MASK_FILE)
    if not annotations:
        print("Hata: Geçerli anotasyon bulunamadı!")
        return

    healthy_images = [f for f in os.listdir(HEALTHY_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in healthy_images:
        img_path = os.path.join(HEALTHY_DIR, filename)
        original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if original_img is None:
            print(f"Uyarı: {filename} okunamadı")
            continue

        # Kemik segmentasyonu
        _, bone_mask = cv2.threshold(original_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Biomekanik analiz
        stress_map = calculate_biomechanical_stress(bone_mask)
        max_stress = np.max(stress_map)

        # Kırık maskesi oluşturuluyor
        selected_annotation = random.choice(annotations)
        fracture_mask = create_fracture_mask(original_img.shape, selected_annotation, max_stress)

        # Kırık için en uygun konum belirleniyor
        y, x = select_optimal_fracture_spot(stress_map)
        height, width = fracture_mask.shape
        roi = original_img[y:y+height, x:x+width]

        if roi.shape == fracture_mask.shape:
            # Kırık maskesi kemik görüntüsüne uygulanıyor
            original_img[y:y+height, x:x+width] = apply_fracture_to_bone(roi, fracture_mask)

        output_path = os.path.join(OUT_DIR, f"SYN_{filename}")
        cv2.imwrite(output_path, original_img)
        print(f"Oluşturuldu: {output_path}")

if __name__ == "__main__":
    # Ana işlem fonksiyonunu çalıştır
    process_images()
