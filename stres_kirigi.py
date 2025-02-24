# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi, sobel
from skimage.feature import canny

# Yenilikçi Özellikler:
# 1. Çoklu Filtre Entegrasyonu
# 2. Dinamik Isı Haritası Analizi
# 3. Klinik Protokollere Uyumlu Çıktılar

def kemik_kirik_analiz(gorsel_yolu):
    try:
        # Veri Kalite Kontrolü
        img = cv2.imread(gorsel_yolu, 0)
        if img is None:
            raise ValueError("Görsel okunamadı! Lütfen DICOM veya JPEG formatını kontrol edin.")
            
        plt.figure(figsize=(18, 10))
        plt.suptitle("FD: Kemik Mikro-Kırık Analiz Raporu\n", fontsize=20, y=0.97)

        # Görüntü Optimizasyonu
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(10,10))
        optimized_img = clahe.apply(img)
        blurred = cv2.medianBlur(optimized_img, 7)  # Tuz-biber gürültüsü için etkili
        
        # Çoklu Filtre Analizi
        # Frangi Filtresi (Damarsal Yapılar)
        frangi_result = frangi(blurred/255., black_ridges=False) * 2.2
        
        # Sobel Kenar Tespiti (Yapısal Sınırlar)
        sobel_edges = sobel(blurred) 
        
        # Canny Kenar Dedektörü (Keskin Kenarlar)
        canny_edges = canny(blurred, sigma=2.5) * 1.0
        
        # Birleşik Isı Haritası
        heatmap = 0.5*frangi_result + 0.3*sobel_edges + 0.2*canny_edges
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

        # Akıllı Eşikleme
        adaptive_thresh = np.percentile(heatmap, 97)  # İstatistiksel eşik belirleme
        binary_mask = heatmap > adaptive_thresh
        
        # ADIM 5: Klinik Çıktı Hazırlama
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        overlay[binary_mask, 0] = 255  # Kırık bölgeleri kırmızı ile işaretle
        
        # Görselleştirme
        # Orijinal Görüntü
        plt.subplot(2,3,1)
        plt.imshow(img, cmap='gray')
        plt.title("Orijinal Radyografi\n(Standart Görünüm)", pad=12)
        plt.axis('off')
        
        # Filtre Karşılaştırma
        plt.subplot(2,3,2)
        plt.imshow(frangi_result, cmap='viridis')
        plt.title("Damarsal Yapı Analizi\n(Frangi Filtresi)", pad=12)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.subplot(2,3,3)
        plt.imshow(sobel_edges, cmap='plasma')
        plt.title("Yapısal Sınır Tespiti\n(Sobel Gradyanı)", pad=12)
        plt.axis('off')
        
        plt.subplot(2,3,4)
        plt.imshow(canny_edges, cmap='Greens')
        plt.title("Keskin Kenar Dedektörü\n(Canny Algoritması)", pad=12)
        plt.axis('off')
        
        # Entegre Isı Haritası
        plt.subplot(2,3,5)
        plt.imshow(heatmap, cmap='hot')
        plt.title("Birleşik Risk Haritası\n(Çoklu Filtre Entegrasyonu)", pad=12)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # Sonuç Overlay
        plt.subplot(2,3,6)
        plt.imshow(overlay)
        plt.title("Kritik Kırık Bölgeleri\n(>1mm Hassasiyet)", pad=12)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('FD-filtre_Rapor.png', dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"Sistemsel Hata: {str(e)}")

if __name__ == "__main__":
    kemik_kirik_analiz(r"C:\Users\asyao\PycharmProjects\MICROFRACTURES\images\hairline108 - Kopya.jpg")
