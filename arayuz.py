import os
import tkinter as tk
from io import BytesIO
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as f
from PIL import Image, ImageTk
from keras import backend as K
from skimage.filters import frangi
from torchvision import models, transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from ultralytics import YOLO

import easy_qt

# =============== Medikal Konfigürasyon/yapılandırma ===============
# Burada, farklı kırık türleri ve diğer sabit bilgiler tanımlanıyor.
CLASS_INFO = {
    0: {"name": "Comminuted"},
    1: {"name": "Compression Crush"},
    2: {"name": "Hairline"},
    3: {"name": "Impacted"},
    4: {"name": "Longitudinal"},
    5: {"name": "Oblique"},
    6: {"name": "Spiral"}
}
ROI_SIZE = (256, 256)  # İlgi alanı boyutu (Region Of Interest)

# =============== Model Tanımlamaları ===============
# Bu bölümde farklı makine öğrenmesi modelleri tanımlanıyor.

# Basit konvolüsyonel sinir ağı, PyTorch Lightning ile tanımlanmıştır.
class ConvolutionalNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # İlk konvolüsyon katmanı: 3 giriş kanalından 6 çıkış kanala
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        # İkinci konvolüsyon katmanı: 6'dan 16'ya
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        # Tam bağlantılı katmanlar
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, 2)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 54 * 54)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        # Çıkış için log softmax kullanıyoruz
        return f.log_softmax(self.fc4(x), dim=1)

# EfficientNet modeli, transfer öğrenme ile kullanılmak üzere ayarlanmıştır.
class EfficientNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super(EfficientNet, self).__init__()
        # Pretrained ağırlıklarla EfficientNet_b7 kullanıyoruz
        self.backbone = models.efficientnet_b7(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = self.backbone.classifier[1].in_features
        # Çıkış katmanını, medikal sınıflandırma için yeniden tanımlıyoruz
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# Özel TensorFlow kayıp fonksiyonu (HairlineLoss) tanımlaması
class HairlineLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='HairlineLoss'):
        super().__init__(name=name)
        self.reduction = reduction

    def call(self, y_true, y_pred):
        # Kök farkların karesinin ortalaması hesaplanır
        return K.mean(K.square(y_true - y_pred), axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'reduction': self.reduction})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(reduction=config['reduction'], name=config['name'])

# TensorFlow UNet modelini saran wrapper sınıfı
class TensorFlowUNetWrapper:
    def __init__(self, model_path, input_size=ROI_SIZE):
        # TensorFlow versiyon uyumluluğunu kontrol et
        self.verify_dependencies()
        print(f"TF Version: {tf.__version__}")

        # GPU sorunları varsa CPU kullanılmasını sağlıyoruz
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        try:
            # Modeli dosyadan yükle
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={'HairlineLoss': HairlineLoss},
                compile=False
            )
            self.model.trainable = False  # Eğitim dışı modda çalışacak
            self.input_size = input_size
        except Exception as e:
            raise RuntimeError(f"Model yükleme hatası: {str(e)}")

    def verify_dependencies(self):
        # Gerekli kütüphane versiyonlarını kontrol et
        required = {
            'TensorFlow': '2.12.0',
            'PyTorch': '1.13.1',
            'OpenCV': '4.7.0'
        }

        try:
            import tensorflow as tf
            assert tf.__version__ == required['TensorFlow']
            # Diğer kütüphaneler için benzer kontroller eklenebilir.
        except AssertionError as e:
            self.show_error(f"Versiyon uyumsuzluğu: {str(e)}")
            self.root.destroy()

    def predict(self, pil_image):
        # Modelin beklentilerine uygun hale getirmek için PIL görüntüsünü numpy dizisine çeviriyoruz
        img = pil_image.resize(self.input_size)
        img_array = np.array(img, dtype=np.float32)

        # Eğer alfa kanalı varsa, 3 kanala indiriyoruz
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]

        # Modelin beklentisine göre normalize ediyoruz ([-1,1] aralığı)
        img_array = (img_array / 127.5) - 1.0

        return self.model.predict(np.expand_dims(img_array, axis=0))

    def show_error(self, param):
        # Hata mesajı göstermek için placeholder metot
        pass

# Yükleme fonksiyonu: Tüm modelleri belleğe alıyoruz.
def load_models(device):
    models_dict = {}
    # İkili sınıflandırıcıyı yüklüyoruz
    binary = ConvolutionalNetwork()
    checkpoint = torch.load(r"C:\Users\asyao\PycharmProjects\FD\models\binary.pt", map_location=device)
    binary.load_state_dict(checkpoint['state_dict'])
    models_dict['binary'] = binary.eval().to(device)

    # Sınıflandırma modeli: EfficientNet kullanılıyor
    classification_model = EfficientNet(pretrained=True, num_classes=7)
    checkpoint = torch.load(r"C:\Users\asyao\PycharmProjects\FD\models\best_checkpoint.pth", map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model_dict = classification_model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(filtered_dict)
    classification_model.load_state_dict(model_dict)
    models_dict['type'] = classification_model.eval().to(device)

    # YOLOv8 detektörü yüklüyoruz
    models_dict['yolo'] = YOLO(r"C:\Users\asyao\PycharmProjects\FD\models\yolo.pt")
    return models_dict

# =============== Kırık Analiz Uygulaması Arayüzü ===============
# Bu sınıf, medikal kırık tespiti için GUI arayüzünü oluşturur.
class FractureAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FD: Kırık Tespit Sistemi")
        self.root.attributes('-fullscreen', True)
        self.setup_styles()

        # Cihazı belirliyoruz (GPU varsa kullanılıyor)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = load_models(self.device)
        self.current_image = None
        self.type_label = None
        self.prob_label = None

        # UNet segmentasyon modelini yüklüyoruz
        self.load_unet_model(r"C:\Users\asyao\Downloads\final_model2 (1).keras")
        # Alternatif olarak DeepLabV3'ü placeholder olarak başlatıyoruz
        self.init_unet()

        self.show_entrance()

    def init_unet(self):
        """Placeholder segmentasyon modeli (DeepLabV3) başlatılıyor."""
        self.unet_model_placeholder = deeplabv3_resnet50(pretrained=True).eval().to(self.device)

    def load_unet_model(self, model_path):
        try:
            # Model dosyasının varlığını kontrol et
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"UNet modeli bulunamadı: {model_path}")

            self.unet_model = TensorFlowUNetWrapper(model_path)
            print("UNet modeli başarıyla yüklendi")
        except Exception as e:
            print(f"UNet yükleme hatası: {str(e)}")
            # Hata durumunda DeepLabV3'e geçiyoruz
            self.unet_model = None
            self.init_unet()

    def setup_styles(self):
        # Arayüzde kullanılacak renkler tanımlanıyor
        self.colors = {
            'background': "#ffffff",
            'primary': "#2d2d2d",
            'danger': "#d32f2f",
            'success': "#388e3c",
            'warning': "#f57c00"
        }

    def show_entrance(self):
        # Açılış ekranı oluşturuluyor
        easy_qt.clear_screen(self.root)
        easy_qt.create_image(self.root, r"C:\Users\asyao\PycharmProjects\FD\assets\Enterance.png", 0, 0)
        easy_qt.create_button_with_image(self.root, "assets/button.png", 688, 888, self.main_load_image)
        easy_qt.create_button_with_image(self.root, "assets/close.png", 38, 42, self.root.destroy)
        easy_qt.create_button_with_image(self.root, "assets/contact.png", 231, 1027, self.open_contact)
        easy_qt.create_button_with_image(self.root, "assets/language_option.png", 1817, 50, self.change_language)

    def apply_filter(self, filter_type):
        if self.current_image is None:
            return

        img_np = np.array(self.current_image)

        # Farklı filtreler uygulanıyor; her biri için ilgili açıklama satırları:
        if filter_type == "sobel":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            filtered = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            img_np = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        elif filter_type == "canny":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_np = cv2.Canny(gray, 50, 150)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        elif filter_type == "heatmap":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_np = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        else:  # Orijinal görüntü
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Filtrelenmiş görüntüyü arayüzde güncelliyoruz
        self.update_filtered_display(img_np)

    def show_main_interface(self):
        # Ana arayüz oluşturuluyor
        easy_qt.clear_screen(self.root)
        easy_qt.create_image(self.root, r"C:\Users\asyao\PycharmProjects\FD\assets\main.png", 0, 0)

        # Orijinal ve YOLO ile işaretlenmiş görüntüler için etiketler
        self.original_label = easy_qt.create_label(self.root, "", 1, "black", 80, 228)
        self.yolo_label = easy_qt.create_label(self.root, "", 1, "black", 744, 228)

        # Rapor bölgesi: Tanısal bilgilerin gösterildiği alan
        self.report_frame = tk.Frame(self.root, bg=self.colors['background'], bd=2, relief="groove")
        self.report_frame.place(x=1373, y=240, width=480, height=530)

        # Arayüz düğmeleri
        easy_qt.create_button_with_image(self.root, "assets/download.png", 220, 879, self.save_results)
        easy_qt.create_button_with_image(self.root, "assets/load.png", 1358, 856, self.load_image)
        easy_qt.create_button_with_image(self.root, "assets/refreash.png", 50, 879, self.show_entrance)
        easy_qt.create_button_with_image(self.root, "assets/trash.png", 400, 879, self.clear_analysis)
        easy_qt.create_button_with_image(self.root, "assets/exit.png", 1730, 859, self.root.destroy)

        # Filtreleme düğmeleri
        easy_qt.create_button_with_image(self.root, "assets/sobel.png", 44, 155, lambda: self.apply_filter("sobel"))
        easy_qt.create_button_with_image(self.root, "assets/canny.png", 244, 155, lambda: self.apply_filter("canny"))
        easy_qt.create_button_with_image(self.root, "assets/heatmap.png", 444, 155,
                                         lambda: self.apply_filter("heatmap"))
        easy_qt.create_button(self.root, "Orijinal Görüntü", 17, "white", "black",
                              lambda: self.apply_filter("original"), 249, 761)

        # Saçık (hairline) kırık arayüzüne geçiş
        easy_qt.create_button_with_image(self.root, "assets/stres.png", 1362, 956, self.show_hairline_interface)

    def run_unet_segmentation(self, image):
        if self.unet_model is None:
            print("UNet modeli yüklenemedi.")

        try:
            # Giriş görüntüsü boyutu bilgisi
            print(f"Giriş görüntü boyutu: {image.size}")
            mask = self.unet_model.predict(image)
            print(f"Maske boyutu: {mask.shape}")

            # Çıktı formatına göre düzenleme
            if mask.ndim == 4:
                mask = mask[0]
            if mask.shape[-1] == 1:
                mask = mask[..., 0]

            # Belirli eşik değerinin üzerindeki pikseller beyaz olarak işaretleniyor
            return Image.fromarray((mask > 0.65).astype(np.uint8) * 255)
        except Exception as e:
            print(f"Segmentasyon hatası: {str(e)}")
            return image

    def show_hairline_interface(self):
        # Saçık kırık tespiti arayüzü oluşturuluyor
        easy_qt.clear_screen(self.root)
        easy_qt.create_image(self.root, r"C:\Users\asyao\PycharmProjects\FD\assets\Hairline.png", 0, 0)

        # Orijinal ve işlenmiş görüntüler için etiketler
        self.hairline_original_label = easy_qt.create_label(self.root, "", 1, "black", 623, 265)
        self.hairline_processed_label = easy_qt.create_label(self.root, "", 1, "black", 1376, 167)

        # Filtre kontrol düğmeleri
        easy_qt.create_button_with_image(self.root, "assets/frangi.png", 48, 561, self.apply_frangi_filter)
        easy_qt.create_button_with_image(self.root, "assets/clache.png", 51, 665, self.apply_clahe)
        easy_qt.create_button(self.root, "Orijinal Görüntü", 17, "white", "black",
                              lambda: self.update_hairline_display(self.current_image), 634, 315)

        # Tanısal bilgilerin gösterileceği etiketler
        easy_qt.create_label(self.root, "Alan: 0px²", 20, "white", 330, 289)
        easy_qt.create_label(self.root, "Kırık Sayısı: 0", 20, "white", 330, 320)
        easy_qt.create_label(self.root, "Ortalama Uzunluk: 0", 20, "white", 330, 350)
        easy_qt.create_label(self.root, "Derece: 0", 20, "white", 330, 380)

        # Navigasyon düğmeleri
        easy_qt.create_button_with_image(self.root, "assets/back.png", 1767, 714, self.show_main_interface)
        easy_qt.create_button_with_image(self.root, "assets/download.png", 220, 879, self.save_results)
        easy_qt.create_button_with_image(self.root, "assets/load.png", 1358, 856, self.load_image)
        easy_qt.create_button_with_image(self.root, "assets/refreash.png", 50, 879, self.show_entrance)
        easy_qt.create_button_with_image(self.root, "assets/trash.png", 400, 879, self.clear_analysis)
        easy_qt.create_button_with_image(self.root, "assets/exit.png", 1730, 859, self.root.destroy)

        # Eğer görüntü yüklü ise, arayüz otomatik olarak güncelleniyor
        if self.current_image:
            self.update_hairline_display(self.current_image)
            self.update_hairline_info()

    def update_hairline_display(self, image):
        # Saçık kırık arayüzünde orijinal görüntü güncelleniyor
        orig_img = image.resize((400, 400))
        orig_photo = ImageTk.PhotoImage(orig_img)
        self.hairline_original_label.configure(image=orig_photo)
        self.hairline_original_label.image = orig_photo

        # Segmentasyon uygulayarak işlenmiş görüntü güncelleniyor
        segmentation = self.run_unet_segmentation(image)
        self.update_processed_view(segmentation)

    def update_processed_view(self, processed_img):
        proc_img = processed_img.resize((400, 400))
        proc_photo = ImageTk.PhotoImage(proc_img)
        self.hairline_processed_label.configure(image=proc_photo)
        self.hairline_processed_label.image = proc_photo

    def update_hairline_info(self):
        # Burada, saçık kırık arayüzü için tanısal bilgilerin güncellenmesi yapılabilir.
        if hasattr(self, 'yolo_result') and hasattr(self, 'fracture_type'):
            # Örnek: Güven skoru gibi bilgilerin güncellenmesi
            pass

    def apply_frangi_filter(self):
        if self.current_image is None:
            return
        img_np = np.array(self.current_image.convert('L'))
        normalized = img_np / 255.0
        frangi_img = frangi(normalized) * 255
        processed_img = Image.fromarray(frangi_img.astype(np.uint8))
        self.update_processed_view(processed_img)
        self.update_hairline_info()

    def apply_clahe(self):
        if self.current_image is None:
            return
        img_np = np.array(self.current_image.convert('L'))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(img_np)
        processed_img = Image.fromarray(clahe_img)
        self.update_processed_view(processed_img)
        self.update_hairline_info()

    def main_load_image(self):
        # Ana arayüzü göster ve ardından görüntüyü yükle
        self.show_main_interface()
        self.load_image()

    def load_image(self):
        # Kullanıcıdan görüntü dosyası seçimi alınıyor
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if path:
            self.process_image(path)

    def is_fracture(self, image_path):
        # İkili sınıflandırıcı ile kırık olup olmadığını belirle
        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.models['binary'](img_tensor)
            probs = torch.exp(output)
            fracture_prob = probs[0, 0].item()
        return fracture_prob > 0.5, fracture_prob * 100

    def run_binary_classification(self, img):
        # İkili sınıflandırıcı üzerinden kırık olasılığı hesaplanıyor
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        tensor = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.models['binary'](tensor)
        probs = torch.exp(output)
        fracture_prob = probs[0, 0].item() * 100
        return fracture_prob > 20, fracture_prob

    def run_type_classification(self, img, yolo_results):
        # Sınıflandırma modeli ile kırık tipini belirleme
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        tensor = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.models['type'](tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        sorted_probs = np.sort(probs)[::-1]
        confidence = calculate_confidence(probs, yolo_results)
        best_class = np.argmax(probs)
        return {
            'type': CLASS_INFO[best_class]['name'],
            'confidence': confidence,
            'probabilities': probs,
            'explanations': {
                'base_prob': np.max(probs) * 100,
                'margin': (sorted_probs[0] - sorted_probs[1]) * 100,
                'yolo_contribution': (np.mean(yolo_results['confidence']).item() * 100
                                      if len(yolo_results['confidence']) > 0 else 0)
            }
        }

    def show_probability_chart(self, probs):
        # Olasılıkları çubuk grafik olarak gösteriyoruz
        fig, ax = plt.subplots(figsize=(4.5, 2.5))
        classes = [CLASS_INFO[i]['name'] for i in range(len(probs))]
        probabilities = probs * 100
        sorted_data = sorted(zip(classes, probabilities), key=lambda x: -x[1])
        ax.barh([x[0] for x in sorted_data], [x[1] for x in sorted_data], color=self.colors['danger'])
        ax.set_xlabel("Olasılık (%)", fontsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        chart_img = Image.open(buf)
        chart_photo = ImageTk.PhotoImage(chart_img)
        chart_label = tk.Label(self.report_frame, image=chart_photo, bg=self.colors['background'])
        chart_label.image = chart_photo
        chart_label.pack(pady=10)

    def process_image(self, path):
        try:
            # Seçilen görüntü yükleniyor ve işleniyor
            img = Image.open(path).convert("RGB")
            self.current_image = img
            update_display(self.original_label, img.resize((500, 500)))
            binary_detect, binary_prob = self.is_fracture(path)
            yolo_result = self.run_yolo_detection(img)
            has_fracture = binary_detect or len(yolo_result['boxes']) > 0
            fracture_type = {"type": "Belirsiz", "confidence": 0.0, "probabilities": []}
            self.yolo_result = yolo_result
            self.fracture_type = fracture_type
            if has_fracture:
                fracture_type = self.run_type_classification(img, yolo_result)
                if len(yolo_result['boxes']) > 0:
                    fracture_type['confidence'] = min(fracture_type['confidence'], 50)
            self.update_yolo_display(yolo_result['annotated'])
            self.generate_medical_report(
                has_fracture=has_fracture,
                binary_prob=binary_prob,
                yolo=yolo_result,
                fracture_type=fracture_type
            )
        except Exception as e:
            self.show_error(f"İşlem hatası: {str(e)}")

    def update_yolo_display(self, annotated_img):
        # YOLO tarafından işaretlenmiş görüntüyü güncelliyoruz
        yolo_img = Image.fromarray(annotated_img[..., ::-1]).resize((500, 500))
        update_display(self.yolo_label, yolo_img)

    def show_error(self, message):
        # Hata mesajlarını kullanıcıya göstermek için küçük bir pencere oluşturulur
        error_window = tk.Toplevel(self.root)
        error_window.title("Critical Error")
        error_window.geometry("400x200")

        tk.Label(error_window, text=message,
                 wraplength=380, fg="red", padx=20, pady=20).pack()

    def save_results(self):
        # Sonuçları kaydetmek için placeholder metot
        pass

    def clear_analysis(self):
        # Analiz sonuçlarını temizlemek için placeholder metot
        pass

    def open_contact(self):
        # İletişim bilgilerini göstermek için placeholder metot
        pass

    def change_language(self):
        # Dil seçenekleri değiştirilebilir, placeholder metot
        pass

    def show_yolo_findings(self, yolo):
        # YOLO tespit detayları gösterilebilir, placeholder metot
        pass

    def run_yolo_detection(self, img):
        try:
            # YOLO deteksiyonu için görüntüyü ön işleme tabi tutuyoruz
            cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            sobel_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(sobel_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(sobel_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobel_x, sobel_y)
            sobel = np.uint8(255 * sobel / np.max(sobel))
            sobel_bgr = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
            results = self.models['yolo'](sobel_bgr)
            annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            return {
                'annotated': annotated,
                'boxes': results[0].boxes.xyxy.cpu().numpy(),
                'confidence': results[0].boxes.conf.cpu().numpy()
            }
        except Exception as e:
            self.show_error(f"YOLO Hatası: {str(e)}")
            return {'annotated': np.array(img), 'boxes': [], 'confidence': []}

    def generate_medical_report(self, has_fracture, binary_prob, yolo, fracture_type):
        # Tanısal rapor arayüzü güncelleniyor
        if self.type_label:
            self.type_label.destroy()
        if self.prob_label:
            self.prob_label.destroy()

        self.type_label = tk.Label(self.root, text=f"{fracture_type.get('type', 'Belirsiz')}",
                                   font=("Helvetica", 20, "bold"), fg="white", bg="black",
                                   padx=8, pady=4)
        self.type_label.place(x=953, y=877)

        self.prob_label = tk.Label(self.root, text=f"{fracture_type.get('confidence', 0):.1f}",
                                   font=("Helvetica", 12, "bold"), fg="white", bg="black",
                                   padx=8, pady=4)
        self.prob_label.place(x=909, y=982)

        for widget in self.report_frame.winfo_children():
            widget.destroy()

        status_text = "KIRIK TESPİT EDİLDİ" if has_fracture else "KIRIK BULGU YOK"
        status_color = self.colors['danger'] if has_fracture else self.colors['success']
        tk.Label(self.report_frame, text=status_text, font=("Helvetica", 16, "bold"),
                 fg=status_color, bg=self.colors['background']).pack(pady=12, anchor=tk.NW)

        details_frame = tk.Frame(self.report_frame, bg=self.colors['background'])
        details_frame.pack(fill=tk.X, pady=8, padx=5)
        tk.Label(details_frame, text="Klinik Bulgular:", font=("Helvetica", 12, "underline"),
                 fg=self.colors['primary'], bg=self.colors['background']).pack(anchor=tk.W)

        if not has_fracture or (has_fracture and len(yolo['boxes']) == 0):
            tk.Label(details_frame, text=f"İkili Sınıflandırıcı Sonucu: %{binary_prob:.1f}",
                     font=("Helvetica", 11), fg=self.colors['primary'],
                     bg=self.colors['background']).pack(anchor=tk.W, pady=4)
        if len(yolo['boxes']) > 0:
            yolo_frame = tk.Frame(details_frame, bg=self.colors['background'])
            yolo_frame.pack(anchor=tk.W, pady=6)
            tk.Label(yolo_frame, text="Tespit Edilen Kırık Sayısı:",
                     font=("Helvetica", 11), fg=self.colors['primary'],
                     bg=self.colors['background']).pack(side=tk.LEFT)
            tk.Label(yolo_frame, text=str(len(yolo['boxes'])),
                     font=("Helvetica", 11, "bold"), fg=self.colors['danger'],
                     bg=self.colors['background']).pack(side=tk.LEFT, padx=8)

        tk.Label(self.root, text=f"%{fracture_type.get('confidence', 0):.1f}",
                 font=("Helvetica", 20, "bold"), fg="white", bg="black",
                 padx=8, pady=4).place(x=909, y=982)

        exp = fracture_type.get('explanations', {})
        details_text = (
            f"Temel Olasılık: %{exp.get('base_prob', 0):.1f}\n"
            f"Karar Marjı: %{exp.get('margin', 0):.1f}\n"
            f"Yardımcı Tespit Katkısı: %{exp.get('yolo_contribution', 0):.1f}"
        )
        tk.Label(details_frame, text=details_text, font=("Helvetica", 10),
                 fg=self.colors['primary'], bg=self.colors['background'],
                 justify=tk.LEFT).pack(anchor=tk.W, pady=8)

        if fracture_type and fracture_type.get('confidence', 0) > 0:
            type_frame = tk.Frame(details_frame, bg=self.colors['background'])
            type_frame.pack(anchor=tk.W, pady=8)
            tk.Label(type_frame, text="Tahmini Kırık Tipi:",
                     font=("Helvetica", 12), fg=self.colors['primary'],
                     bg=self.colors['background']).pack(side=tk.LEFT)
            tk.Label(type_frame, text=f"{fracture_type['type']} (%{fracture_type['confidence']:.1f})",
                     font=("Helvetica", 12, "bold"), fg=self.colors['danger'],
                     bg=self.colors['background']).pack(side=tk.LEFT, padx=8)
            self.show_probability_chart(fracture_type['probabilities'])

        tk.Label(self.report_frame,
                 text="*Bu otomatik analiz ön tanı amaçlıdır. Kesin teşhis için radyolojik değerlendirme şarttır.",
                 font=("Helvetica", 8, "italic"), fg=self.colors['danger'],
                 bg=self.colors['background']).pack(side=tk.BOTTOM, pady=5)

    def show_yolo_details(self, yolo):
        # YOLO tespit detayları için bilgi paneli oluşturulabilir
        yolo_frame = tk.Frame(self.report_frame, bg=self.colors['background'])
        yolo_frame.pack(pady=10, fill=tk.X)
        tk.Label(yolo_frame, text="Tespit Edilen Kırık Sayısı:",
                 font=("Helvetica", 10), bg=self.colors['background']).pack(side=tk.LEFT)
        tk.Label(yolo_frame, text=str(len(yolo['boxes'])),
                 font=("Helvetica", 10, "bold"), fg=self.colors['danger'],
                 bg=self.colors['background']).pack(side=tk.LEFT)

    def show_type_details(self, fracture_type):
        # Kırık tipi detayları için bilgi paneli oluşturuluyor
        type_frame = tk.Frame(self.report_frame, bg=self.colors['background'])
        type_frame.pack(pady=10, fill=tk.X)
        tk.Label(type_frame, text="Tahmini Kırık Tipi:",
                 font=("Helvetica", 12), bg=self.colors['background']).pack(side=tk.LEFT)
        tk.Label(type_frame, text=f"{fracture_type['type']} (%{fracture_type['confidence']:.1f})",
                 font=("Helvetica", 12, "bold"), fg=self.colors['danger'],
                 bg=self.colors['background']).pack(side=tk.LEFT, padx=8)

    def update_filtered_display(self, img_np):
        # Filtrelenmiş görüntünün ekranda güncellenmesi için metot (placeholder)
        pass

# Küçük yardımcı fonksiyon: Tkinter etiketine güncellenmiş görüntü ekler
def update_display(label, image):
    photo = ImageTk.PhotoImage(image)
    label.configure(image=photo)
    label.image = photo

# Küresel fonksiyon: Hesaplama işlemleri (juryya gösterebileceğimiz detaylı hesaplamalar)
def calculate_confidence(probs, yolo_results):
    sorted_probs = np.sort(probs)[::-1]
    base_conf = sorted_probs[0] * 100
    margin = (sorted_probs[0] - sorted_probs[1]) * 100
    yolo_conf = yolo_results['confidence']
    num_detections = len(yolo_results['boxes'])
    weights = {'base': 0.6, 'margin': 0.2, 'yolo': 0.2}
    yolo_score = np.mean(yolo_conf).item() * 100 if num_detections > 0 else 0
    margin_modifier = min(1.0, margin / 20)
    confidence = (
            weights['base'] * base_conf +
            weights['margin'] * base_conf * margin_modifier +
            weights['yolo'] * yolo_score
    )
    confidence = 100 * (1 / (1 + np.exp(-0.1 * (confidence - 85))))
    return min(95, max(5, confidence))

# Programın ana çalıştırma noktası
if __name__ == "__main__":
    root = tk.Tk()
    app = FractureAnalysisApp(root)
    root.mainloop()
