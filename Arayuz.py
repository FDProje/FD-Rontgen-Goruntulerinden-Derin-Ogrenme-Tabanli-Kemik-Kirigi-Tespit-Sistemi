import tkinter as tk
from tkinter import PhotoImage, filedialog, Toplevel, Label, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from click import command
from ultralytics import YOLO
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image

# YOLO modelini yükleyin
model = YOLO(r"C:\Users\kuzey\PycharmProjects\Fractures2\best.pt")

# EfficientNet model class
class EfficientNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super(EfficientNet, self).__init__()
        self.backbone = models.efficientnet_b7(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

checkpoint_path = r'C:\Users\kuzey\PycharmProjects\Fractures2\best_checkpoint.pth'
classification_model = EfficientNet()

def apply_sobel_filter(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    return sobel

def detect_fractures_with_yolov8(sobel_image, model):
    sobel_bgr = cv2.cvtColor(sobel_image, cv2.COLOR_GRAY2BGR)
    results = model(sobel_bgr)
    labeled_image = results[0].plot()
    return cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)

def on_upload_click():
    # Dosya seçici açılır
    file_path = filedialog.askopenfilename(filetypes=[("Resim Dosyaları", "*.png *.jpg *.jpeg *.bmp *.gif")])
    if not file_path:
        return

    try:
        # Resmi işleyin
        sobel_image = apply_sobel_filter(file_path)
        yolo_image = detect_fractures_with_yolov8(sobel_image, model)

        # İşlenmiş resmi yeni bir pencerede gösterin
        show_processed_image(file_path, sobel_image, yolo_image)
    except Exception as e:
        messagebox.showerror("Error", f"Bir hata oluştu: {e}")


def detect_fracture_type(image_path, model):
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    img_tensor = img_tensor.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)

    _, predicted_class = torch.max(outputs, 1)

    class_names = {
        1: "Comminuted",
        2: "Compression Crush",
        3: "Hairline",
        4: "Impacted",
        5: "Longitudinal",
        6: "Oblique",
        7: "Spiral"
    }
    return class_names.get(predicted_class.item() + 1, "Unknown")


def show_processed_image(file_path, sobel_image, yolo_image):
    def on_upload_click_2():
        file_path = filedialog.askopenfilename(filetypes=[("Resim Dosyaları", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if not file_path:
            return

        try:
            sobel_image = apply_sobel_filter(file_path)
            yolo_image = detect_fractures_with_yolov8(sobel_image, model)
            fracture_type = detect_fracture_type(file_path, classification_model)  # Classify fracture type
            show_processed_image(file_path, sobel_image, yolo_image)
            fracture_label["text"] = f"Kırık/Çatlak Çeşidi: {fracture_type}"
            new_window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Bir hata oluştu: {e}")

    def exit_click():
        root.destroy()
        new_window.destroy()

    fracture_type = detect_fracture_type(file_path, classification_model)  # Classify fracture type

    new_window = Toplevel()
    new_window.title("X-Ray Kırık & Çatlak Tespit Sistemi")
    new_window.geometry("1920x1080")
    new_window.configure(bg="#a4b0f5")

    original_image = cv2.imread(file_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    original_image_pil = Image.fromarray(original_image_rgb).resize((240, 240))
    sobel_image_pil = Image.fromarray(sobel_image).resize((240, 240))
    yolo_image_pil = Image.fromarray(yolo_image).resize((240, 240))

    original_image_tk = ImageTk.PhotoImage(original_image_pil)
    sobel_image_tk = ImageTk.PhotoImage(sobel_image_pil)
    yolo_image_tk = ImageTk.PhotoImage(yolo_image_pil)

    original_label = Label(new_window, image=original_image_tk)
    original_label.image = original_image_tk
    original_label.grid(row=0, column=1, padx=10, pady=20)

    sobel_label = Label(new_window, image=sobel_image_tk)
    sobel_label.image = sobel_image_tk
    sobel_label.grid(row=0, column=2, padx=10, pady=20)

    yolo_label = Label(new_window, image=yolo_image_tk)
    yolo_label.image = yolo_image_tk
    yolo_label.grid(row=0, column=3, padx=10, pady=20)

    original_text_label = Label(new_window, text="Orijinal Resim", font=("Arial", 12, "bold"))
    original_text_label.grid(row=1, column=1, pady=(5, 10))
    original_text_label.configure(bg="#a4b0f5")

    sobel_text_label = Label(new_window, text="Sobel Filtreli Resim", font=("Arial", 12, "bold"))
    sobel_text_label.grid(row=1, column=2, pady=(5, 10))
    sobel_text_label.configure(bg="#a4b0f5")

    yolo_text_label = Label(new_window, text="Kemik Kırığı İşaretlenmiş Resim", font=("Arial", 12, "bold"))
    yolo_text_label.grid(row=1, column=3, pady=(5, 10))
    yolo_text_label.configure(bg="#a4b0f5")

    upload_button_2 = tk.Button(
        new_window, text="RESİM YÜKLE", bg="#e63946", fg="white", font=("Arial", 14, "bold"), command=on_upload_click_2,
    )
    upload_button_2.grid(row=2, column=1, pady=(5, 10))

    fracture_label = tk.Label(
        new_window, text=f"Kırık/Çatlak Çeşidi: {fracture_type}", bg="black", fg="white", font=("Arial", 14, "bold")
    )
    fracture_label.grid(row=2, column=2, pady=(5, 10))

    exit_button = tk.Button(
        new_window, text="ÇIKIŞ", bg="#e63946", fg="white", font=("Arial", 14, "bold"), command=exit_click,
    )
    exit_button.grid(row=2, column=3, pady=(5, 10))

# Ana pencere oluşturma
root = tk.Tk()
root.title("X-Ray Kırık & Çatlak Tespit Sistemi")
root.geometry("1920x1080")
root.configure(bg="#a4b0f5")

# Pencereyi tam ekran başlatma
root.attributes("-fullscreen", True)

# Üst çubuk (Frame) oluşturma
top_bar = tk.Frame(root, bg="#a4b0f5")
top_bar.pack(fill="x", side="top")

# Sol üst köşeye resim ekleme
image_path_sol = r"C:\Users\kuzey\PycharmProjects\Fractures2\Arayüz_Iconları\Sol_Üst.png"  # Resminizin tam yolu
img_sol = PhotoImage(file=image_path_sol)
image_label_sol = tk.Label(top_bar, image=img_sol, bg="#a4b0f5")
image_label_sol.pack(side="left", padx=10, pady=5)

# Sağ üst köşeye resim ekleme
image_path_sag = r"C:\Users\kuzey\PycharmProjects\Fractures2\Arayüz_Iconları\Sağ_Üst.png"  # Resminizin tam yolu
img_sag = PhotoImage(file=image_path_sag)
image_label_sag = tk.Label(top_bar, image=img_sag, bg="#a4b0f5")
image_label_sag.pack(side="right", padx=10, pady=5)

# Logoyu merkeze yerleştirme
canvas = tk.Canvas(root, width=700, height=400, bg="#a4b0f5", highlightthickness=0)
canvas.pack(pady=0)
logo_path = r"C:\Users\kuzey\PycharmProjects\Fractures2\Arayüz_Iconları\Logo.png"
logo_img = PhotoImage(file=logo_path)
canvas.create_image(350, 200, image=logo_img)

# Başlık kısmı (logonun altına)
header_frame = tk.Frame(root, bg="#2b2a45")  # Lacivert şerit için bir çerçeve oluştur
header_frame.pack(fill="x", pady=(10, 0))  # Lacivert çerçeve tam genişlikte

header = tk.Label(
    header_frame, text="X-Ray Kırık & Çatlak Tespit Sistemi",
    bg="#2b2a45", fg="#f5f5dc", font=("Comic Sans", 50, "bold")  # Krem renk ve kalın yazı
)
header.pack(pady=5)

# Gizlilik politikası metni
privacy_label = tk.Label(
    root, text="GİZLİLİK POLİTİKASI VE HİZMET ŞARTLARI",
    bg="#a4b0f5", fg="#2b2a45", font=("Arial", 12)
)
privacy_label.pack(pady=10)

privacy_checkbox = tk.Checkbutton(
    root, text="GİZLİLİK ŞARTLARINI VE HİZMET POLİTİKASINI OKUDUM VE ANLADIM",
    bg="#a4b0f5", fg="#2b2a45", font=("Arial", 10), anchor="w"
)
privacy_checkbox.pack(pady=5)

# Resim yükleme butonu
upload_button = tk.Button(
    root, text="RESİM YÜKLE", bg="#e63946", fg="white", font=("Arial", 14, "bold"),
    command=on_upload_click
)
upload_button.pack(pady=20)

# Sürüm metni
version_label = tk.Label(
    root, text="versiyon 1.1", bg="#a4b0f5", fg="#2b2a45", font=("Arial", 8)
)
version_label.pack(side="bottom", pady=0)

# Tam ekrandan çıkmak için 'Esc' tuşuna basıldığında pencereyi kapatma
def exit_fullscreen(event):
    root.attributes("-fullscreen", False)

# Klavye kısayolu ekleme
root.bind("<Escape>", exit_fullscreen)

# Pencereyi çalıştırma
root.mainloop()
