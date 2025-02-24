# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras import layers, models, losses, callbacks
import numpy as np
import matplotlib.pyplot as plt
import pathlib

# ###################### YAPILANDIRMA ######################
RESIM_BOYUTU = (1024, 1024)
ROI_BOYUTU = (128, 128)        # odak bölgesi boyutu
BATCH_SIZE = 8
VERI_YOLU = "/veri/klasörü"     # Veri seti

# ###################### VERİ HAZIRLAMA ######################
class TibbiVeriYukleyici:
    def __init__(self, resim_klasoru, maske_klasoru):
        self.resim_yollari = sorted(list(pathlib.Path(resim_klasoru).glob("*.jpg")))
        self.maske_yollari = sorted(list(pathlib.Path(maske_klasoru).glob("*.png")))
        if len(self.resim_yollari) != len(self.maske_yollari):
            raise ValueError("Resim ve maske sayıları uyuşmuyor!")

    def _ozel_onislem(self, resim_yolu, maske_yolu):
        """Radyolojik görüntüler için ön işlemler"""
        # Görüntü yükleme
        resim = tf.image.decode_jpeg(tf.io.read_file(resim_yolu), channels=3)
        resim = tf.image.resize(resim, RESIM_BOYUTU, method='bicubic')
        resim = tf.cast(resim, tf.float32) / 255.0  # Normalizasyon

        # Maske işleme
        maske = tf.image.decode_png(tf.io.read_file(maske_yolu), channels=1)
        maske = tf.image.resize(maske, RESIM_BOYUTU, method='nearest')
        maske = tf.cast(maske > 127, tf.float32)  # İkili maskeye çevirme

        return resim, maske

    def _akilli_kirpma(self, resim, maske):
        """Kırık şüphesi olan bölgeyi otomatik seç"""
        # Maske üzerinde yoğunluk analizi
        yogunluk_haritasi = tf.reduce_sum(maske, axis=[0,1])
        x_merkez = tf.argmax(yogunluk_haritasi)
        
        # Yatayda 150px'lik kritik bölge (metafiz bölgesi)
        x_baslangic = tf.maximum(x_merkez - 75, 0)
        x_bitis = tf.minimum(x_merkez + 75, RESIM_BOYUTU[1])
        
        return resim[:, x_baslangic:x_bitis, :], maske[:, x_baslangic:x_bitis, :]

    def _veri_arttirma(self, resim, maske):
        """Tıbbi görüntüler için güvenli artırma teknikleri"""
        # Rastgele yatay çevirme
        if tf.random.uniform(()) > 0.5:
            resim = tf.image.flip_left_right(resim)
            maske = tf.image.flip_left_right(maske)

        resim = tf.image.random_contrast(resim, 0.8, 1.2)  # Kontrast ayarlama

        resim = tf.image.random_brightness(resim, 0.1) # Sınırlı parlaklık ayarı
        resim = tf.clip_by_value(resim, 0.0, 1.0)
        
        return resim, maske

    def veri_akisi(self, egitim_orani=0.8, egitim_indeksesi=None):
        """Veri setini eğitim ve test olarak ayır"""
        veri_boyutu = len(self.resim_yollari)
        egitim_boyutu = int(veri_boyutu * egitim_orani)
        
        # karıştırma ve bölme
        karisik_indeksler = tf.random.shuffle(tf.range(veri_boyutu))
        egitim_indeksleri = karisik_indeksler[:egitim_boyutu]
        test_indeksleri = karisik_indeksler[egitim_boyutu:]

        egitim_ds = tf.data.Dataset.from_tensor_slices(
            (tf.gather(self.resim_yollari, egitim_indeksesi),
            (tf.gather(self.maske_yollari, egitim_indeksesi))
        ).map(self._ozel_onislem, num_parallel_calls=tf.data.AUTOTUNE)
        .map(self._akilli_kirpma, num_parallel_calls=tf.data.AUTOTUNE)
        .map(self._veri_arttirma, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE))
        
        test_ds = ... # Benzer test seti
        
        return egitim_ds, test_ds

# ###################### MODEL MİMARİSİ ######################
class KemikKirikDetektor(models.Model):
    def __init__(self):
        super().__init__()
        
        # Encoder (Özellik Çıkarıcı)
        self.katman1 = self._conv_blok(32, 3)
        self.katman2 = self._conv_blok(64, 3)
        self.katman3 = self._conv_blok(128, 5)  # Geniş alanlı filtre
        
        # Decoder (Segmentasyon)
        self.upsample1 = layers.UpSampling2D(2)
        self.deconv1 = layers.Conv2DTranspose(64, 3, padding='same', activation='relu')
        self.upsample2 = layers.UpSampling2D(2)
        self.deconv2 = layers.Conv2DTranspose(32, 3, padding='same', activation='relu')
        
        # Çıktı Katmanı
        self.cikis = layers.Conv2D(1, 1, activation='sigmoid')
        
    def _conv_blok(self, filtre, boyut):
        """Özel konvolüsyon bloğu"""
        return tf.keras.Sequential([
            layers.Conv2D(filtre, boyut, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.2)
        ])
    
    def call(self, inputs):
        x = self.katman1(inputs)
        x = self.katman2(x)
        x = self.katman3(x)
        
        x = self.upsample1(x)
        x = self.deconv1(x)
        x = self.upsample2(x)
        x = self.deconv2(x)
        
        return self.cikis(x)

# ###################### EĞİTİM ve DEĞERLENDİRME ######################
def main():
    # veri hazırlama
    yukleyici = TibbiVeriYukleyici(f"{VERI_YOLU}/resimler", f"{VERI_YOLU}/maskeler")
    egitim_ds, test_ds = yukleyici.veri_akisi()
    
    # model oluşturma
    model = KemikKirikDetektor()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # checkpointler
    checkpoint = callbacks.ModelCheckpoint(
        "en_iyi_model.h5",
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    
    early_stop = callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True
    )
    
    # eğitim
    tarihce = model.fit(
        egitim_ds,
        validation_data=test_ds,
        epochs=50,
        callbacks=[checkpoint, early_stop]
    )
    
    # performans görselleştirme
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(tarihce.history['loss'], label='Eğitim Kaybı')
    plt.plot(tarihce.history['val_loss'], label='Doğrulama Kaybı')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(tarihce.history['precision'], label='Kesinlik')
    plt.plot(tarihce.history['recall'], label='Duyarlılık')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
