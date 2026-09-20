# Yüz İfadesi Tanıma (FER2013): CNN, Residual ve CBAM Dikkat Mekanizması

**FER2013** veri setinde 48×48 gri tonlamalı yüz görüntülerinden 7 duygu sınıfını (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) tahmin eden üç ardışık deneyin karşılaştırması. Her yeni deney, önceki sonucu iyileştirmeyi hedefler.

Kod: [`Facial-Expression-Prediction.ipynb`](Facial-Expression-Prediction.ipynb) (TensorFlow / Keras)

---

## Sonuçlar (test seti: 7.178 görüntü)

| # | Deney | Parametre | Test doğruluğu | Macro F1 | Weighted F1 |
|---|---|---:|:---:|:---:|:---:|
| 1 | VGG tarzı CNN (baseline) | 10,54 M | 66.48% | 0.6336 | 0.6625 |
| 2 | Residual + kanal dikkati + class weight | 5,32 M | 62.83% | 0.5811 | 0.6134 |
| 3 | **CLAHE + bilateral filtre + Residual + CBAM** | 13,62 M | **68.00%** | **0.6616** | **0.6776** |

Sınıf bazında F1 skorları:

| Sınıf | Test örneği | Deney 1 | Deney 2 | Deney 3 |
|---|---:|:---:|:---:|:---:|
| angry | 958 | 0.5840 | 0.5585 | 0.6071 |
| disgust | 111 | 0.5635 | 0.4985 | 0.6598 |
| fear | 1024 | 0.4518 | 0.3112 | 0.4975 |
| happy | 1774 | 0.8789 | 0.8629 | 0.8732 |
| neutral | 1233 | 0.6153 | 0.6036 | 0.6403 |
| sad | 1247 | 0.5722 | 0.4685 | 0.5690 |
| surprise | 831 | 0.7696 | 0.7643 | 0.7839 |

Gözlemler:

- **`happy`** ve **`surprise`** her deneyde en yüksek F1'e sahip sınıflar; **`fear`** ve **`sad`** en düşük F1'e sahip olanlardır.
- **Deney 2**, class weight ile `disgust` recall'unu 0.77'ye çıkardı, fakat bunun bedeli düşük precision (0.37) ve `fear` recall'unun 0.21'e düşmesi oldu; genel doğruluk baseline'ın altında kaldı. `%70+` hedefi hiçbir deneyde aşılmadı.
- **Deney 3**, önişleme ve CBAM ile baseline'a göre +1.5 puan (66.48% → 68.00%) ile en iyi sonucu verdi.

---

## Veri Seti

[Kaggle: `msambare/fer2013`](https://www.kaggle.com/datasets/msambare/fer2013) (Kaggle API çıktısında lisans `DbCL-1.0` olarak görünür)

| Özellik | Değer |
|---|---|
| Görüntü boyutu | 48×48, gri tonlama |
| Sınıf sayısı | 7 |
| Eğitim klasörü | 28.709 görüntü |
| Test klasörü | 7.178 görüntü |

Doğrulama seti, eğitim klasöründen ayrılır:

| Deney | Doğrulama oranı | Eğitim / Doğrulama |
|---|:---:|:---:|
| 1 | %15 | 24.406 / 4.303 |
| 2 | %20 | 22.968 / 5.741 |
| 3 | (aşağıdaki nota bakın) | – |

---

## Deneyler

```
FER2013 (48x48 gri)
   │
   ├─► Deney 1: ImageDataGenerator (augmentation) ─► VGG tarzı CNN
   │            Conv-BN-ReLU ×(2-2-3-2) ─► Dense 1024 ─► Dense 512 ─► softmax
   │
   ├─► Deney 2: + brightness aug., class weight ─► Residual bloklar + kanal dikkati (SE tarzı)
   │            AdamW + warmup'lı cosine decay
   │
   └─► Deney 3: CLAHE + bilateral filtre ─► Residual bloklar + CBAM (kanal + uzamsal dikkat)
                Adam + cosine annealing (30 epoch'luk döngüler)
                          │
                          ▼
              Test seti: classification_report + confusion matrix
```

| | Deney 1 | Deney 2 | Deney 3 |
|---|---|---|---|
| Mimari | 4 blok VGG tarzı CNN | 4 residual blok, 3. blokta SE tarzı dikkat | 9 residual blok + CBAM, Global Avg Pooling |
| Önişleme | Yeniden ölçekleme (÷255) | ÷255 | CLAHE (clip 2.0, 8×8) + bilateral filtre |
| Augmentation | Döndürme 20°, kaydırma 0.15, zoom 0.15, shear 0.15, yatay çevirme | Döndürme 25°, kaydırma 0.2, zoom 0.2, shear 0.15, parlaklık [0.8, 1.2], yatay çevirme | Döndürme 25°, kaydırma 0.2, zoom 0.2, shear 0.2, yatay çevirme |
| Optimizer | Adam | AdamW (wd 1e-4) | Adam |
| Learning rate | 1e-3 + ReduceLROnPlateau | 5e-4 + 5 epoch warmup, cosine decay | 1e-3 + cosine annealing (30 epoch döngü) |
| Batch size | 128 | 64 | 128 |
| Class weight | – | Var (`balanced`, `disgust` ≈ 9.4) | – |
| Early stopping | patience 15 (`val_loss`) | patience 20 (`val_loss`) | patience 20 (`val_loss`) |
| Çalışan epoch | 67 (erken durdu, en iyi ağırlıklar epoch 52) | 100 (en iyi epoch 97) | 100 (en iyi epoch 89) |

Ortak ayarlar: maksimum 100 epoch, `sparse_categorical_crossentropy`, mixed precision (`mixed_float16`), `ModelCheckpoint`.

---

## Kurulum ve Çalıştırma

Notebook Google Colab (A100 GPU) için yazılmıştır ve `google.colab.files` kullanır.

```bash
pip install tensorflow opencv-python-headless scikit-learn seaborn matplotlib pandas numpy kaggle
```

1. [Kaggle ayarlarından](https://www.kaggle.com/settings) bir API token (`kaggle.json`) oluşturun.
2. Notebook'u Colab'da açın ve GPU çalışma zamanı seçin.
3. Hücre çalıştığında istenen yerde `kaggle.json` dosyasını yükleyin; veri seti otomatik indirilir (`/content/FER2013`).
4. Üç deneyin her biri ayrı bir hücredir ve veriyi kendisi indirir; istediğiniz deneyi bağımsız çalıştırabilirsiniz.

Her deney sonunda model (`.keras`), sonuç JSON'u, eğitim grafikleri ve confusion matrix üretilir.

> Yerelde çalıştırırken `google.colab.files` çağrılarını kaldırıp veri yolunu (`DATA_PATH`) kendi dizininize göre ayarlayın.

---

## Notlar ve Sınırlılıklar

- **Deney 3'te doğrulama seti bağımsız değildir.** Bu deneyde `val_gen`, `train/` klasörünün tamamından (augmentation'sız) oluşturulmuştur; yani doğrulama görüntüleri aynı zamanda eğitimde de kullanılmıştır. Bu nedenle raporlanan `Best Val Accuracy (0.8086)` genelleme başarısını göstermez ve **kullanılmamalıdır**. Erken durdurma ve checkpoint seçimi de bu doğrulamaya dayandığından model seçimi ideal değildir. **Yukarıdaki test sonuçları ise ayrı `test/` klasörüne dayandığı için geçerlidir.**
- Deney 1 ve 2'de doğrulama alt kümesi, eğitimle aynı `ImageDataGenerator` (`train_datagen`) ile üretildiği için augmentation'a tabi tutulmuştur; doğrulama metrikleri bu nedenle bir miktar kötümser olabilir.
- Sonuçlar tek çalıştırma / tek tohum sonuçlarıdır; deneyler arasındaki 1-2 puanlık farklar rastgelelikten etkilenmiş olabilir.
- Deneylerde mimari, önişleme, LR programı ve class weight aynı anda değiştiği için her bileşenin bireysel katkısı ayrıştırılamaz (ablation yapılmamıştır).
- `disgust` sınıfı çok küçüktür (test setinde 111 örnek); bu sınıfın metrikleri oynaktır.
- FER2013 etiketleri gürültülüdür ve düşük çözünürlüklü görüntüler içerir; bu, elde edilebilecek doğruluğu sınırlayan bilinen bir faktördür.

## Olası Geliştirmeler

- Deney 3 için `train/` klasöründen gerçek bir doğrulama ayrımı yapıp yeniden eğitmek
- Ablation: CLAHE, CBAM ve class weight'i tek tek açıp kapatmak
- Önceden eğitilmiş (transfer learning) ağlarla karşılaştırma
- Farklı tohumlarla tekrar ve test-time augmentation / topluluk (ensemble)
- Karışıklık matrisi ve örnek tahmin görsellerini `assets/` altında README'ye eklemek

## Repo Yapısı

```
facial-expression-fer2013/
├── Facial-Expression-Prediction.ipynb
└── README.md
```
