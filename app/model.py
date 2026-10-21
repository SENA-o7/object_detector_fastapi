# ============================================================================
# model.py : Colab'da eğittiğim modeli burada yüklüyor ve tek bir görüntü için
# tahmin yapıyorum. Mimariyi ve ön işlemeyi eğitimdekiyle (bbox_regressor.py,
# train.py) birebir aynı tuttum; farklı olursa ağırlıklar yüklenmez ya da
# tahminler bozulur.
# ============================================================================
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import Dropout, Identity, Linear, Module, ReLU, Sequential, Sigmoid
from torchvision.models import resnet50

# Eğitimin sonunda kaydettiğim ağırlıklar (state_dict)
AGIRLIK_YOLU = Path(__file__).resolve().parent.parent / "model" / "detector_weights.pth"

# Eğitimde LabelEncoder sınıfları alfabetik sıraladı (büyük harf önce): 0, 1, 2.
# le.pickle dosyasını açmak için scikit-learn kurmamak adına sırayı buraya yazdım.
SINIFLAR = ["Motorbikes", "airplanes", "schooner"]

# Eğitimde kullandığım ön işleme değerleri (ImageNet ortalama / standart sapma)
INPUT_SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ObjectDetector(Module):
    # ResNet50 gövdesinin üstüne bir kutu (regressor) ve bir sınıf (classifier) kafası koydum

    def __init__(self, baseModel, numClasses):
        super(ObjectDetector, self).__init__()
        self.baseModel = baseModel
        self.numClasses = numClasses
        in_features = baseModel.fc.in_features

        self.regressor = Sequential(
            Linear(in_features, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 4),
            Sigmoid(),
        )
        self.classifier = Sequential(
            Linear(in_features, 512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            ReLU(),
            Dropout(),
            Linear(512, self.numClasses),
        )
        self.baseModel.fc = Identity()

    def forward(self, x):
        features = self.baseModel(x)
        return (self.regressor(features), self.classifier(features))


def model_yukle():
    # weights=None verdim: ImageNet ağırlıklarını internetten indirmeme gerek yok,
    # gövde dahil tüm ağırlıklar benim .pth dosyamdan geliyor
    model = ObjectDetector(resnet50(weights=None), len(SINIFLAR))
    # state_dict sadece tensör içerdiği için güvenli olan weights_only=True ile yüklüyorum
    model.load_state_dict(torch.load(AGIRLIK_YOLU, map_location="cpu", weights_only=True))
    model.eval()  # Değerlendirme moduna aldım: Dropout kapalı
    return model


# Modeli servis açılırken bir kez yüklüyorum (her istekte diskten okumamak için)
_model = model_yukle()


def tahmin_et(goruntu_baytlari):
    # 1) Yüklenen dosyanın baytlarını görüntüye çeviriyorum (OpenCV BGR okur)
    image = cv2.imdecode(np.frombuffer(goruntu_baytlari, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Dosya bir görüntü olarak okunamadı")
    (h, w) = image.shape[:2]  # Orijinal boyutu saklıyorum (kutuyu geri ölçeklemek için)

    # 2) Eğitimdeki gibi ön işliyorum: RGB, 224x224, 0-1 aralığı, normalize, (1,3,224,224)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
    tensor = ((tensor - MEAN) / STD).unsqueeze(0)

    # 3) Modeli çalıştırıyorum
    with torch.inference_mode():
        predBBox, predLogits = _model(tensor)

    # 4) Logitleri olasılığa çeviriyorum; en yüksek olan tahminim
    olasiliklar = torch.softmax(predLogits, dim=1)[0].tolist()
    indeks = int(np.argmax(olasiliklar))

    # 5) 0-1 aralığındaki kutuyu orijinal piksel koordinatlarına geri ölçekliyorum
    x1, y1, x2, y2 = predBBox[0].tolist()

    return {
        "predicted_label": SINIFLAR[indeks],
        "probability": round(olasiliklar[indeks], 4),
        "probabilities": {s: round(p, 4) for s, p in zip(SINIFLAR, olasiliklar)},
        "bbox": [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)],
    }
