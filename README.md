# Nesne Tespit Modeli – FastAPI + Docker

Google Colab'da PyTorch ile eğitilen nesne tespit modelinin (ResNet50 gövde + sınıf ve bounding box kafaları) FastAPI ile Docker konteynerinde sunulması.

- **Sınıflar:** `Motorbikes`, `airplanes`, `schooner` (Caltech-101 alt kümesi)
- **Eğitim notebook'u:** https://colab.research.google.com/drive/1dPLU-4oPdNScUad0C1W2FzxoTVgsMtwE
- **Platform:** imaj `linux/arm64` için derlenir (Apple Silicon Mac)

## Çalıştırma

```bash
git clone <bu-reponun-adresi>
cd object-detector-fastapi
docker-compose up
```

Servis `http://localhost:7001` adresinde açılır. Arayüzden denemek için: `http://localhost:7001/docs`

## Kullanım

```bash
curl -X POST -F "file=@gorsel.jpg" http://localhost:7001/predict
```

Örnek cevap:

```json
{
  "filename": "gorsel.jpg",
  "predicted_label": "Motorbikes",
  "probability": 1.0,
  "probabilities": {"Motorbikes": 1.0, "airplanes": 0.0, "schooner": 0.0},
  "bbox": [32, 22, 229, 152]
}
```

`bbox`: tahmin edilen kutunun orijinal görüntüdeki piksel koordinatları `[xmin, ymin, xmax, ymax]`.

## Dosyalar

| Dosya | Görevi |
|---|---|
| `app/model.py` | Model mimarisi, ağırlıkların yüklenmesi, ön işleme ve tahmin |
| `app/main.py` | FastAPI uç noktaları (`GET /`, `POST /predict`) |
| `model/detector_weights.pth` | Colab'da eğitilen modelin ağırlıkları (state_dict) |
| `Dockerfile` | CPU sürümü PyTorch ile Python 3.12 imajı |
| `docker-compose.yml` | arm64 platformu ve 7001 portu |
