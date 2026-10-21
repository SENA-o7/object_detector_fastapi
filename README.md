# Nesne Tespit Modeli: FastAPI + Docker

Bir önceki görevde Google Colab'da PyTorch ile bir nesne tespit modeli eğittim. Bu repoda o modeli **FastAPI** ile bir web servisine dönüştürüp **Docker** konteynerinde sunuyorum. Servis bir görüntü alıyor, görüntünün hangi sınıfa ait olduğunu ve nesnenin görüntüde nerede olduğunu **JSON** olarak dönüyor. Ayrıca sonucu görüntünün üzerinde kutuyla gösteren küçük bir test arayüzü hazırladım.

- **Sınıflar:** `Motorbikes`, `airplanes`, `schooner`
- **Veri seti:** [Caltech-101'in 3 sınıflık alt kümesi (Kaggle)](https://www.kaggle.com/datasets/maricinnamon/caltech101-airplanes-motorbikes-schooners)
- **Eğitim notebook'u:** https://colab.research.google.com/drive/1dPLU-4oPdNScUad0C1W2FzxoTVgsMtwE
- **Test doğruluğu (Colab):** %99.7 (333 görüntünün 332'si doğru)
- **Platform:** imaj `linux/arm64` için derleniyor (Apple Silicon M1/M2/M3/M4 Mac)

---

## Çalıştırma

```bash
git clone  https://github.com/SENA-o7/object_detector_fastapi
cd object-detector-fastapi
docker-compose up
```

Bilgisayarda sadece Docker kurulu olması yeterli. Python, PyTorch gibi başka bir şey kurmaya gerek yok. İlk derleme PyTorch'u indirdiği için birkaç dakika sürüyor.

| Adres | Ne var |
|---|---|
| http://localhost:7001 | Test arayüzü: görüntüyü sürükle bırak, sonucu kutuyla gör |
| http://localhost:7001/predict | `POST` ile görüntü gönderilen asıl uç nokta, JSON döner |
| http://localhost:7001/docs | FastAPI'nin otomatik ürettiği API belgesi ve deneme sayfası |

Kapatmak için: `docker-compose down`

## Kullanım

```bash
curl -X POST -F "file=@gorsel.jpg" http://localhost:7001/predict
```

> Windows PowerShell'de `curl` komutu aslında `Invoke-WebRequest`'in takma adı olduğu için `-F` parametresini tanımıyor. Orada `curl.exe` yazmak gerekiyor.

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

| Alan | Anlamı |
|---|---|
| `predicted_label` | Modelin tahmin ettiği sınıf |
| `probability` | Tahmin edilen sınıfın olasılığı (softmax) |
| `probabilities` | Üç sınıfın olasılıkları |
| `bbox` | Nesnenin orijinal görüntüdeki piksel koordinatları `[x1, y1, x2, y2]` |

Görüntü olmayan bir dosya gönderilirse servis `400` hatası ve `{"detail": "Dosya bir görüntü olarak okunamadı"}` cevabı dönüyor.

---

## Repo yapısı

```
object-detector-fastapi/
├── app/
│   ├── __init__.py          # app klasörünü Python paketi yapıyor (app.main import edilebilsin diye)
│   ├── main.py              # FastAPI: GET / (arayüz), POST /predict (tahmin)
│   ├── model.py             # Model mimarisi, ağırlık yükleme, ön işleme, tahmin
│   └── static/index.html    # Test arayüzü (tek dosya, HTML + CSS + JS)
├── model/
│   └── detector_weights.pth # Colab'da eğittiğim ağırlıklar (state_dict, ~95 MB)
├── requirements.txt         # Python paketleri
├── Dockerfile               # İmajın tarifi
└── docker-compose.yml       # Tek komutla çalıştırma, port ve ARM ayarı
```

Görev "sadece çıkarım için gerekli kodlar olsun" dediği için eğitim kodlarını ve eğitim çıktılarını bu repoya koymadım. Ayrıntısı aşağıdaki "Neleri bilerek koymadım" bölümünde.

---

## Neyi neden kullandım?

| Araç | Neden |
|---|---|
| **PyTorch (CPU sürümü)** | Model PyTorch ile eğitildi, ağırlıkları yüklemek için aynı kütüphane gerekiyor. Mac'te NVIDIA GPU (CUDA) olmadığı için CUDA'lı sürüm işe yaramıyor, üstelik GB'larca fazladan yer kaplıyor. Bu yüzden sadece CPU sürümünü kurdum. |
| **torchvision** | Modelin gövdesi olan `resnet50` mimarisi buradan geliyor. |
| **FastAPI** | Modeli HTTP üzerinden erişilebilir bir servise dönüştürmek için kullandım. Birkaç satırla adres tanımlanabiliyor, dosya yükleme destekleniyor, döndürülen `dict` otomatik olarak JSON'a çevriliyor ve `/docs` sayfası kendiliğinden oluşuyor. Görevin örnek reposu da FastAPI kullanıyor. |
| **uvicorn** | FastAPI bir çatı (framework), kendi başına ağı dinlemiyor. Portu dinleyip istekleri FastAPI'ye ileten asıl web sunucusu uvicorn. |
| **python-multipart** | FastAPI'de dosya yükleme (`UploadFile`) bu paket olmadan çalışmıyor. Tarayıcının gönderdiği `multipart/form-data` gövdesini bu paket çözüyor. |
| **opencv-python-headless** | Eğitimde görüntüleri OpenCV ile okuyup işlemiştim. Ön işleme birebir aynı olsun diye serviste de OpenCV kullandım. "Headless" sürümü ekran ve GUI kütüphanesi gerektirmiyor, sunucu için yeterli ve daha hafif. |
| **numpy** | Yüklenen baytları OpenCV'nin okuyabileceği diziye çevirmek için. |
| **Docker** | Model benim bilgisayarımda çalışıyor ama başka bir makinede Python ve kütüphane sürümleri farklı olabilir. Docker işletim sistemini, Python'u, kütüphaneleri, kodu ve ağırlıkları tek bir imajda paketliyor. Böylece her makinede aynı ortam oluşuyor ve tek komutla çalışıyor. |
| **docker-compose** | Görev `docker-compose up` ile çalışmasını istiyor. Port ve platform ayarlarını her seferinde komut satırına yazmak yerine bir dosyada tutuyorum. |

---

## Nasıl bağladım?

### 1. Eğitimden servise: veri seti değil, ağırlıklar taşındı

Veri seti **sadece Colab'da, eğitim sırasında** kullanıldı. Servisin içinde veri seti yok. Eğitimden servise taşıdığım tek şey modelin öğrendiği sayılar, yani `detector_weights.pth` dosyası.

```
[COLAB: eğitim, bir kez yapıldı]
  Kaggle veri seti (görüntüler + kutu etiketleri)
    → ön işleme (224×224, normalize)
    → ResNet50 + 2 kafa ile eğitim
    → torch.save(model.state_dict()) → detector_weights.pth
                                             │ Colab'dan zip olarak indirdim
                                             ▼
[BU REPO: çıkarım, her istekte]
  model/detector_weights.pth → app/model.py yüklüyor → tahmin
```

`.pth` dosyasında sadece sayılar var. Bu sayıların doğru çalışması için dört şeyi eğitimle **birebir aynı** tutmam gerekti:

| Aynı olması gereken | Neden | Farklı olursa |
|---|---|---|
| **Mimari** (`ObjectDetector` sınıfı) | Ağırlıklar katman isimleriyle kayıtlı (örneğin `regressor.0.weight`). Koddaki katmanların isimleri ve boyutları bunlarla eşleşmeli. | `load_state_dict` hata veriyor |
| **Ön işleme** (BGR→RGB, 224×224, /255, ImageNet ortalama ve standart sapma) | Model eğitimde bu formatta veri gördü. | Hata vermiyor ama tahminler **sessizce bozuluyor** |
| **Sınıf sırası** `["Motorbikes", "airplanes", "schooner"]` | Model sadece 0, 1, 2 indeksini veriyor. Bu sırayı eğitimdeki `LabelEncoder` alfabetik olarak belirledi (büyük harf önce geliyor). | Motosiklete "uçak" diyor |
| **Kutu formatı** (0–1 arası `[x1, y1, x2, y2]`) | Eğitimde kutuları görüntü boyutuna bölerek normalize ettim, regressor'ün sonunda da `Sigmoid` var. | Kutu yanlış yere çiziliyor. Bu yüzden servis kutuyu görüntünün genişlik ve yüksekliğiyle çarpıp piksele çeviriyor. |

### 2. Model: `app/model.py`

- **Mimari:** ResNet50 gövdesi 2048 boyutlu bir özellik vektörü çıkarıyor. Bunun üstünde iki kafa var: `regressor` 4 sayılık bir kutu, `classifier` 3 sınıf skoru üretiyor. Gövdenin kendi `fc` katmanını `Identity()` ile kaldırdım.
- **Yükleme:** `resnet50(weights=None)` ile ImageNet ağırlıklarını internetten indirmiyorum, çünkü gövde dahil tüm ağırlıklar benim `.pth` dosyamda. `torch.load(..., weights_only=True)` güvenli yükleme yöntemi: dosyadan sadece tensör okuyor, kod çalıştırmıyor. `model.eval()` Dropout'u kapatıyor, böylece aynı görüntü her seferinde aynı sonucu veriyor.
- **Bir kez yükleme:** Model, modül import edildiğinde (`_model = model_yukle()`) yani servis açılırken bir kez RAM'e yükleniyor. Her istekte 95 MB'lık dosyayı tekrar okumuyorum.
- **`tahmin_et()` adımları:**
  1. Baytları `cv2.imdecode` ile görüntüye çeviriyorum. Görüntü değilse `ValueError` fırlatıyorum.
  2. Eğitimdeki ön işlemeyi uyguluyorum.
  3. Modeli `torch.inference_mode()` içinde çalıştırıyorum (gradyan hesaplanmıyor, daha hızlı).
  4. Sınıf skorlarını `softmax` ile olasılığa çevirip en yükseğini seçiyorum.
  5. Kutuyu orijinal görüntünün piksel koordinatlarına çeviriyorum.

### 3. FastAPI: `app/main.py`

Konteyner açıldığında olanlar:

1. Docker `uvicorn app.main:app --host 0.0.0.0 --port 7001` komutunu çalıştırıyor. `app.main:app` ifadesi "app paketindeki main.py dosyasında bulunan `app` değişkeni" demek.
2. `main.py` import edilirken `from app.model import tahmin_et` satırı `model.py` dosyasını da import ediyor ve model yükleniyor.
3. `@app.get("/")` ve `@app.post("/predict")` dekoratörleri, fonksiyonları FastAPI'nin "adres → fonksiyon" tablosuna kaydediyor.

Bir istek geldiğinde izlenen yol:

```
POST /predict (multipart/form-data, alan adı: file)
  → uvicorn → FastAPI yolu eşliyor → predict()
  → python-multipart gövdeyi çözüyor → UploadFile
  → await file.read() → baytlar → tahmin_et()
  → dict → FastAPI JSON'a çeviriyor → 200
  (ValueError gelirse → HTTPException → 400)
```

`GET /` adresi `FileResponse` ile arayüz sayfasını döndürüyor.

### 4. Arayüz: `app/static/index.html`

Tek bir dosya ve hiçbir JavaScript kütüphanesi kullanmıyor. Sayfa, görüntüyü `fetch("/predict")` ile aynı API'ye gönderiyor. Yani `curl`'ün ve `/docs` sayfasının yaptığı isteğin aynısını yapıyor. Gelen JSON'daki `bbox` değerini bir `<canvas>` üzerine çiziyor. Tuvali görüntünün **gerçek boyutunda** açtığım için API'nin verdiği piksel koordinatlarını dönüştürmeden doğrudan çizebiliyorum. Olasılıkları çubuklarla gösteriyor. Tasarımı pembe ve pikselli, sayfa bilgisayarın ayarına göre koyu ya da açık temada açılıyor.

### 5. Docker: `Dockerfile` ve `docker-compose.yml`

**Dockerfile** imajı katman katman kuruyor:

```dockerfile
FROM python:3.12-slim                       # Hafif Debian + Python (ARM ve x86 sürümü var)
RUN pip install torch torchvision ... /cpu  # En ağır adım, sadece CPU sürümü
RUN pip install -r requirements.txt         # fastapi, uvicorn, multipart, opencv, numpy
COPY app ./app                              # Kodum
COPY model ./model                          # Ağırlıklar
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7001"]
```

Docker her satırın sonucunu önbelleğe alıyor ve bir satır değişince sadece o satırdan sonrasını yeniden kuruyor. Bu yüzden ağır ve az değişen PyTorch kurulumunu en başa, sık değişen kodu en sona koydum. Kodu değiştirdiğimde PyTorch yeniden inmiyor, derleme birkaç saniyede bitiyor.

**docker-compose.yml**:

- `build: .` bu klasördeki Dockerfile'dan imaj oluşturuyor.
- `platform: linux/arm64` imajı ARM işlemciler için derliyor. Ayrıntısı bir sonraki bölümde.
- `ports: "7001:7001"` bilgisayarın 7001 portunu konteynerin 7001 portuna bağlıyor.

```
Tarayıcı → localhost:7001 (bilgisayar)
             │ Docker port yönlendirmesi
             ▼
          konteyner:7001 → uvicorn (--host 0.0.0.0)
```

### 6. ARM (Mac) desteği

M serisi Mac'lerin işlemcisi **ARM** mimarisinde, Intel ve AMD bilgisayarlar ise **x86**. Bu ikisi farklı makine kodu çalıştırıyor. `platform: linux/arm64` satırı sayesinde `python:3.12-slim` imajının da pip paketlerinin de ARM sürümleri indiriliyor. Mac'te imaj doğrudan, yani emülasyon olmadan çalışıyor.

Benim bilgisayarım x86 olduğu için ARM imajını Docker Desktop'ın **QEMU emülasyonu** ile derleyip test ettim. Konteynerin içinde `uname -m` komutu `aarch64` sonucunu verdi. Bu, kodun ARM ortamında çalıştığını gösteriyor.

---

## Karşılaştığım sorunlar, önlemler ve çözümler

| Sorun | Sebep | Çözüm |
|---|---|---|
| Ağırlıkları yüklemek için mimariyi tekrar yazmam gerekti | `detector_weights.pth` sadece `state_dict`, yani sadece sayılar içeriyor. Modelin yapısı dosyada yok. | `ObjectDetector` sınıfını Colab'daki `bbox_regressor.py` ile birebir aynı yazdım. Katman isimleri eşleşince `load_state_dict` hatasız çalıştı. |
| Sınıf isimleri `le.pickle` dosyasında kayıtlıydı | Bu dosyayı açmak için servise scikit-learn kurmak gerekiyordu. Sadece bir liste okumak için büyük bir bağımlılık eklemiş olacaktım. | `LabelEncoder` sınıfları alfabetik sıraladığı için sıra belliydi. Listeyi koda yazdım. |
| PyTorch'un normal kurulumu çok büyüktü | Varsayılan `pip install torch` komutu CUDA kütüphanelerini de indiriyor. Mac'te bunlar hiç kullanılmıyor. | `--index-url https://download.pytorch.org/whl/cpu` ile sadece CPU sürümünü kurdum. |
| `python-multipart` olmadan dosya yükleme çalışmaz | FastAPI `UploadFile` için bu pakete ihtiyaç duyuyor ama bunu kendisi kurmuyor. | `python-multipart` paketini `requirements.txt` dosyasına ekledim. |
| Varsayılan ayarla servise konteyner dışından erişilemez | uvicorn varsayılan olarak `127.0.0.1` adresini dinliyor, bu da sadece konteynerin kendi içi demek. | `--host 0.0.0.0` ile tüm ağ arayüzlerini dinlemesini sağladım. |
| Benim bilgisayarım x86, hedef ise ARM Mac | x86 için derlenen bir imaj Mac'te ya hiç çalışmıyor ya da yavaş bir emülasyonla çalışıyor. | `platform: linux/arm64` ekledim ve QEMU emülasyonuyla test ettim. Derleme yavaş sürdü ama `aarch64` üzerinde doğru tahminler aldım. |
| Görüntü olmayan dosya gelirse kontrol olmadan servis 500 hatası verirdi | `cv2.imdecode` okuyamadığı dosya için `None` döndürüyor, sonraki satırlar bu yüzden hata veriyor. | `None` gelirse `ValueError` fırlatıyorum, `main.py` bunu yakalayıp `400` hatası ve anlaşılır bir mesaj döndürüyor. |
| PowerShell'de `curl -F` çalışmadı | Windows PowerShell'de `curl`, `Invoke-WebRequest` komutunun takma adı. | `curl.exe` yazarak gerçek curl programını çalıştırdım. |
| Arayüzde kutu etiketi bazen sade yazıyla çıkıyordu | Canvas'a yazı, piksel yazı tipi henüz yüklenmeden çiziliyordu. | Çizmeden önce `document.fonts.load(...)` ile yazı tipinin yüklenmesini bekledim. |
| Arayüzde "TAHMİN" başlığı "TAHMiN" olarak görünüyordu | Piksel yazı tipinde Türkçe "İ" harfi yok. | Başlığı "Sonuç" olarak değiştirdim. |

---

## Test

Modelin eğitimde **görmediği** test görüntüleriyle denedim. Bu görüntüleri Colab'daki `test_paths.txt` listesinden seçtim ve Kaggle veri setinden aldım. Testler ARM imajında, emülasyon altında yapıldı:

| Görüntü | Tahmin | Olasılık |
|---|---|---|
| Motorbikes/image_0226.jpg | Motorbikes ✅ | 0.9992 |
| Motorbikes/image_0621.jpg | Motorbikes ✅ | 1.0 |
| airplanes/image_0529.jpg | airplanes ✅ | 1.0 |
| airplanes/image_0629.jpg | airplanes ✅ | 1.0 |
| schooner/image_0030.jpg | schooner ✅ | 0.9966 |
| schooner/image_0037.jpg | schooner ✅ | 0.9792 |
| metin dosyası (.txt) | HTTP 400 ✅ | "Dosya bir görüntü olarak okunamadı" |

**Bilinen sınır:** Model sadece bu 3 sınıfı biliyor ve "hiçbiri" diye bir seçeneği yok. Bir kedi fotoğrafı yüklenirse de bu üç sınıftan birini söylüyor.

---

## Neleri bilerek koymadım

Görev "gereksiz kod veya yapılandırma olmasın" dediği için:

| Colab'da vardı | Neden koymadım |
|---|---|
| `train.py`, `veri_okuyucu.py`, `custom_tensor_dataset.py` | Eğitim için yazılmış dosyalar, tahmin yaparken gerekmiyor. |
| Veri seti | Sadece eğitimde kullanılıyor. Servis kullanıcının yüklediği görüntüyle çalışıyor. |
| `detector.pth` (modelin tamamı) | `detector_weights.pth` yeterli ve daha güvenli. İkisini birden koymak 100 MB fazladan yer demek. |
| `le.pickle` | Açmak için scikit-learn gerekiyordu. Sınıf listesini koda yazdım. |
| `history.json`, grafikler, `test_paths.txt` | Eğitim ve rapor çıktıları, servisle ilgisi yok. |

Örnek repodaki ([ashmibanerjee/img-classifier-fastapi](https://github.com/ashmibanerjee/img-classifier-fastapi)) TensorFlow, locust ve test dosyalarını da almadım. O repo görüntüyü URL'den alıyor ve 8000 portunu kullanıyor. Bu görevde görüntü dosya olarak yükleniyor ve port 7001 isteniyor. O repodan sadece FastAPI + Dockerfile + docker-compose yapısının fikrini aldım.
