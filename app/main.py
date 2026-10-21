# ============================================================================
# main.py : FastAPI servisi. Yüklenen görüntünün sınıflandırma sonucunu JSON döner.
# ============================================================================
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.model import tahmin_et

app = FastAPI(title="Nesne Tespit API")


@app.get("/")
def ana_sayfa():
    return {"msg": "Servis çalışıyor. POST /predict ile görüntü yükleyin (form alanı: file)."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    icerik = await file.read()
    try:
        sonuc = tahmin_et(icerik)
    except ValueError as e:
        # Görüntü olmayan dosya gönderilirse 400 (hatalı istek) döneriz
        raise HTTPException(status_code=400, detail=str(e))
    return {"filename": file.filename, **sonuc}
