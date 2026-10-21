# ============================================================================
# main.py : FastAPI servisim. Ana sayfada test arayüzünü gösteriyorum,
# /predict ile yüklenen görüntünün tahmin sonucunu JSON olarak dönüyorum.
# ============================================================================
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.model import tahmin_et

app = FastAPI(title="Nesne Tespit API")

ARAYUZ = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/")
def ana_sayfa():
    # Tarayıcıdan görüntü yükleyip sonucu kutusuyla görebilmek için hazırladığım sayfa
    return FileResponse(ARAYUZ)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    icerik = await file.read()
    try:
        sonuc = tahmin_et(icerik)
    except ValueError as e:
        # Görüntü olmayan bir dosya gelirse 400 (hatalı istek) dönüyorum
        raise HTTPException(status_code=400, detail=str(e))
    return {"filename": file.filename, **sonuc}
