FROM python:3.12-slim

WORKDIR /code

# PyTorch'un sadece CPU sürümü (CUDA'lı sürüm GB'larca yer kaplar ve Mac'te işe yaramaz)
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY model ./model

EXPOSE 7001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7001"]
