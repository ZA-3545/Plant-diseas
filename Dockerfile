FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir fastapi uvicorn python-multipart pillow

COPY api.py .
COPY finetuned_mobilenet_best.pth .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]