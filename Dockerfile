# CPU image for the DepthSR demo (also used by Hugging Face Spaces, sdk: docker)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/tmp/hf GRADIO_SERVER_NAME=0.0.0.0
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only torch keeps the image small (~1 GB instead of ~5 GB)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

COPY . .
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 7860
CMD ["python", "app.py"]
