# Use uma imagem oficial do Python como base
FROM python:3.12-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Instala dependências do sistema necessárias para YOLO, OpenCV (cv2) e PaddleOCR
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
        libopencv-dev \
        ffmpeg \
        wget \
        git \
        && rm -rf /var/lib/apt/lists/*

# Copia os arquivos de dependências (ajuste conforme necessário)
COPY requirements.txt .

# Instala as dependências do projeto
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o restante do código do projeto para o container
COPY . .

# Define o comando padrão para rodar o projeto (ajuste conforme necessário)
CMD ["python", "main.py"]