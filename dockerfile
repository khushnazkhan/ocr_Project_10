# Dockerfile for Custom OCR (YOLO + Tesseract) App
# CPU-only image using Python 3.10-slim; installs tesseract and python deps.

FROM python:3.10-slim

# Install system deps and Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    libleptonica-dev \
    libtesseract-dev \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Streamlit configuration
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "ui_app_polished.py", "--server.port=8501", "--server.address=0.0.0.0"]
