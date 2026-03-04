# =====================================================
# Stage 1 — Build Executable
# =====================================================
FROM python:3.10-slim AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pyinstaller

# Build onefile executable
RUN pyinstaller \
    --onefile \
    --name lpr_app \
    --collect-all ultralytics \
    --hidden-import=lap \
    --hidden-import=cv2 \
    app/main.py



# =====================================================
# Stage 2 — Runtime (Clean Image)
# =====================================================
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install ONLY runtime libs (no build tools)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy built binary
COPY --from=builder /app/dist/lpr_app /app/lpr_app

# Copy public key (if license uses it)
COPY public_key.pem /app/public_key.pem

# Create runtime folders
RUN mkdir -p /app/license /app/logs /app/outputs

# Make executable safe
RUN chmod +x /app/lpr_app

# Start application
CMD ["./lpr_app"]
