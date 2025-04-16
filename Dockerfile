# Dockerfile (Updated Again)
FROM python:3.9-slim

# Install system dependencies:
# - gcc: C Compiler
# - libc6-dev: C Standard Library development headers (includes limits.h, etc.) <--- ADDED
# - libportaudio2: Runtime library for PortAudio
# - portaudio19-dev: Development headers for PortAudio
# - pkg-config: Helper tool for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    libportaudio2 \
    portaudio19-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies (pyaudio should build successfully now)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Run the Streamlit app when the container launches
# Explicitly setting port and address in CMD is robust
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]