# Dockerfile (Updated)
FROM python:3.9-slim

# Install system dependencies:
# - libportaudio2: Runtime library for PortAudio (needed by sounddevice/pyaudio)
# - portaudio19-dev: Development headers for PortAudio (needed to *compile* pyaudio)
# - gcc: The C compiler (needed to *compile* pyaudio)
# - pkg-config: Helper tool often used during compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
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

# Set environment variables (optional if specified in CMD, but can be good practice)
# ENV STREAMLIT_SERVER_PORT=8501
# ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the Streamlit app when the container launches
# Explicitly setting port and address in CMD is robust
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]