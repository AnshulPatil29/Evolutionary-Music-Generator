# Dockerfile
FROM python:3.9-slim

# Install system dependency for sounddevice (PortAudio)
RUN apt-get update && apt-get install -y --no-install-recommends libportaudio2 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
CMD ["streamlit", "run", "app.py"]