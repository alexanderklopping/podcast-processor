FROM python:3.11-slim

# Install ffmpeg and git
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY podcast_processor.py .
COPY podcasts.json .

# Run the processor
CMD ["python", "podcast_processor.py"]
