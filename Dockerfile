FROM python:3.12-slim

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY mediaverwerker/ mediaverwerker/
COPY podcasts.json .

RUN pip install --no-cache-dir -e .

CMD ["mediaverwerker", "process"]
