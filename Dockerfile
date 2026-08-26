FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY Requirements.txt ./
RUN pip install --no-cache-dir -r Requirements.txt

COPY app.py ./
COPY src ./src
COPY templates ./templates
COPY static ./static
COPY models ./models

RUN mkdir -p uploads outputs/video_evidence data

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "600", "app:app"]
