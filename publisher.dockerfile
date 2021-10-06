FROM python:3.9.5-slim

WORKDIR /app

COPY requirements requirements

RUN pip install --no-cache-dir -r requirements/publisher.txt

COPY . .

CMD ["python3", "sigeml/publisher.py"]
