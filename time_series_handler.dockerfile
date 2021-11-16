FROM python:3.9.5-slim

WORKDIR /app

COPY requirements requirements

RUN pip install --no-cache-dir -r requirements/time_series.txt

COPY . .

CMD ["python3", "run_time_series_handler.py"]
