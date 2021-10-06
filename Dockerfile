FROM python:3.9.5-slim

WORKDIR /app

COPY requirements requirements

RUN apt-get update && apt-get -y install libpq-dev gcc

RUN pip install --no-cache-dir -r requirements/base.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "sigeml.api:app", "--host", "0.0.0.0", "--reload"]
