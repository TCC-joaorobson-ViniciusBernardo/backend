FROM python:3.9.5-slim

WORKDIR /app

COPY requirements requirements

RUN pip install --no-cache-dir -r requirements/base.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--reload"]
