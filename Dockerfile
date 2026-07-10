FROM python:3.14

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://aka.ms/InstallAzureCLIDeb | bash

COPY . .

CMD ["python", "app.py"]