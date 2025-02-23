FROM python:3.12.3-slim

ENV MODEL_PATH="/models/saved_model"
ENV DEVICE="cpu"

COPY models /models

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py ./config.py
COPY model.py ./model.py
COPY main.py ./main.py

CMD ["fastapi", "run", "main.py"]

EXPOSE 8000