FROM python:3.12.3-bookworm
VOLUME shared/

WORKDIR /app
COPY requirements.txt ./requirements.txt
COPY main.py ./main.py
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "./main.py"]
