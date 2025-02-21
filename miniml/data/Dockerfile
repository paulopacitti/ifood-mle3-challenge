FROM python:3.12.3-bookworm
# shared volume to save dataset
VOLUME output/

WORKDIR /app

# install python deps
COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY main.py ./main.py
ENTRYPOINT ["python", "./main.py"]
