FROM ubuntu:18.04

RUN apt-get update && apt-get -y upgrade \
    && apt-get -y install python3.8 \
    && apt -y install python3-pip \
    && pip3 install --upgrade pip

WORKDIR /app

COPY . /app

RUN pip3 --no-cache-dir install -r requirements.txt 

EXPOSE 10000


ENTRYPOINT  ["python3"]

CMD ["auto_training_data.py"]

ENTRYPOINT  ["python3"]

CMD ["train.py"]

ENTRYPOINT  ["python3"]

CMD ["app.py"]
