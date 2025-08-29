FROM python:3.10

ENV CONTAINER_HOME=/home/lobster/Docker/websync

ADD . $CONTAINER_HOME
WORKDIR $CONTAINER_HOME

RUN pip install -r $CONTAINER_HOME/requirements.txt

