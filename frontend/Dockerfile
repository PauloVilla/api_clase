FROM python:3.9-slim-buster

WORKDIR /code

RUN apt-get update && apt-get install -y procps && pip install -U pip \
    && rm /etc/localtime  \
    && ln -s /usr/share/zoneinfo/America/Mexico_City /etc/localtime

COPY ./requirements.txt /code/requirements.txt

RUN pip install -r ./requirements.txt

COPY ./iris-app.py /code/iris-app.py

EXPOSE 8501

CMD ["streamlit", "run", "iris-app.py", "--server.port=8501", "--server.address=0.0.0.0"]