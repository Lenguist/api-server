# syntax=docker/dockerfile:1

FROM python:3.7

COPY . /app
WORKDIR /app

RUN pip install allennlp
RUN pip3 install Flask==2.0.1

EXPOSE 5000

CMD [ "python3", "application.py"]
