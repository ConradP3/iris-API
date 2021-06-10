FROM continuumio/miniconda3:4.8.2
MAINTAINER UNP, https://unp.education

COPY ./application /usr/local/python

EXPOSE 5000

WORKDIR /usr/local/python

RUN pip install -r requirements.txt

CMD python iris_API.py
