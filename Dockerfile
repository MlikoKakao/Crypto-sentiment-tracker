FROM python:3.12-slim

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-api.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements-api.txt

COPY . .
