FROM python:3.12-slim

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY run_app.py ./

RUN pip install --upgrade pip \
    && pip install -e .

CMD ["python", "run_app.py"]

