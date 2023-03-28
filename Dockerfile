# use slim instead of alpine to use numpy wheels (and avoid building numpy from source)
FROM python:3.8.8-slim
LABEL maintainer="klaus.eckelt@jku.at"

WORKDIR /marjorie
COPY ./ ./

RUN pip3 install --no-cache-dir --requirement requirements.txt

CMD ["python", "index.py"]

EXPOSE 8050