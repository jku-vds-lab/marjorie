# use slim instead of alpine to use numpy wheels (and avoid building numpy from source)
FROM python:3.8.8-slim
LABEL maintainer="klaus.eckelt@jku.at"

WORKDIR /marjorie
COPY ./ ./

RUN pip3 install --no-cache-dir --requirement requirements.txt

CMD ["python", "index.py"]

EXPOSE 8050

# 1. Switch to production mode
#   in index.py: Set debug to False
# 2. Build image and push to AWS
#   aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 478950388974.dkr.ecr.eu-central-1.amazonaws.com
#   docker build -t marjorie .
#   docker tag marjorie:latest 478950388974.dkr.ecr.eu-central-1.amazonaws.com/marjorie:latest
#   docker push 478950388974.dkr.ecr.eu-central-1.amazonaws.com/marjorie:latest