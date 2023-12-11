FROM python:3.10-slim

WORKDIR /app

COPY bge-large-zh_api.py requirements.txt /app/

RUN pip install -r requirements.txt

EXPOSE 6006

CMD ["python3", "bge-large-zh_api.py"]
