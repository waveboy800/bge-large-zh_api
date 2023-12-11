FROM python:3.10-slim

WORKDIR /app

COPY m3e-base_api.py requirements.txt /app/

RUN pip install -r requirements.txt

EXPOSE 6006

CMD ["python3", "m3e-base_api.py"]