FROM python:3.12

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080 5000 8081

CMD ["bash", "run_pipeline.sh"]
