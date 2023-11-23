FROM python:3.11.5

WORKDIR /app

EXPOSE 8086

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY entrypoint.sh ./
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]