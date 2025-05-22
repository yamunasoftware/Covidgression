FROM python:3.8-slim
WORKDIR /main
USER root
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
CMD [ "bash", "covidgression.sh" ]