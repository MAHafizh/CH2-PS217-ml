FROM python:3.10-slim

ENV PYTHONBUFFERED True

ENV APP_HOME /app

WORKDIR $APP_HOME

COPY . ./

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["sh", "-c", "python -m flask run --host=0.0.0.0 --port=$PORT"]