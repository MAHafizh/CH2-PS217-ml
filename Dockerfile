FROM python:3.10-slim

ENV PYTHONBUFFERED True

ENV APP_HOME /app

WORKDIR $APP_HOME

COPY . ./

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libopencv-dev python3-opencv

RUN pip install -r requirements.txt

EXPOSE 5000

CMD exec gunicorn --bind :$PORT app:app

