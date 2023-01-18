FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="America/New_York"
RUN apt-get update && apt-get install -y \
  git apt-utils python3-pip libgl1-mesa-glx libglib2.0-0

# Install python packages
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .
EXPOSE 8000

ADD models.py .

#COPY weights weights
ADD download.py .
RUN python3 download.py

# Add your custom app code, init() and inference()
ADD app.py .


CMD python3 -u server.py
