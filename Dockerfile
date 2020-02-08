FROM bryanlimy/projects:0.1-cuda10.1-cudnn7.6.5-conda

RUN apt update -y
RUN apt install build-essential -y

COPY setup.sh /home/bryanlimy/setup.sh
COPY requirements.txt /home/bryanlimy/requirements.txt
