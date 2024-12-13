from nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN mkdir -p /root/.config/pip
RUN echo '[global]\nindex-url = https://mirrors.bfsu.edu.cn/pypi/web/simple' > /root/.config/pip/pip.conf

RUN sed -i "s@http://.*archive.ubuntu.com@http://mirrors.bfsu.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@http://mirrors.bfsu.edu.cn@g" /etc/apt/sources.list
RUN apt update \
 && DEBIAN_FRONTEND=noninteractive apt install -y gcc g++ python3 python3-dev python3-pip libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements.txt
RUN python3 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3 -m pip install -r requirements.txt

EXPOSE 8000
CMD python3 app.py