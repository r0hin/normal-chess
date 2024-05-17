FROM ubuntu:20.04 as base

SHELL ["/bin/sh", "-c"]

CMD ["bash"]

# DISABLE PROMPT DURING PACKAGES INSTALLATION
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update

RUN apt upgrade

# Install caffe
RUN apt -y install caffe-cpu

# Install  caffe-cpu dependencies
RUN apt -y install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler

# Install python
RUN apt -y install python

# Install python dependencies
RUN apt -y install python3-pip

RUN pip install flask opencv-python scipy scikit-image torchvision torch 

RUN pip install numpy==1.21

COPY . /app

EXPOSE 5000

ENTRYPOINT [ "python3", "/app/server.py" ]