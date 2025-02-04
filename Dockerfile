# Use a PyTorch base image with CUDA support (version 1.9.1, CUDA 11.1, cuDNN 8)
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# Set the working directory inside the container
WORKDIR /app

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN  rm -rf /var/lib/apt/lists/* \
         /etc/apt/sources.list.d/cuda.list \
          /etc/apt/sources.list.d/nvidia-ml.list 
# Update package list and install necessary system packages
RUN apt update && \
    apt install -y  \
    openssh-server \ 
    rsync \
    wget \
    bzip2 zip unzip \
    libvips libvips-tools libvips-dev git \
    libgl1 libglib2.0-0 

# Upgrade pip and install Python dependencies
# RUN pip3 install --upgrade pip && \
# 	#pip3 install scikit-learn wandb numpy pandas openslide-python h5py jupyter && \
# 	pip3 install jupyter wandb openslide-python && \
# 	pip3 install scikit-learn pandas h5py future verstack && \
# 	pip3 install --no-cache-dir hydra-core==1.2.0 
COPY requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip && \
	#pip3 install scikit-learn wandb numpy pandas openslide-python h5py jupyter && \
	pip3 install -r /app/requirements.txt

# Clone and install smooth-topk
RUN git clone https://github.com/oval-group/smooth-topk.git && \
    cd smooth-topk && \
    python3 setup.py install && \
    cd .. && \
    rm -rf smooth-topk

# Clone the CLAM repository
RUN git clone https://github.com/mahmoodlab/CLAM.git /opt/CLAM

# Set PYTHONPATH to include the CLAM repository
ENV PYTHONPATH="/opt/CLAM:/app:${PYTHONPATH}"

# Copy the application code
COPY . /app