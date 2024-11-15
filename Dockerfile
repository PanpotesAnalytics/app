# Use NVIDIA CUDA base image with Python support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    git \
    wget \
    build-essential \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install GDAL, OpenCV, Streamlit, and other Python libraries
RUN pip3 install rasterio \
    opencv-python-headless \
    geopandas \
    matplotlib \
    numpy \
    streamlit

# Install PyTorch, TorchVision, and Torchaudio with the CUDA 12.4 version
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install jupyterlab

# Clone and install segment-geospatial from GitHub
RUN git clone https://github.com/opengeos/segment-geospatial.git
RUN pip3 install -e segment-geospatial

RUN pip uninstall numpy -y
RUN pip install numpy==1.26.4

# Set the working directory
WORKDIR /app

# Copy your application files into the container
COPY . /app

ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5000

# Expose ports for JupyterLab and Streamlit
EXPOSE 8888 8501

# Set the entrypoint to start JupyterLab by default, or Streamlit if specified
ENTRYPOINT ["sh", "-c", "if [ \"$1\" = 'streamlit' ]; then streamlit run app.py --server.port 8501 --server.address 0.0.0.0; else jupyter lab --ip=0.0.0.0 --no-browser --allow-root; fi", "--"]
