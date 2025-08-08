# Use an available official NVIDIA CUDA 10.1 base image with OpenGL support.
# This ensures the correct drivers and libraries are available system-wide.
FROM nvidia/cudagl:10.1-devel-ubuntu18.04

# Set the working directory inside the container
WORKDIR /app

# --- Install Miniconda ---
# Set environment variables for a non-interactive installation.
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Add the NVIDIA public key to fix GPG error, then update and install Miniconda.
# The key (A4B469963BF863CC) is required to verify the NVIDIA CUDA repository.
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update && \
    apt-get install -y --no-install-recommends wget bzip2 && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    conda clean -afy && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Install System Dependencies ---
# Install libraries required by OpenCV, Open3D, etc.
# The base image already contains many of these, but this ensures they are present.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the Conda environment file into the container.
COPY environment.yml .

# Accept the Conda Terms of Service for the 'defaults' channel non-interactively.
# This is required to prevent build failures in newer versions of Conda.
RUN conda tos accept --override-channels --channel defaults

# Create the Conda environment from the environment.yml file.
# The `cudatoolkit=10.1` in the yml file will be satisfied by the base image.
RUN conda env create -f environment.yml

# --- Configure the shell for automatic environment activation ---
# Initialize Conda for the bash shell.
RUN conda init bash

# Add a command to the .bashrc file to automatically activate the environment.
# This will run every time a new interactive shell is started.
RUN echo "conda activate sc3k" >> /root/.bashrc

# Set a default command to run when the container starts.
# This will launch a bash shell, which will then automatically activate the environment.
CMD ["/bin/bash"]
