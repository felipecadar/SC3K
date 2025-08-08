#!/bin/bash

# --- Configuration ---
# The name for your Docker image.
IMAGE_NAME="sc3k-image"

# The directory inside the container where your code will be mounted.
# This should match the WORKDIR in your Dockerfile.
CONTAINER_WORKDIR="/app"

# --- Script Logic ---

# Check if the Docker image exists locally.
# The `docker images -q` command returns the ID of the image if it exists,
# and an empty string otherwise.
if [[ -z "$(docker images -q $IMAGE_NAME 2> /dev/null)" ]]; then
  echo "Image '$IMAGE_NAME' not found. Building it now..."
  # Build the Docker image from the Dockerfile in the current directory.
  docker build -t $IMAGE_NAME .
else
  echo "Image '$IMAGE_NAME' found locally."
fi

echo "---"
echo "Starting container from image: $IMAGE_NAME"
echo "Mapping current directory ($(pwd)) to $CONTAINER_WORKDIR"

# Run the Docker container with the following options:
# --rm: Automatically remove the container when it exits to keep things clean.
# -it:  Run in interactive mode and allocate a pseudo-TTY.
# --gpus all: Make all host GPUs available inside the container.
# -v "$(pwd)":$CONTAINER_WORKDIR: Mount the current host directory to the container's working directory.
docker run --rm -it --gpus all -v "$(pwd)":$CONTAINER_WORKDIR $IMAGE_NAME
