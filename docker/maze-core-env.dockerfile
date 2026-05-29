# syntax=docker/dockerfile:1.2
# Builds image maze_core_env.

ARG BUILD_IMAGE=condaforge/mambaforge:4.11.0-0
ARG BASE_IMAGE=condaforge/mambaforge:4.11.0-0

###################################################
# Image for conda/mamba environment building.
###################################################

FROM ${BUILD_IMAGE} as maze_core_env_build

# Install build dependencies needed for compiled pip packages (e.g. box2d-py).
RUN apt-get update && \
    apt-get install -y build-essential g++ swig libbox2d-dev && \
    rm -rf /var/lib/apt/lists/*

# Install environment.
COPY maze-core-environment.yml .
RUN mamba env create -p /env --file maze-core-environment.yml
RUN conda clean -afy
RUN echo "conda activate /env" >> /root/.bashrc
ENV PATH /env/bin:$PATH

###################################################
# Image for built, clean environment.
###################################################

FROM ${BASE_IMAGE}

# Install system dependencies and Maze.
RUN apt-get update && \
    apt-get install -y xvfb redis-server python-opengl libbox2d-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy conda environment.
ENV CONDA_ENVS_PATH=/
ENV PATH /env/bin:$PATH
COPY --from=maze_core_env_build /env /env
COPY --from=maze_core_env_build /root/.bashrc /root/
