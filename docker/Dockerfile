# syntax=docker/dockerfile:1.2
# Multi-stage build with split of conda env and source code

FROM nvidia/cuda:10.2-runtime as maze_core_conda
WORKDIR /usr/src/app

# Include custom ubuntu dependency
RUN apt-get update && apt-get install -y wget git python-opengl xvfb make gcc g++ htop redis-server && rm -rf /var/lib/apt/lists/*

RUN  wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
ENV PATH /opt/conda/bin:$PATH

# Make gitlab.enlite.ai host known
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan gitlab.enlite.ai >> ~/.ssh/known_hosts

WORKDIR /usr/src/maze

# Install conda

# Need explicit permissions as the vary from CI env to local env (and therefore lead to cache invalidation)
COPY --chmod=400 environment.yml environment.yml
COPY --chmod=400 setup.py setup.py
COPY --chmod=400 maze/__init__.py maze/__init__.py

RUN conda env create -n env
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# execute the Maze installation exactly as described in README.md
RUN pip install -e .[testing]
# install RLLib
RUN pip install ray[rllib] tensorflow

FROM scratch as maze_core_code
# Copy code
COPY . .

# Ouput Image
FROM maze_core_conda
COPY --from=maze_core_code . .
RUN pip install -e .
