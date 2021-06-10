# syntax=docker/dockerfile:1.2
# Build image maze-core.

ARG MAZE_CORE_ENV=maze_core_env:latest

FROM ${MAZE_CORE_ENV}
WORKDIR /usr/src/maze/
COPY . .
RUN pip install -e .