# syntax=docker/dockerfile:1.2
# Maze playground Docker file: Maze + Jupyter lab starting as entrypoint. Used for Docker deployment workflow on Github.

FROM enliteai/maze:latest

RUN pip install ipykernel && python -m ipykernel install --user --name env --display-name maze-env

# Start Jupyter lab.
EXPOSE 8888
ENTRYPOINT ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]