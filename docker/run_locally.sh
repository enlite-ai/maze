#!/bin/bash
if [ "$#" -eq 0 ]; then
   echo "Please specify also commands as arguments like: $0 bash"
   exit 1
fi
cd "$(dirname "$0")"/..

# build image and extracte imaged id
IMAGE_ID=$(skaffold build -q -p nopush|jq -er .builds[0].tag)
if [[ $? -ne 0 ]]
then
  echo "skaffold build failed, retry:"
  skaffold build
  exit 1
fi

# Start with only the exection folder ro mounted and a writeable logs folder
docker run -it \
  -v /data/trainings_data/enliteai-loop/data-execution:/data/trainings_data/enliteai-loop/data-execution:ro\
  -v "$(pwd)/logs:/data/logs"\
  --gpus 1\
  $IMAGE_ID\
  "$@"