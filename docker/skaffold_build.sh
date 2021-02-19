#!/bin/bash
#Builds an image using remote buildkit or docker
#
#Supports following parameters passed as environment variables
#
#TARGET
#DOCKERFILE
#BUILDARG

set -e
if ! command -v jq &> /dev/null
then
    echo "jq could not be found please install"
    exit
fi


if [ -z "$SSH_AUTH_SOCK" ] ; then
  echo >&2 'No SSH agent is running, but this is required for building

Please start the SSH agent with:

  eval `ssh-agent`
  ssh-add
  '
fi

if [ ! -z "$DOCKER_REMOTE_HOST" ] ; then
  echo "DOCKER_REMOTE_HOST setup is not supported anymore, building is now always remote" >&2
fi

DOCKERFILE_VAR="${DOCKERFILE:-docker/Dockerfile}"

function build_local {
  export DOCKER_BUILDKIT=1
  # Convert list of BUILDARG to docker parameters
  DOCKER_BUILDARGS=`for a in $BUILDARG; do echo "--build-arg $a"; done`
  if [ ! -z "$TARGET" ] ; then
       DOCKER_TARGET="--target $TARGET"
  fi
  docker build --ssh default $DOCKER_BUILDARGS $DOCKER_TARGET -f $DOCKERFILE_VAR -t $IMAGES .
}


# Remove last :.* string
IMAGE_WITHOUT_TAG=`echo $IMAGES|sed 's/:[^:]*$//'`

if [[ ${PUSH_IMAGE} == "true" ]]
then
  # if we want to have the images remote we build them remote

  #check if buildctl is available
  command -v buildctl >/dev/null 2>&1 ||
   { echo >&2 "require buildctl but it's not installed. Please install with

    sudo bash -c  'cd /usr/local/; tar xfz <(curl -L https://github.com/moby/buildkit/releases/download/v0.8.0/buildkit-v0.8.0.linux-amd64.tar.gz) bin/buildctl'

    Aborting."; exit 1; }

  # Select a buildkitd pod
  # Use a seeded sort to select always the same pod for the same context
  SELECTED_POD=$(kubectl get pods --selector=app=buildkitd  -o json  |
  jq -r '.items[] |   select(.status.conditions[] | select(.type == "Ready") | .status=="True")   |.metadata.name' |\
 shuf -n 1 --random-source=<(echo $BUILD_CONTEXT|shasum))

  if [[ -z $SELECTED_POD ]]
  then
    echo >&2 "No buildkitd pod found in cluster. Falling back to local build"
    build_local
    docker push $IMAGES
  else
    echo "Start build on $SELECTED_POD"

   # Convert list of BUILDARG to docker parameters
    BUILDKIT_BUILDARGS=`for a in $BUILDARG; do echo "--opt build-arg:$a=${!a} "; done`
    if [ ! -z "$TARGET" ] ; then
       BUILDKIT_TARGET="--opt target=$TARGET"
    fi
    # Buildkit splits dockerfile name and dockerfile location
    BUILDKIT_DOCKERFILE_DIR=`dirname $DOCKERFILE_VAR`
    BUILDKIT_DOCKERFILE_NAME=`basename $DOCKERFILE_VAR`
    # Build remote, use the registry as cache
    buildctl --addr kube-pod://"$SELECTED_POD" build --frontend=dockerfile.v0 --local context=.\
     --local dockerfile=$BUILDKIT_DOCKERFILE_DIR --opt filename=$BUILDKIT_DOCKERFILE_NAME\
    --ssh default=$SSH_AUTH_SOCK --output type=image,name=$IMAGES,push=true\
    --export-cache type=registry,mode=max,ref=$IMAGE_WITHOUT_TAG:buildcache \
    --import-cache type=registry,mode=max,ref=$IMAGE_WITHOUT_TAG:buildcache \
    $BUILDKIT_BUILDARGS $BUILDKIT_TARGET
  fi

else
  build_local
fi

