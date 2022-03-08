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

DOCKERFILE_VAR="${DOCKERFILE:-docker/maze-core.dockerfile}"

function build_local {
  export DOCKER_BUILDKIT=1
  # Convert list of BUILDARG to docker parameters
  DOCKER_BUILDARGS=`for a in $BUILDARG; do echo "--build-arg $a"; done`
  if [ ! -z "$TARGET" ] ; then
       DOCKER_TARGET="--target $TARGET"
  fi
  # Cache from built image with same tag/pipeline ID, if specified.
  if [ ! -z "$CACHE_FROM" ] ; then
       full_cache_image_name=$CACHE_FROM:$(echo $IMAGE | sed 's/.*://')
       DOCKER_CACHE_FROM="--cache-from=$full_cache_image_name"
       docker pull $full_cache_image_name || true
  fi

  docker build --ssh default $DOCKER_BUILDARGS $DOCKER_TARGET -f $DOCKERFILE_VAR $DOCKER_CACHE_FROM -t $IMAGE .
}

# Function to resolve build arguments. Useful because right now there are two types of docker build arguments that are
# treated differently:
#   - (1) References to docker images, such as MAZE_CORE_CODA. When pushing an image, these are passed with the image
#   hash. We assume that these always start with "MAZE_".
#   - (2) Other build arguments, e.g. COMPONENT for extension.dockerfile. These shouldn't be modified like docker image
#   references/hashes.
function resolve_buildarg () {
  if [[ $1 == MAZE_* ]]; then suffix="=${!1} "; else suffix=" "; fi;
  echo "--opt build-arg:$1"$suffix
}

# Remove last :.* string
IMAGE_WITHOUT_TAG=`echo $IMAGE|sed 's/:[^:]*$//'`

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
  SELECTED_POD=$(kubectl get pods --selector=app=buildkitd -o json |
    jq -r '.items[] | select(.status.phase=="Running") | .metadata.name' |
    shuf -n 1 --random-source=<(echo $BUILD_CONTEXT | shasum))

  if [[ -z $SELECTED_POD ]]
  then
    echo >&2 "No buildkitd pod found in cluster. Falling back to local build"
    build_local
    docker push $IMAGE
  else
    echo "Start build on $SELECTED_POD"

   # Convert list of BUILDARG to docker parameters
    BUILDKIT_BUILDARGS=`for a in $BUILDARG; do resolve_buildarg $a; done`
    if [ ! -z "$TARGET" ] ; then
       BUILDKIT_TARGET="--opt target=$TARGET"
    fi
    # Buildkit splits dockerfile name and dockerfile location
    BUILDKIT_DOCKERFILE_DIR=`dirname $DOCKERFILE_VAR`
    BUILDKIT_DOCKERFILE_NAME=`basename $DOCKERFILE_VAR`
    # Build remote, use the registry as cache
    buildctl --addr kube-pod://"$SELECTED_POD" build --frontend=dockerfile.v0 --local context=.\
     --local dockerfile=$BUILDKIT_DOCKERFILE_DIR --opt filename=$BUILDKIT_DOCKERFILE_NAME\
    --ssh default=$SSH_AUTH_SOCK --output type=image,name=$IMAGE,push=true\
    --export-cache type=registry,ref=$IMAGE_WITHOUT_TAG:buildcache,push=true \
    --import-cache type=registry,ref=$IMAGE_WITHOUT_TAG:buildcache \
    $BUILDKIT_BUILDARGS $BUILDKIT_TARGET
  fi

else
  build_local
fi

