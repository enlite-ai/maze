variable "MAZE_CORE_ENV_IMAGE" {}
variable "MAZE_CORE_ENV_IMAGE_BUILD_CACHE" {}

variable "MAZE_CORE_IMAGE" {}
variable "MAZE_CORE_IMAGE_BUILD_CACHE" {}

variable "MAZE_IMAGE" {}
variable "MAZE_IMAGE_BUILD_CACHE" {}

group "default" {
  targets = ["maze-core-env", "maze-core", "maze"]
}

# Maze Core Env
# This service builds the base Maze Core Env image from source.
# It is intended to be used as a dependency for other builds,
# such as the maze-core and maze image, which uses this as its base.
target "maze-core-env" {
  context = "../"
  dockerfile = "docker/maze-core-env.dockerfile"
  tags = [
    "${MAZE_CORE_ENV_IMAGE}"
  ]
  cache-from = [
    "type=registry,ref=${MAZE_CORE_ENV_IMAGE_BUILD_CACHE}"
  ]
  cache-to = [
    "type=registry,ref=${MAZE_CORE_ENV_IMAGE_BUILD_CACHE},mode=max"
  ]
}

# Maze Core
# This service builds the base Maze Core image from source.
target "maze-core" {
  context = "../"
  dockerfile = "docker/maze-core.dockerfile"
  tags = [
    "${MAZE_CORE_IMAGE}"
  ]
  cache-from = [
    "type=registry,ref=${MAZE_CORE_IMAGE_BUILD_CACHE}"
  ]
  cache-to = [
    "type=registry,ref=${MAZE_CORE_IMAGE_BUILD_CACHE},mode=max"
  ]
  args = {
    MAZE_CORE_ENV: "${MAZE_CORE_ENV_IMAGE}"
  }
  depends_on = ["maze-core-env"]
}

# Maze
# This service builds the base Maze image from source.
target "maze" {
  context = "../../"
  dockerfile = "deployment/maze.dockerfile"
  tags = [
    "${MAZE_IMAGE}"
  ]
  cache-from = [
    "type=registry,ref=${MAZE_IMAGE_BUILD_CACHE}"
  ]
  cache-to = [
    "type=registry,ref=${MAZE_IMAGE_BUILD_CACHE},mode=max"
  ]
  args = {
    MAZE_CORE_ENV: "${MAZE_CORE_ENV_IMAGE}"
  }
  depends_on = ["maze-core-env"]
}
