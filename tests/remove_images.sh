#!/usr/bin/env bash

source reactor.rc

CONTAINER_IMAGE="$DOCKER_HUB_ORG/${DOCKER_IMAGE_TAG}:${DOCKER_IMAGE_VERSION}"

docker rmi -f ${CONTAINER_IMAGE}
