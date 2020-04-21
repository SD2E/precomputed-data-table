#!/usr/bin/env bash

THIS=$(basename $0)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DIR/common.sh"

read_app_ini $1

docker rmi -f ${CONTAINER_IMAGE}
