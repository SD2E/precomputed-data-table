

UNDER_CI=0
UNDER_MACOS=0
CI_PLATFORM=
CI_UID=$(id -u ${USER})
CI_GID=$(id -g ${USER})
CI_CONTAINER_NAME="test-${RANDOM}-${RANDOM}"

function die(){
    echo "[ERROR] $1"
    exit 1
}

function log(){
    echo "[INFO] $1"
}

function detect_ci() {

  ## Are we running on Mac?
  if [[ "$OSTYPE" =~ "darwin" ]]; then
    UNDER_MACOS=1
  fi

  ## detect whether we're running under continous integration
  if [ ! -z "${TRAVIS}" ]; then
    if [ "${TRAVIS}" == "true" ]; then
      UNDER_CI=1
      CI_PLATFORM="Travis"
      CI_UID=$(id -u travis)
      CI_GID=$(id -g travis)
    fi
  fi

  if [ ! -z "${JENKINS_URL}" ]; then
    UNDER_CI=1
    CI_PLATFORM="Jenkins"
    CI_UID=$(id -u jenkins)
    CI_GID=$(id -g jenkins)
  fi

  if ((UNDER_MACOS)); then
    log "Operating system is Mac OS X (Caveat emptor)."
  else
    log "Operating system is ${OSTYPE}."
  fi

  if ((UNDER_CI)); then
    log "Continuous integration platform is ($CI_PLATFORM)"
  else
    log "Not running under continuous integration"
  fi

}

function random_hex() {

  if [ -z "$1" ]; then
    length_hex=16
  else
    length_hex=$1
  fi
  hexx=$(cat /dev/urandom | env LC_CTYPE=C tr -dc 'a-zA-Z0-9' | fold -w $length_hex | head -n 1)
  echo -n $hexx
}

function read_reactor_rc() {

  if [ -z "$REACTOR_RC" ]; then
      REACTOR_RC="reactor.rc"
  fi
  if [ -f ${REACTOR_RC} ]; then
      source ${REACTOR_RC}
  else
      log "No reactors config file ${REACTOR_RC} found"
  fi

  if ((DOCKER_USE_COMMIT_HASH));
  then
    DOCKER_IMAGE_VERSION=$(git rev-parse --short HEAD)
  fi

  CONTAINER_IMAGE="$DOCKER_HUB_ORG/${DOCKER_IMAGE_TAG}:${DOCKER_IMAGE_VERSION}"

  if [ -z "${REACTOR_SECRETS_FILE}" ]; then
      REACTOR_SECRETS_FILE="secrets.json"
  fi

  # Env-level variable to let users opt out of the
  # function-by-function replacement of Bash code
  # with AgavePy workalikes in the Agave CLI
  if [[ "$AGAVE_PREFER_PYTHON" != "0" ]]; then
    AGAVE_PREFER_PYTHON=1
  fi

}

function read_app_ini() {

    INIFILE=$1
    if [ -z "$INIFILE" ]; then
      INIFILE="app.ini"
    fi
    if [ ! -f "$INIFILE" ]; then
      die "Unable to find or access $INIFILE"
    fi

    # Very bad INI reader
    export DOCKER_HUB_ORG=$(egrep "^username" $INIFILE | awk -F '=' '{ print $2 }' | tr -d " ")
    export DOCKER_IMAGE_TAG=$(egrep "^repo" $INIFILE | awk -F '=' '{ print $2 }' | tr -d " ")
    export DOCKER_IMAGE_VERSION=$(egrep "^tag" $INIFILE | awk -F '=' '{ print $2 }' | tr -d " ")
    export APP_NAME=$(egrep "^name" $INIFILE | awk -F '=' '{ print $2 }' | tr -d " ")
    export APP_VERSION=$(egrep "^version" $INIFILE | awk -F '=' '{ print $2 }' | tr -d " ")
    export APP_ID="${APP_NAME}-${APP_VERSION}"
    export TESTING_DIRECTORY=$(egrep "^directory" $INIFILE | awk -F '=' '{ print $2 }' | tr -d " ")

    CONTAINER_IMAGE="${DOCKER_IMAGE_TAG}"
    if [ ! -z "${DOCKER_HUB_ORG}" ]; then
      CONTAINER_IMAGE="${DOCKER_HUB_ORG}/${CONTAINER_IMAGE}"
    fi
    if [ ! -z "${DOCKER_IMAGE_VERSION}" ]; then
      CONTAINER_IMAGE="${CONTAINER_IMAGE}:${DOCKER_IMAGE_VERSION}"
    fi
    export CONTAINER_IMAGE

    echo "ini: $INIFILE"
    echo "container: $CONTAINER_IMAGE"
    echo "appId: $APP_ID"

}