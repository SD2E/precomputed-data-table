#!/usr/bin/env bash

ACTOR_ID=
if [ -f ".ACTOR_ID" ]; then
    ACTOR_ID=$(cat .ACTOR_ID)
fi

deployopts=""
if [ ! -z "$ACTOR_ID" ]; then
    deployopts="${deployopts} -U ${ACTOR_ID}"
fi

source reactor.rc

# if [ ! -z "${REACTOR_ALIAS}" ]; then
#     syd add "${REACTOR_ALIAS}" ${ACTOR_ID}
#     syd acl "${REACTOR_ALIAS}" world --read
# fi

abaco share -u ABACO_WORLD -p READ ${ACTOR_ID}
abaco share -u world -p READ ${ACTOR_ID}
abaco share -u jed -p UPDATE ${ACTOR_ID}
abaco share -u gzynda -p UPDATE ${ACTOR_ID}
