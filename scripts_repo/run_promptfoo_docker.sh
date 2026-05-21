#!/usr/bin/env bash

if [[ -n "$(docker ps -aq --filter 'name=^promptfoo$')" ]]; then
  echo "Container 'promptfoo' already exists; stopping and deleting it..." >&2
  docker rm -f promptfoo >/dev/null
fi

# Command for running the promptfoo UI server in the background.
docker run -d \
  --name promptfoo \
  -p 3000:3000 \
  -v ~/.promptfoo:/home/promptfoo/.promptfoo \
  ghcr.io/promptfoo/promptfoo:latest