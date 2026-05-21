#!/usr/bin/env bash

if [[ -n "$(docker ps -q --filter 'name=^promptfoo$')" ]]; then
  echo "Container 'promptfoo' is already running; stopping it..." >&2
  docker rm -f promptfoo >/dev/null
fi

# Command for running the promptfoo UI server in the background.
docker run -d \
  --name promptfoo \
  -p 3000:3000 \
  -v ~/.promptfoo:/home/promptfoo/.promptfoo \
  ghcr.io/promptfoo/promptfoo:latest