#!/usr/bin/env bash

# Command for running the promptfoo UI server in the background.
docker run -d \
  --name promptfoo \
  -p 3000:3000 \
  -v ~/.promptfoo:/home/promptfoo/.promptfoo \
  ghcr.io/promptfoo/promptfoo:latest