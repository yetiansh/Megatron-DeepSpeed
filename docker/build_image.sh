#!/bin/bash
DOCKER_BUILDKIT=1 nvidia-docker build -f Dockerfile -t megatron_ds:exp .