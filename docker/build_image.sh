#!/bin/bash
nvidia-docker buildx create --use --name larger_log --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=50000000
DOCKER_BUILDKIT=1 nvidia-docker buildx build -f Dockerfile -t megatron_ds:exp .