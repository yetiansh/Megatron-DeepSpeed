#!/bin/bash
DOCKER_BINARY="nvidia-docker"
DEVICE_PREFIX=/dev/infiniband
${DOCKER_BINARY} run --pid=host -it --net=host --gpus all --privileged --cap-add=IPC_LOCK --shm-size=4g --ulimit memlock=-1:-1 megatron_ds:exp bash ${COMMAND[@]}