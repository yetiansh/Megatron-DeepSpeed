# according to recommendation of lastest compatible container
# FROM nvcr.io/nvidia/pytorch:20.12-py3
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev git curl sudo wget ccache libz-dev \
                        libllvm-8-ocaml-dev libllvm8 llvm-8 llvm-8-dev             \
                        llvm-8-doc llvm-8-examples llvm-8-runtime                  \
                        clang-8 clang-tools-8 clang-8-doc libclang-common-8-dev    \
                        libclang-8-dev libclang1-8 clang-format-10                 \
                        python-clang-8 libfuzzer-8-dev lldb-8 lld-8                \
                        libc++-8-dev libc++abi-8-dev libomp-8-dev clang-tidy-8     \
                        openssh-server vim libtool autoconf libgflags-dev          \
                        libgoogle-glog-dev zip

RUN curl -s -L https://github.com/Kitware/CMake/releases/download/v3.19.2/cmake-3.19.2-Linux-x86_64.sh -o cmake.sh \
  && sh cmake.sh --skip-license --prefix=/usr/local/

ENV MOFED_VER 5.3-1.0.0.1
ENV OS_VER ubuntu18.04
ENV PLATFORM x86_64

ARG EFA_INSTALLER_VERSION=latest
ARG AWS_OFI_NCCL_VERSION=aws
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:$PATH

# install aws efa software
# it looks like EFA contradicts with nvcr.io/nvidia/pytorch:20.12-py3.
# should install when switch to nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify

## Install NCCL
RUN git clone https://github.com/NVIDIA/nccl /opt/nccl \
    && cd /opt/nccl \
    && git checkout v2.10.3-1 \
    && make -j src.build CUDA_HOME=/usr/local/cuda

# install aws ofi plugin
RUN git clone https://github.com/aws/aws-ofi-nccl.git /opt/aws-ofi-nccl \
    && cd /opt/aws-ofi-nccl \
    && git checkout ${AWS_OFI_NCCL_VERSION} \
    && ./autogen.sh \
    && ./configure --prefix=/opt/aws-ofi-nccl/install \
       --with-libfabric=/opt/amazon/efa/ \
       --with-cuda=/usr/local/cuda \
       --with-nccl=/opt/nccl/build \
       --with-mpi=/opt/amazon/openmpi/ \
    && make && make install


RUN cd /usr/local/bin && ln -s /usr/bin/python3 python

RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV HOME /root
ENV SSHDIR ${HOME}/.ssh/
RUN mkdir -p ${SSHDIR}

ADD ./resources/ssh/config ${SSHDIR}/config

RUN chmod -R 600 ${SSHDIR}* && \
    chown -R ${USER}:${USER} ${SSHDIR}

RUN rm -fr ${HOME}/.openmpi && mkdir -p ${HOME}/.openmpi
RUN chown -R ${USER}:${USER} ${HOME}/.openmpi

RUN python3 -m pip install --upgrade pip && python3 -m pip install tqdm

# Download checkpoint and datasets for model training
RUN echo "checkpoint"
RUN git clone --recursive https://github.com/yetiansh/Megatron-DeepSpeed.git /Megatron-DeepSpeed && \
    cd /Megatron-DeepSpeed && git checkout origin/exp && \
    ./ds_download_ckpt.sh && python3 download_dataset.py
RUN cd /Megatron-DeepSpeed && git fetch origin && git checkout origin/exp

##############################################################################
## Add deepspeed user
###############################################################################
# Add a deepspeed user with user id 8877
RUN useradd --create-home --uid 1000 --shell /bin/bash deepspeed
RUN usermod -aG sudo deepspeed
RUN echo "deepspeed ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# # Change to non-root privilege
USER deepspeed

ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

# Install DeepSpeed
RUN echo "checkpoint_1"
RUN git clone --recursive https://github.com/yetiansh/DeepSpeed.git ${STAGE_DIR}/DeepSpeed && \
    cd ${STAGE_DIR}/DeepSpeed && git checkout origin/exp && \
    ./install.sh --pip_sudo
RUN python -c "import deepspeed; print(deepspeed.__version__)"

# Change to root privilege
USER root

# install Tutel and python dependencies
ADD ./resources/requirements.txt /requirements.txt
RUN python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
RUN python3 -m pip install -r /requirements.txt

# install apex
ENV MAX_JOBS=32
RUN git clone https://github.com/NVIDIA/apex /apex && cd /apex && \
    pip install -v --disable-pip-version-check \
        --no-cache-dir --global-option="--cpp_ext" \
        --global-option="--cuda_ext" ./