FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV APP_ROOT="/opt/app"
ENV USER_NAME="app"
ENV USER_UID="10001"
ENV PATH="${APP_ROOT}/bin:$PATH"

RUN rm /bin/sh && ln -s /bin/bash /bin/sh ##set bash

#COPY docker/bin/ ${APP_ROOT}/bin/
# allow to run random user id (openshift compatibility)
RUN useradd -l -u ${USER_UID} -r -g 0 -d ${APP_ROOT} -s /sbin/nologin -c "${USER_NAME} user" ${USER_NAME} && \
    mkdir -p ${APP_ROOT}/bin && \
    chmod -R u+x ${APP_ROOT}/bin && \
    chgrp -R 0 ${APP_ROOT} && \
    chmod -R g=u ${APP_ROOT} /etc/passwd

RUN apt-get update && apt-get -qq -y install curl bzip2 wget 
RUN apt-get update && apt-get -qq -y install build-essential libglib2.0-0 libsm6 \
    libxext6 libxrender-dev git fontconfig debconf debconf-utils

#minio client
RUN curl -o ${APP_ROOT}/bin/mc https://dl.min.io/client/mc/release/linux-amd64/mc && \
    chmod +x ${APP_ROOT}/bin/mc

### conda
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3.7 \
    && conda update conda \
    && conda clean --all --yes
RUN conda install pip 

ENV PATH /opt/conda/bin:$PATH

## nvm

# Set debconf to run non-interactively
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections


ENV NVM_DIR ${APP_ROOT}/.nvm
RUN mkdir ${NVM_DIR}
ENV NODE_VERSION 12.16.2
# replace shell with bash so we can source files
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# install nvm
# https://github.com/creationix/nvm#install-script
RUN curl --silent -o- https://raw.githubusercontent.com/creationix/nvm/v0.31.2/install.sh | bash

# install node and npm
RUN source $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default

# add node and npm to path so the commands are available
ENV NODE_PATH $NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# confirm installation
RUN node -v
RUN npm -v
RUN chmod -R g+wx ${APP_ROOT}

WORKDIR ${APP_ROOT}/inference
COPY . .
RUN pip install -r inference_requirements.txt
RUN cd webapp && npm i && npm run build

RUN chmod -R g+wx ${WORKDIR}

ENV HOME="${APP_ROOT}/inference"







# RUN cd lib && ./make.sh
CMD ["/bin/bash"]