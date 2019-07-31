FROM python:3.6.4-jessie

# Install tools
RUN apt-get update && \
    apt-get install -y wget ssh && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /src/*.deb

RUN python3.6 -m pip install --upgrade pip

WORKDIR /usr/xpctl

ARG backend="mongo"
ARG host="local"

COPY orchestration/docker/server-requirements.${backend}.txt /usr/xpctl/server-requirements.txt
COPY orchestration/secrets/xpctlcred-${backend}-${host}.yaml /usr/xpctl/xpctlcred.yaml

COPY xpctl /usr/xpctl/

# fetch baseline and install
ADD https://github.com/dpressel/baseline/archive/master.tar.gz /usr/baseline-master.tar.gz
RUN cd /usr && tar xzf baseline-master.tar.gz && \
    cd ./baseline-master/python && \
       ./install_dev.sh baseline notest


# we will phase this out eventually, but for now an old version of xpctl still exists in baseline. remove that.
RUN rm -f /usr/baseline/python/xpctl

RUN python3.6 -m pip install -e xpserver

ENV PYTHONPATH /usr/:/usr/xpctl
ENV APP_NAME=xpctlserver
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

EXPOSE ${port}


ENTRYPOINT ["xpctl-server"]
