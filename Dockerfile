FROM nvidia/cuda:7.5-devel
MAINTAINER Christian Tinauer <christian.tinauer@neuroimaging.at>

RUN \
	apt-get update && apt-get install -y --no-install-recommends --force-yes \
		ruby \
		zlib1g-dev \
		cmake \
		libgdcm2-dev && \
	rm -rf /var/lib/apt/lists/*

RUN mkdir /workspace && mkdir /workspace/relaxometry

WORKDIR /workspace/relaxometry

ADD CMakeLists.txt CMakeLists.txt
ADD config config
ADD setup setup
#ADD setup_dependencies setup_dependencies
ADD src src
ADD dependencies/nifticlib_bin/nifticlib dependencies/nifticlib_bin/nifticlib

#ADD data data
#ADD scripts scripts

RUN cd /workspace/relaxometry && ./setup

ENV FSLOUTPUTTYPE NIFTI_GZ

CMD /bin/bash
