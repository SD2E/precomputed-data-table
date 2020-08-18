FROM sd2e/python3:miniconda

ARG LANGUAGE=python3-miniconda
ARG VERSION=0.5.5
ARG AGAVEPY_BRANCH=develop

# Discoverable version inside the container
RUN echo "TACC.cloud Apps\nType: ${LANGUAGE}\nVersion: ${VERSION}" > /etc/apps-VERSION

# Helpful env variable
ENV APPS_VERSION=${VERSION}

#RUN conda update -n base conda
RUN conda update -n base -c defaults conda

COPY config.yml /config.yml
COPY src /src

COPY pysd2cat src/pysd2cat
COPY fcs_signal_prediction /src/fcs_signal_prediction

COPY analysis_envs /analysis_envs
RUN conda env create -f /analysis_envs/fcs_signal_prediction.yml