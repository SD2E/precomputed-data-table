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
COPY grouped_control_prediction /src/grouped_control_prediction

COPY conda_env.yml /conda_env.yml
RUN conda env create -f /conda_env.yml