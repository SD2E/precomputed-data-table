FROM sd2e/python3:miniconda

ARG LANGUAGE=python3-miniconda
ARG VERSION=0.5.5
ARG AGAVEPY_BRANCH=develop

# Discoverable version inside the container
RUN echo "TACC.cloud Apps\nType: ${LANGUAGE}\nVersion: ${VERSION}" > /etc/apps-VERSION
# Helpful env variable
ENV APPS_VERSION=${VERSION}

RUN conda update -n base conda

ADD config.yml /config.yml
ADD src /src

# Comment each of the following analysis dependencies in conda_env.yml first
COPY xplan-od-growth-analysis /src/xplan-od-growth-analysis

ADD xplan-od-growth-analysis.yml /xplan-od-growth-analysis.yml
RUN conda env create -f /xplan-od-growth-analysis.yml

#RUN echo "source activate sd2-precomputed_data_table" > ~/.bashrc