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

COPY xplan-od-growth-analysis /src/xplan-od-growth-analysis

ADD analysis_envs /analysis_envs
RUN conda env create -f /analysis_envs/xplan-od-growth-analysis.yml
RUN conda env create -f /analysis_envs/wasserstein_tenfold_comparisons.yml

COPY omics_tools /src/omics_tools
ADD omics_tools.yml /omics_tools.yml
RUN conda env create -f /omics_tools.yml